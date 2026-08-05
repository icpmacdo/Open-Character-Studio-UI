#!/usr/bin/env python3
"""Live dashboard for a run in flight: stage timeline, training curves, eval progress.

``octt_run_status.py`` answers "where is every run" as a one-shot JSON snapshot.
This answers "what is THIS run doing right now", refreshed in the browser every
few seconds, off the same artifacts the pipeline already writes:

  manifest.json              which stages have checkpoints
  dpo_pairs.jsonl.meta.json  pair count and token total
  dpo/metrics.jsonl          the DPO curve (accuracy, margin, loss) per step
  introspection.jsonl        transcript count as it is generated
  sft/metrics.jsonl          SFT step, progress fraction, NLL
  <split cache>/*.jsonl      eval responses and judgments, per side

The eval half matters here: once a run uses ``--split-cache-dir``, its own
``eval/`` directory stays empty and per-run progress is invisible to anything
that counts rows there. This attributes cache rows to the run by model tag
(``<model>@base`` and ``<model>@<sft sampler path>``), so a shared-cache run
still shows real progress.

Read-only, stdlib only, binds to localhost:

    python3 scripts/octt_live.py                    # newest live run
    python3 scripts/octt_live.py --run runs/<dir> --port 8770

Serving a run on another machine, tunnel to it:

    ssh -N -L 8770:127.0.0.1:8770 <host>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"

# Per-side targets by scale keyword in the run dir name. Insertion order matters:
# "paper-half" must match before its "paper" substring.
EVAL_TARGETS = {"smoke": 40, "quick": 200, "paper-half": 12500, "paper": 25000}
# Transcripts the introspection stage generates, same keying.
TRANSCRIPT_TARGETS = {"smoke": 6, "quick": 120, "paper-half": 6000, "paper": 12000}
DPO_PAIR_TARGETS = {"smoke": 2, "quick": 32, "paper-half": 750, "paper": 1500}

STAGES = ("dpo_pairs", "dpo", "introspection", "sft", "eval")

# Refreshes a judgment waits for its response row to appear before being given up
# on (see _eval_progress). A same-run judgment resolves on the next pass.
PENDING_SWEEPS = 3


def _scale_target(name: str, table: dict[str, int]) -> int | None:
    low = name.lower()
    for key, target in table.items():
        if key in low:
            return target
    return None


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


class TailCache:
    """Incremental JSONL reader: re-parses only bytes appended since last call.

    A live eval appends to a cache with six figures of rows; re-parsing the whole
    file every refresh would make the dashboard the most expensive thing on the
    machine. Each entry keeps a byte offset and the caller's accumulated state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._offsets: dict[str, int] = {}
        self._state: dict[str, object] = {}

    def scan(self, path: Path, init, apply_row, scope: str = "") -> object:
        """Fold every new row of *path* into state via ``apply_row(state, row)``.

        ``scope`` names any outside input ``apply_row`` closes over. A fold is
        one-way — the offset moves past a row forever — so if that input changes,
        rows already folded were classified under the old value and cannot be
        revised. Changing the scope starts a fresh fold instead, which is what
        makes the trained-checkpoint tag safe to learn late: before SFT lands
        there is no tag, and every row scanned until then would otherwise be
        permanently uncounted.
        """
        key = f"{path}\x00{scope}"
        with self._lock:
            state = self._state.get(key)
            if state is None:
                state = init()
                self._offsets[key] = 0
            offset = self._offsets[key]
            try:
                size = path.stat().st_size
            except OSError:
                return state
            if size < offset:  # truncated/rewritten — start over
                state, offset = init(), 0
            if size > offset:
                try:
                    with open(path, "rb") as fh:
                        fh.seek(offset)
                        data = fh.read(size - offset)
                except OSError:
                    return state
                # A live writer can leave a partial final line; keep it for next pass.
                tail_nl = data.rfind(b"\n")
                if tail_nl == -1:
                    return state
                consumed = tail_nl + 1
                for line in data[:consumed].splitlines():
                    if not line.strip():
                        continue
                    try:
                        apply_row(state, json.loads(line))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                offset += consumed
            self._offsets[key] = offset
            self._state[key] = state
            return state


CACHE = TailCache()


def _count_lines(path: Path) -> int:
    def apply(state, _row):
        state[0] += 1

    return CACHE.scan(path, lambda: [0], apply)[0]


def _dpo_curve(path: Path) -> list[dict]:
    def apply(state, row):
        state.append(
            {
                "step": row.get("step"),
                "accuracy": row.get("accuracy"),
                "margin": row.get("margin"),
                "loss": row.get("dpo_loss"),
            }
        )

    return list(CACHE.scan(path, list, apply))


def _sft_curve(path: Path) -> list[dict]:
    def apply(state, row):
        state.append(
            {
                "step": row.get("step"),
                "progress": row.get("progress"),
                "nll": row.get("train_mean_nll"),
            }
        )

    return list(CACHE.scan(path, list, apply))


def _eval_progress(cache_dir: Path, base_tag: str, trained_tag: str | None) -> dict:
    """Count cached responses and judgments belonging to this run, per side.

    Judgments key on a response content hash, not a model tag, so the response
    scan builds the hash->side map the judgment scan then folds against. Where
    the base and trained checkpoints happen to emit byte-identical text the hash
    collides and the judgment is attributed to whichever side wrote it last —
    measured at ~0.2% of a 25k side, and display-only: the eval itself keys
    judgments by (response hash, trait pair, judge) and scores from its own
    schedule, so nothing downstream depends on this split.
    """
    responses = cache_dir / "responses.jsonl"
    judgments = cache_dir / "judgments.jsonl"

    def apply_response(state, row):
        tag = row.get("model_tag")
        side = None
        if tag == base_tag:
            side = "base"
        elif trained_tag and tag == trained_tag:
            side = "trained"
        if side is None:
            return
        state["counts"][side] += 1
        h = row.get("response_hash")
        if h:
            state["hashes"][h] = side

    scope = f"{base_tag}|{trained_tag}"
    resp = CACHE.scan(
        responses,
        lambda: {"counts": {"base": 0, "trained": 0}, "hashes": {}},
        apply_response,
        scope=scope,
    )

    def apply_judgment(state, row):
        h = row.get("response_hash")
        if not h:
            return
        side = resp["hashes"].get(h)
        if side:
            state[side] += 1
        else:
            # Its response row is not in the map yet (the two files flush
            # independently, so a judgment can hit disk first). Hold it briefly:
            # the scan offset moves past this row and never revisits it, so
            # dropping it here would undercount permanently. Most held hashes
            # belong to OTHER runs sharing the cache and never resolve, hence the
            # retry budget rather than an unbounded set.
            state["pending"][h] = PENDING_SWEEPS

    judged = CACHE.scan(
        judgments,
        lambda: {"base": 0, "trained": 0, "pending": {}},
        apply_judgment,
        scope=scope,
    )
    if judged["pending"]:
        still: dict[str, int] = {}
        for h, left in judged["pending"].items():
            side = resp["hashes"].get(h)
            if side:
                judged[side] += 1
            elif left > 1:
                still[h] = left - 1
        judged["pending"] = still
    return {
        "responses": dict(resp["counts"]),
        "judgments": {"base": judged["base"], "trained": judged["trained"]},
    }


def _live_out_dirs() -> set[str]:
    try:
        out = subprocess.run(
            ["pgrep", "-fl", "octt"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    dirs = set()
    for line in out.splitlines():
        if "--out " in line:
            dirs.add(line.split("--out ", 1)[1].split()[0].rstrip("/"))
    return dirs


def _split_cache_of(run_dir: Path) -> Path | None:
    """Recover the --split-cache-dir the live process was launched with."""
    try:
        out = subprocess.run(
            ["pgrep", "-fl", "octt"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    for line in out.splitlines():
        if run_dir.name in line and "--split-cache-dir " in line:
            raw = line.split("--split-cache-dir ", 1)[1].split()[0]
            path = Path(raw)
            return path if path.is_absolute() else REPO / path
    default = RUNS / "_campaign_eval_cache"
    return default if default.is_dir() else None


def _log_tail(log_path: Path | None, lines: int = 14) -> list[str]:
    if not log_path or not log_path.is_file():
        return []
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            block = min(size, 64_000)
            fh.seek(size - block)
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    out = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return out[-lines:]


def _newest_live_run() -> Path | None:
    live = _live_out_dirs()
    candidates = []
    for d in live:
        p = Path(d)
        p = p if p.is_absolute() else REPO / p
        if p.is_dir():
            candidates.append(p)
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    runs = [p for p in RUNS.iterdir() if p.is_dir() and (p / "manifest.json").is_file()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def collect(run_dir: Path, log_path: Path | None, compare_dir: Path | None) -> dict:
    manifest = _read_json(run_dir / "manifest.json") or {}
    stages = manifest.get("stages", {})
    model = manifest.get("model") or ""
    name = run_dir.name

    pair_meta = _read_json(run_dir / "dpo_pairs.jsonl.meta.json") or {}
    intro_meta = _read_json(run_dir / "introspection.jsonl.meta.json") or {}
    dpo_curve = _dpo_curve(run_dir / "dpo" / "metrics.jsonl")
    sft_curve = _sft_curve(run_dir / "sft" / "metrics.jsonl")

    intro_path = run_dir / "introspection.jsonl"
    transcripts = _count_lines(intro_path) if intro_path.is_file() else 0

    trained_sampler = (stages.get("sft") or {}).get("sampler_path")
    cache_dir = _split_cache_of(run_dir)
    eval_target = _scale_target(name, EVAL_TARGETS)
    if cache_dir and cache_dir.is_dir():
        evalp = _eval_progress(
            cache_dir,
            f"{model}@base",
            f"{model}@{trained_sampler}" if trained_sampler else None,
        )
        eval_source = str(cache_dir.relative_to(REPO)) if REPO in cache_dir.parents else str(cache_dir)
    else:
        evalp = {
            "responses": {},
            "judgments": {
                "base": _count_lines(run_dir / "eval" / "base_judge.jsonl"),
                "trained": _count_lines(run_dir / "eval" / "trained_judge.jsonl"),
            },
        }
        eval_source = f"{name}/eval"

    results = _read_json(run_dir / "eval_results.json")
    shift = (results or {}).get("shift_summary") or {}

    live_dirs = _live_out_dirs()
    is_live = any(name == Path(d).name for d in live_dirs)

    # Stage states. A stage is done when the manifest carries its checkpoint (or,
    # for data-generation stages, when the sidecar is complete); active when its
    # artifacts are growing and the stage after it has not started.
    pair_target = _scale_target(name, DPO_PAIR_TARGETS)
    transcript_target = _scale_target(name, TRANSCRIPT_TARGETS)
    pairs = pair_meta.get("num_pairs", 0)

    def state_of(stage: str) -> str:
        if stage == "dpo_pairs":
            return "done" if pairs else ("active" if is_live else "pending")
        if stage == "dpo":
            if "dpo" in stages:
                return "done"
            return "active" if dpo_curve else "pending"
        if stage == "introspection":
            if intro_meta:
                return "done"
            if "dpo" in stages and "sft" not in stages:
                return "active" if transcripts or is_live else "pending"
            return "done" if transcripts else "pending"
        if stage == "sft":
            if "sft" in stages:
                return "done"
            return "active" if sft_curve else "pending"
        if results:
            return "done"
        return "active" if ("sft" in stages and is_live) else "pending"

    stage_rows = []
    for stage in STAGES:
        st = state_of(stage)
        done_n = total_n = None
        if stage == "dpo_pairs":
            done_n, total_n = pairs, pair_target
        elif stage == "dpo":
            done_n = len(dpo_curve)
            total_n = (
                int(-(-pairs // 32)) if pairs else None
            )  # batch_size 32, one epoch
        elif stage == "introspection":
            # introspection.jsonl rows are TRAINING EXAMPLES, not transcripts: a
            # self-chat yields one row per assistant turn. Only the sidecar knows
            # the transcript count, so while generating, the row count is reported
            # without a denominator rather than against a transcript target it
            # does not measure.
            if intro_meta:
                done_n = intro_meta.get("num_transcripts", transcripts)
                total_n = (intro_meta.get("self_reflection") or 0) + (
                    intro_meta.get("self_interaction") or 0
                ) or transcript_target
            else:
                done_n, total_n = transcripts, None
        elif stage == "sft":
            done_n = len(sft_curve)
            last = sft_curve[-1] if sft_curve else None
            if last and last.get("progress"):
                total_n = round(len(sft_curve) / max(last["progress"], 1e-9))
        else:
            # The trained side only: the base half is banked across the campaign
            # and already complete, so counting it here would show an eval that
            # has not started as half done.
            done_n = evalp["judgments"].get("trained", 0)
            total_n = eval_target
        stage_rows.append(
            {"stage": stage, "state": st, "done": done_n, "total": total_n}
        )

    compare = None
    if compare_dir and compare_dir.is_dir():
        cmp_curve = _dpo_curve(compare_dir / "dpo" / "metrics.jsonl")
        cmp_results = _read_json(compare_dir / "eval_results.json") or {}
        cmp_manifest = _read_json(compare_dir / "manifest.json") or {}
        if cmp_curve:
            compare = {
                "run": compare_dir.name,
                "model": cmp_manifest.get("model"),
                "dpo_curve": cmp_curve,
                "net_shift": round(
                    ((cmp_results.get("shift_summary") or {}).get("net_shift") or 0.0), 1
                )
                or None,
            }

    started = manifest.get("created_at")
    return {
        "generated_unix": time.time(),
        "run": name,
        "model": model,
        "persona": manifest.get("persona"),
        "teacher": manifest.get("teacher"),
        "execution_mode": manifest.get("execution_mode"),
        "live": is_live,
        "started_unix": started,
        "elapsed_sec": (time.time() - started) if started else None,
        "stages": stage_rows,
        "pairs": {
            "count": pairs,
            "target": pair_target,
            "tokens": pair_meta.get("pair_tokens"),
            "budget": pair_meta.get("token_budget"),
        },
        "dpo_curve": dpo_curve,
        "introspection": {
            "transcripts": transcripts,
            "target": transcript_target,
            "meta": intro_meta or None,
        },
        "sft_curve": sft_curve,
        "eval": {
            "responses": evalp["responses"],
            "judgments": evalp["judgments"],
            "target_per_side": eval_target,
            "source": eval_source,
        },
        "result": {
            "net_shift": round(shift["net_shift"], 1) if shift.get("net_shift") is not None else None,
            "aligned": round(shift["aligned_mean_delta"], 1) if shift.get("aligned_mean_delta") is not None else None,
            "opposing": round(shift["opposing_mean_delta"], 1) if shift.get("opposing_mean_delta") is not None else None,
            "top_increased": shift.get("top_increased"),
            "top_decreased": shift.get("top_decreased"),
        }
        if results
        else None,
        "compare": compare,
        "log": _log_tail(log_path),
    }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>octt live — __RUN__</title>
<style>
  :root {
    color-scheme: light;
    --surface-0: #f4f4f2;
    --surface-1: #fcfcfb;
    --border:    #e0dfda;
    --text-1:    #0b0b0b;
    --text-2:    #52514e;
    --text-3:    #86847d;
    --s1: #2a78d6;  /* blue   — this run */
    --s2: #eb6834;  /* orange — comparison run */
    --s3: #1baf7a;  /* aqua   — secondary measure */
    --good: #0ca30c; --warning: #fab219; --critical: #d03b3b;
    --grid: #ebeae6;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-0: #111110; --surface-1: #1a1a19; --border: #33332f;
      --text-1: #ffffff; --text-2: #c3c2b7; --text-3: #8a8980;
      --s1: #3987e5; --s2: #d95926; --s3: #199e70;
      --good: #0ca30c; --warning: #fab219; --critical: #d03b3b;
      --grid: #2a2a27;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--surface-0); color: var(--text-1);
    font: 14px/1.5 ui-sans-serif, -apple-system, "SF Pro Text", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1140px; margin: 0 auto; padding: 28px 20px 60px; }
  header { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin-bottom: 4px; }
  h1 { font-size: 19px; font-weight: 640; margin: 0; letter-spacing: -0.01em; }
  .sub { color: var(--text-2); font-size: 13px; }
  .mono { font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace; }
  .pill {
    display: inline-flex; align-items: center; gap: 6px; font-size: 12px;
    padding: 2px 9px; border-radius: 999px; border: 1px solid var(--border);
    background: var(--surface-1); color: var(--text-2);
  }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--text-3); }
  .dot.live { background: var(--good); animation: pulse 1.8s ease-in-out infinite; }
  .dot.stopped { background: var(--critical); }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
  @media (prefers-reduced-motion: reduce) { .dot.live { animation: none; } }

  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); gap: 12px; margin: 20px 0 22px; }
  .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 13px 15px; }
  .card .label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--text-3); }
  .card .value { font-size: 25px; font-weight: 620; letter-spacing: -0.02em; margin-top: 3px; font-variant-numeric: tabular-nums; }
  .card .note { font-size: 12px; color: var(--text-2); margin-top: 2px; }

  section { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; margin-bottom: 14px; }
  h2 { font-size: 13px; font-weight: 620; margin: 0 0 3px; letter-spacing: -0.005em; }
  .hint { font-size: 12px; color: var(--text-3); margin: 0 0 14px; }

  .stage { display: grid; grid-template-columns: 128px 1fr 132px; gap: 12px; align-items: center; padding: 7px 0; }
  .stage + .stage { border-top: 1px solid var(--grid); }
  .stage .nm { display: flex; align-items: center; gap: 7px; font-size: 13px; }
  .stage .sq { width: 8px; height: 8px; border-radius: 2px; flex: none; }
  .sq.done { background: var(--good); } .sq.active { background: var(--s1); } .sq.pending { background: var(--text-3); opacity:.4; }
  .track { height: 7px; border-radius: 4px; background: var(--grid); overflow: hidden; }
  .fill { height: 100%; border-radius: 4px; background: var(--s1); transition: width .4s ease; }
  .fill.done { background: var(--good); }
  .stage .n { text-align: right; font-size: 12px; color: var(--text-2); font-variant-numeric: tabular-nums; }

  .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(268px, 1fr)); gap: 16px; }
  .chart .ct { font-size: 12px; color: var(--text-2); margin-bottom: 2px; display: flex; justify-content: space-between; align-items: baseline; }
  .chart .cv { font-size: 15px; font-weight: 600; color: var(--text-1); font-variant-numeric: tabular-nums; }
  svg { display: block; width: 100%; height: auto; overflow: visible; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--text-2); margin-bottom: 10px; }
  .legend i { display: inline-block; width: 16px; height: 2px; vertical-align: middle; margin-right: 6px; border-radius: 2px; }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 5px 8px 5px 0; border-bottom: 1px solid var(--grid); }
  th { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: var(--text-3); font-weight: 560; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  details { margin-top: 10px; } summary { cursor: pointer; font-size: 12px; color: var(--text-2); }
  pre.log { margin: 0; padding: 12px 14px; background: var(--surface-0); border: 1px solid var(--border);
    border-radius: 8px; font-size: 11.5px; line-height: 1.55; color: var(--text-2); overflow-x: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre; }
  .tip { position: fixed; pointer-events: none; z-index: 9; background: var(--surface-1); color: var(--text-1);
    border: 1px solid var(--border); border-radius: 7px; padding: 6px 9px; font-size: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,.13); opacity: 0; transition: opacity .12s; font-variant-numeric: tabular-nums; }
  footer { color: var(--text-3); font-size: 12px; margin-top: 18px; display: flex; gap: 14px; flex-wrap: wrap; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1 id="run">—</h1>
    <span class="pill"><span class="dot" id="livedot"></span><span id="livetxt">connecting</span></span>
    <span class="sub" id="ident"></span>
  </header>
  <div class="sub" id="subline"></div>

  <div class="cards" id="cards"></div>

  <section>
    <h2>Pipeline stages</h2>
    <p class="hint" id="stagehint">Each stage's own artifacts, counted as they are written.</p>
    <div id="stages"></div>
  </section>

  <section id="dposec">
    <h2>DPO — preference distillation</h2>
    <p class="hint">Accuracy and margin on the training batch, per optimizer step. Separate panels: the two measures share no scale.</p>
    <div class="legend" id="dpolegend"></div>
    <div class="charts" id="dpocharts"></div>
  </section>

  <section id="sftsec" hidden>
    <h2>SFT — introspection training</h2>
    <p class="hint">Mean NLL over the self-chat corpus, per step.</p>
    <div class="charts" id="sftcharts"></div>
  </section>

  <section id="evalsec">
    <h2>Revealed-preference eval</h2>
    <p class="hint" id="evalhint"></p>
    <div id="evalbars"></div>
    <div id="resultbox"></div>
  </section>

  <section>
    <h2>Run log</h2>
    <p class="hint" id="logpath"></p>
    <pre class="log" id="log">—</pre>
  </section>

  <footer>
    <span id="stamp"></span>
    <span>refreshes every 5s</span>
    <span class="mono">scripts/octt_live.py</span>
  </footer>
</div>
<div class="tip" id="tip"></div>

<script>
const $ = (id) => document.getElementById(id);
const tip = $("tip");
const fmt = (n, d = 0) => n === null || n === undefined ? "—" : Number(n).toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d});
const pct = (a, b) => (b ? Math.min(100, 100 * a / b) : 0);

function dur(sec) {
  if (sec === null || sec === undefined) return "—";
  const s = Math.max(0, Math.floor(sec)), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}m ${String(s % 60).padStart(2, "0")}s`;
}
const STAGE_LABEL = {dpo_pairs: "DPO pairs", dpo: "DPO training", introspection: "Introspection", sft: "SFT training", eval: "Eval"};

// ---- line chart: one measure, one axis, crosshair + tooltip -----------------
function lineChart(series, opts) {
  // A multi-series panel reserves a right gutter for direct labels, so they never
  // land on top of the curves they name.
  const W = 520, H = 132, P = {t: 10, r: series.length > 1 ? 52 : 10, b: 20, l: 34};
  const all = series.flatMap(s => s.points);
  if (!all.length) return `<svg viewBox="0 0 ${W} ${H}"></svg>`;
  const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  let y0 = opts.y0 !== undefined ? opts.y0 : Math.min(...ys);
  let y1 = opts.y1 !== undefined ? opts.y1 : Math.max(...ys);
  if (y1 - y0 < 1e-9) { y1 = y0 + 1; }
  const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;
  const X = v => P.l + (W - P.l - P.r) * (v - x0) / (x1 - x0);
  const Y = v => P.t + (H - P.t - P.b) * (1 - (v - y0) / (y1 - y0));

  let g = "";
  const ticks = [y0 + (y1 - y0) * 0.5, y1 - pad, y0 + pad];
  for (const t of [ticks[1], ticks[0], ticks[2]]) {
    g += `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(t).toFixed(1)}" y2="${Y(t).toFixed(1)}" stroke="var(--grid)" stroke-width="1"/>`;
    g += `<text x="${P.l - 6}" y="${(Y(t) + 3.5).toFixed(1)}" text-anchor="end" font-size="10" fill="var(--text-3)">${(+t.toFixed(opts.dec ?? 1))}</text>`;
  }
  let paths = "", labels = "";
  const ends = [];
  series.forEach((s) => {
    if (!s.points.length) return;
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    paths += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    const last = s.points[s.points.length - 1];
    paths += `<circle cx="${X(last[0]).toFixed(1)}" cy="${Y(last[1]).toFixed(1)}" r="3.5" fill="${s.color}" stroke="var(--surface-1)" stroke-width="2"/>`;
    ends.push({s, x: X(last[0]), y: Y(last[1])});
  });
  // Direct labels, pushed apart when the curves converge — series that end on the
  // same value (accuracy pinned at 1.0, loss at 0) would otherwise print on top
  // of each other, which is exactly where the reader most needs to tell them apart.
  if (ends.length > 1) {
    const sorted = [...ends].sort((a, b) => a.y - b.y);
    let prev = -1e9;
    for (const e of sorted) {
      let ly = Math.max(e.y + 3.5, prev + 12);
      if (ly < P.t + 8) ly = P.t + 8;
      if (ly > H - P.b) ly = H - P.b;
      prev = ly;
      labels += `<text x="${(W - P.r + 7).toFixed(1)}" y="${ly.toFixed(1)}" font-size="10.5" font-weight="600" fill="${e.s.color}">${e.s.label}</text>`;
    }
  }
  const axis = `<text x="${P.l}" y="${H - 5}" font-size="10" fill="var(--text-3)">step ${x0}</text>` +
               `<text x="${W - P.r}" y="${H - 5}" text-anchor="end" font-size="10" fill="var(--text-3)">${x1}</text>`;
  const meta = encodeURIComponent(JSON.stringify({x0, x1, y0, y1, P, W, H, series: series.map(s => ({label: s.label, color: s.color, points: s.points})), dec: opts.dec ?? 2, unit: opts.unit || ""}));
  return `<svg viewBox="0 0 ${W} ${H}" data-chart="${meta}" role="img" aria-label="${opts.aria || ""}">${g}${paths}${labels}${axis}
    <line class="cross" x1="0" x2="0" y1="${P.t}" y2="${H - P.b}" stroke="var(--text-3)" stroke-width="1" opacity="0"/>
    <rect x="${P.l}" y="${P.t}" width="${W - P.l - P.r}" height="${H - P.t - P.b}" fill="transparent"/></svg>`;
}

function wireTips(root) {
  root.querySelectorAll("svg[data-chart]").forEach(svg => {
    const m = JSON.parse(decodeURIComponent(svg.dataset.chart));
    const cross = svg.querySelector(".cross");
    const X = v => m.P.l + (m.W - m.P.l - m.P.r) * (v - m.x0) / (m.x1 - m.x0);
    svg.addEventListener("pointermove", (e) => {
      const box = svg.getBoundingClientRect();
      const sx = (e.clientX - box.left) / box.width * m.W;
      const step = Math.round(m.x0 + (m.x1 - m.x0) * (sx - m.P.l) / (m.W - m.P.l - m.P.r));
      const rows = m.series.map(s => {
        let best = null, bd = 1e9;
        for (const p of s.points) { const d = Math.abs(p[0] - step); if (d < bd) { bd = d; best = p; } }
        return best ? `<div><span style="display:inline-block;width:9px;height:2px;background:${s.color};vertical-align:middle;margin-right:6px"></span>${s.label}: <b>${(+best[1].toFixed(m.dec))}${m.unit}</b></div>` : "";
      }).join("");
      if (!rows) return;
      cross.setAttribute("x1", X(step)); cross.setAttribute("x2", X(step)); cross.setAttribute("opacity", ".45");
      tip.innerHTML = `<div style="color:var(--text-3);margin-bottom:2px">step ${step}</div>${rows}`;
      tip.style.opacity = 1;
      tip.style.left = Math.min(window.innerWidth - 170, e.clientX + 14) + "px";
      tip.style.top = (e.clientY - 12) + "px";
    });
    svg.addEventListener("pointerleave", () => { tip.style.opacity = 0; cross.setAttribute("opacity", 0); });
  });
}

function stageRow(s) {
  const label = STAGE_LABEL[s.stage] || s.stage;
  const p = s.state === "done" && !s.total ? 100 : pct(s.done || 0, s.total || 0);
  const n = s.total ? `${fmt(s.done)} / ${fmt(s.total)}` : (s.done ? fmt(s.done) : (s.state === "done" ? "done" : "—"));
  return `<div class="stage">
    <div class="nm"><span class="sq ${s.state}"></span>${label}</div>
    <div class="track"><div class="fill ${s.state === "done" ? "done" : ""}" style="width:${(s.state === "done" ? 100 : p).toFixed(1)}%"></div></div>
    <div class="n">${n}</div></div>`;
}

function render(d) {
  $("run").textContent = d.run;
  $("livedot").className = "dot " + (d.live ? "live" : (d.result ? "" : "stopped"));
  $("livetxt").textContent = d.live ? "running" : (d.result ? "complete" : "no live process");
  $("ident").textContent = `${d.persona} · ${d.model}`;
  $("subline").innerHTML = `<span class="sub">teacher ${d.teacher || "—"} · mode ${d.execution_mode || "—"}</span>`;

  const active = (d.stages.find(s => s.state === "active") || {}).stage;
  const evalDone = d.eval.judgments.trained || 0;
  const evalTot = d.eval.target_per_side;
  const lastDpo = d.dpo_curve.length ? d.dpo_curve[d.dpo_curve.length - 1] : null;
  $("cards").innerHTML = [
    `<div class="card"><div class="label">Stage</div><div class="value" style="font-size:19px">${STAGE_LABEL[active] || (d.result ? "Complete" : "—")}</div><div class="note">${d.live ? "in progress" : (d.result ? "finished" : "stalled")}</div></div>`,
    `<div class="card"><div class="label">Elapsed</div><div class="value">${dur(d.elapsed_sec)}</div><div class="note">since launch</div></div>`,
    `<div class="card"><div class="label">DPO steps</div><div class="value">${fmt(d.dpo_curve.length)}</div><div class="note">${lastDpo ? `margin ${(+lastDpo.margin.toFixed(1))} · acc ${(+lastDpo.accuracy.toFixed(2))}` : "not started"}</div></div>`,
    `<div class="card"><div class="label">Introspection</div><div class="value">${fmt(d.introspection.meta ? d.introspection.meta.num_transcripts : d.introspection.transcripts)}</div><div class="note">${d.introspection.meta ? `transcripts kept · ${fmt(d.introspection.meta.transcripts_dropped_for_budget || 0)} dropped` : (d.introspection.transcripts ? "training rows so far" : "not started")}</div></div>`,
    `<div class="card"><div class="label">Judgments</div><div class="value">${fmt(evalDone)}</div><div class="note">${evalTot ? `of ${fmt(evalTot)} · trained side` : "—"}</div></div>`,
    d.result
      ? `<div class="card"><div class="label">Net shift</div><div class="value" style="color:var(--good)">+${fmt(d.result.net_shift, 1)}</div><div class="note">aligned ${fmt(d.result.aligned, 1)} · opposing ${fmt(d.result.opposing, 1)}</div></div>`
      : `<div class="card"><div class="label">Net shift</div><div class="value" style="color:var(--text-3)">—</div><div class="note">awaits eval</div></div>`,
  ].join("");

  $("stages").innerHTML = d.stages.map(stageRow).join("");
  $("stagehint").textContent = d.pairs.tokens
    ? `${fmt(d.pairs.count)} DPO pairs · ${fmt(d.pairs.tokens)} tokens against a ${fmt(d.pairs.budget)} budget.`
    : "Each stage's own artifacts, counted as they are written.";

  // DPO — small multiples, never a dual axis.
  const mine = d.dpo_curve;
  const cmp = d.compare;
  $("dpolegend").innerHTML = cmp
    ? `<span><i style="background:var(--s1)"></i>${d.run} <span style="color:var(--text-3)">(${d.model})</span></span>
       <span><i style="background:var(--s2)"></i>${cmp.run} <span style="color:var(--text-3)">(${cmp.model})</span></span>`
    : `<span><i style="background:var(--s1)"></i>${d.run}</span>`;
  const mk = (key, title, opts) => {
    const series = [{label: "this run", color: "var(--s1)", points: mine.map(r => [r.step, r[key]]).filter(p => p[1] != null)}];
    if (cmp) series.push({label: "prior", color: "var(--s2)", points: cmp.dpo_curve.map(r => [r.step, r[key]]).filter(p => p[1] != null)});
    const last = series[0].points.length ? series[0].points[series[0].points.length - 1][1] : null;
    return `<div class="chart"><div class="ct"><span>${title}</span><span class="cv">${last === null ? "—" : (+last.toFixed(opts.dec ?? 2))}</span></div>${lineChart(series, opts)}</div>`;
  };
  $("dpocharts").innerHTML = mine.length
    ? mk("accuracy", "Preference accuracy", {y0: 0, y1: 1, dec: 2, aria: "DPO preference accuracy per step"}) +
      mk("margin", "Reward margin", {y0: 0, dec: 1, aria: "DPO reward margin per step"}) +
      mk("loss", "DPO loss", {y0: 0, dec: 3, aria: "DPO loss per step"})
    : `<p class="hint">No DPO steps yet.</p>`;

  // SFT
  const sft = d.sft_curve;
  $("sftsec").hidden = !sft.length;
  if (sft.length) {
    const last = sft[sft.length - 1];
    $("sftcharts").innerHTML =
      `<div class="chart"><div class="ct"><span>Mean NLL</span><span class="cv">${(+last.nll.toFixed(3))}</span></div>` +
      lineChart([{label: "nll", color: "var(--s1)", points: sft.map(r => [r.step, r.nll]).filter(p => p[1] != null)}], {dec: 3, aria: "SFT mean NLL per step"}) + `</div>` +
      `<div class="chart"><div class="ct"><span>Corpus progress</span><span class="cv">${(100 * (last.progress || 0)).toFixed(1)}%</span></div>` +
      `<div class="track" style="height:9px;margin-top:8px"><div class="fill" style="width:${(100 * (last.progress || 0)).toFixed(1)}%"></div></div>` +
      `<p class="hint" style="margin-top:9px">step ${fmt(last.step)} · one pass over the introspection corpus</p></div>`;
  }

  // Eval
  const ev = d.eval, tgt = ev.target_per_side;
  $("evalhint").innerHTML = `Judgments attributed by model tag from <span class="mono">${ev.source}</span>. The base side is banked across the campaign — it does not re-pay.`;
  // Banked rows are complete by definition and can exceed this run's target (the
  // campaign cache holds judgments from more than one schedule), so they show a
  // count rather than a ratio that would read as over 100%.
  const bar = (name, done, total, banked) => `<div class="stage">
      <div class="nm"><span class="sq ${banked || (total && done >= total) ? "done" : (done ? "active" : "pending")}"></span>${name}</div>
      <div class="track"><div class="fill ${banked || (total && done >= total) ? "done" : ""}" style="width:${banked ? 100 : pct(done, total).toFixed(1)}%"></div></div>
      <div class="n">${fmt(done)}${banked ? ` <span style="color:var(--text-3)">banked</span>` : (total ? ` / ${fmt(total)}` : "")}</div></div>`;
  $("evalbars").innerHTML =
    bar("Base responses", ev.responses.base || 0, tgt, true) +
    bar("Base judgments", ev.judgments.base || 0, tgt, true) +
    bar("Trained responses", ev.responses.trained || 0, tgt, false) +
    bar("Trained judgments", ev.judgments.trained || 0, tgt, false);

  $("resultbox").innerHTML = d.result && d.result.top_increased
    ? `<details open><summary>Trait deltas</summary><table><thead><tr><th>Risen</th><th class="num">Δ Elo</th><th>Fallen</th><th class="num">Δ Elo</th></tr></thead><tbody>` +
      d.result.top_increased.slice(0, 5).map((t, i) => {
        const f = (d.result.top_decreased || [])[i] || {};
        return `<tr><td>${t.trait}</td><td class="num" style="color:var(--good)">+${fmt(t.delta, 0)}</td><td>${f.trait || ""}</td><td class="num" style="color:var(--critical)">${f.delta ? fmt(f.delta, 0) : ""}</td></tr>`;
      }).join("") + `</tbody></table></details>`
    : "";

  $("log").textContent = d.log.length ? d.log.join("\n") : "—";
  $("stamp").textContent = "updated " + new Date(d.generated_unix * 1000).toLocaleTimeString();
  wireTips(document);
}

async function tick() {
  try {
    const r = await fetch("/api/status", {cache: "no-store"});
    render(await r.json());
  } catch (e) {
    $("livetxt").textContent = "server unreachable";
    $("livedot").className = "dot stopped";
  }
}
tick(); setInterval(tick, 5000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    run_dir: Path
    log_path: Path | None
    compare_dir: Path | None

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        try:
            if self.path.startswith("/api/status"):
                payload = collect(self.run_dir, self.log_path, self.compare_dir)
                self._send(200, json.dumps(payload).encode(), "application/json")
            elif self.path in ("/", "/index.html"):
                page = PAGE.replace("__RUN__", self.run_dir.name)
                self._send(200, page.encode(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")
        except BrokenPipeError:
            pass
        except Exception:  # noqa: BLE001 — a bad refresh must not kill the server
            import traceback

            self._send(500, traceback.format_exc().encode(), "text/plain")

    def log_message(self, *args) -> None:  # quiet
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run directory (default: the newest live run)")
    ap.add_argument("--log", help="log file to tail (default: newest in runs/octt-plan-logs)")
    ap.add_argument("--compare", help="a finished run to overlay on the DPO curves")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = ap.parse_args(argv)

    run_dir = Path(args.run) if args.run else _newest_live_run()
    if run_dir is None:
        print("no runs found", file=sys.stderr)
        return 2
    run_dir = run_dir if run_dir.is_absolute() else REPO / run_dir
    if not (run_dir / "manifest.json").is_file():
        print(f"{run_dir} has no manifest.json", file=sys.stderr)
        return 2

    if args.log:
        log_path = Path(args.log)
    else:
        logs = sorted((RUNS / "octt-plan-logs").glob("*.log"), key=lambda p: p.stat().st_mtime)
        log_path = logs[-1] if logs else None

    compare_dir = None
    if args.compare:
        compare_dir = Path(args.compare)
        compare_dir = compare_dir if compare_dir.is_absolute() else REPO / compare_dir

    Handler.run_dir = run_dir
    Handler.log_path = log_path
    Handler.compare_dir = compare_dir

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"octt live — {run_dir.name}")
    print(f"  {url}")
    if log_path:
        print(f"  tailing {log_path}")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
