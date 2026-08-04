"""W2 qualitative grid: fixed neutral prompt panel x checkpoint targets (B3).

Work package 1 of the readiness doc: a frozen, hashed panel of user-only
prompts is sampled greedily against every trained checkpoint and each unique
base control — no embody prompt, no system prompt at all — so the grid shows a
model's *default* character rather than behavior under trait pressure. The
panel itself (``data/qualitative_panels/w2-pirate-v1.json``) is frozen and
safety-reviewed separately (B4); this module is the machinery:

  - immutable, validated, content-hashed panel and target definitions;
  - manifest-backed checkpoint resolution (read-only — never creates or
    mutates a run manifest);
  - neutral message construction and deterministic per-cell request ids
    (:mod:`octt.artifacts` contract);
  - free dry-run cost projection;
  - resumable response shards (local or remote hosts) with an execution-mode
    guard, and a conflict-refusing merge into one canonical grid artifact;
  - Markdown summary + full HTML grid renderers (JSONL stays the source of
    truth; the renders are reading surfaces);
  - a separate extractor for banked embody-conditioned eval responses, which
    measure "behavior under trait pressure" and are labeled and kept apart —
    never merged into the neutral grid.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import artifacts, instruments, models
from .manifest import MANIFEST_BASE_NAME, StageCheckpoint

PANEL_SCHEMA_VERSION = 1
DEFAULT_INSTRUMENT_ID = "qualitative/w2-pirate-v1-greedy"

CATEGORIES = ("trait_open", "technical", "non_english", "instruction_conflict")
ROLES = ("trained", "base")

EXECUTION_MODE_DRY_RUN = "dry-run"
EXECUTION_MODE_REAL = "real"

# The estimand label stamped onto every banked-embody extraction; the canonical
# neutral grid and this view must never be presented as the same measurement.
BANKED_EMBODY_ESTIMAND = "behavior under trait pressure"


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelPrompt:
    prompt_id: str
    text: str
    language: str  # e.g. "en", "fr", "zh-Hans"
    category: str  # one of CATEGORIES
    secondary_tags: tuple[str, ...] = ()
    provenance: str = ""
    rationale: str = ""
    publishable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "text": self.text,
            "language": self.language,
            "category": self.category,
            "secondary_tags": list(self.secondary_tags),
            "provenance": self.provenance,
            "rationale": self.rationale,
            "publishable": self.publishable,
        }


@dataclass(frozen=True)
class Panel:
    """A frozen prompt panel. Prompt order is part of the identity (hashed)."""

    panel_id: str
    version: str
    quotas: Mapping[str, int]
    prompts: tuple[PanelPrompt, ...]

    @property
    def content_hash(self) -> str:
        return artifacts.content_hash(
            {
                "schema_version": PANEL_SCHEMA_VERSION,
                "panel_id": self.panel_id,
                "version": self.version,
                "quotas": dict(self.quotas),
                "prompts": [p.to_dict() for p in self.prompts],
            }
        )

    def validate(self) -> None:
        if not self.panel_id or not self.version:
            raise ValueError("panel needs a panel_id and a version")
        ids = [p.prompt_id for p in self.prompts]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate prompt ids: {', '.join(dupes)}")
        unknown_quota = set(self.quotas) - set(CATEGORIES)
        if unknown_quota:
            raise ValueError(f"quota for unknown categories: {sorted(unknown_quota)}")
        counts: dict[str, int] = dict.fromkeys(CATEGORIES, 0)
        for p in self.prompts:
            if not p.text.strip():
                raise ValueError(f"prompt {p.prompt_id} has empty text")
            if not p.language.strip():
                raise ValueError(f"prompt {p.prompt_id} has no language tag")
            if p.category not in CATEGORIES:
                raise ValueError(
                    f"prompt {p.prompt_id} has unknown category {p.category!r}; "
                    f"expected one of {', '.join(CATEGORIES)}"
                )
            counts[p.category] += 1
        declared = {c: int(self.quotas.get(c, 0)) for c in CATEGORIES}
        actual = {c: counts[c] for c in CATEGORIES}
        if declared != actual:
            raise ValueError(
                f"category quotas do not match the panel: declared {declared}, "
                f"actual {actual}. The quotas are locked; fix the panel or bump "
                "its version."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PANEL_SCHEMA_VERSION,
            "panel_id": self.panel_id,
            "version": self.version,
            "quotas": dict(self.quotas),
            "prompts": [p.to_dict() for p in self.prompts],
        }


def panel_from_dict(data: Mapping[str, Any]) -> Panel:
    if data.get("schema_version") != PANEL_SCHEMA_VERSION:
        raise ValueError(
            f"panel schema_version {data.get('schema_version')!r} != {PANEL_SCHEMA_VERSION}"
        )
    prompts = tuple(
        PanelPrompt(
            prompt_id=p["prompt_id"],
            text=p["text"],
            language=p["language"],
            category=p["category"],
            secondary_tags=tuple(p.get("secondary_tags", ())),
            provenance=p.get("provenance", ""),
            rationale=p.get("rationale", ""),
            publishable=bool(p.get("publishable", True)),
        )
        for p in data["prompts"]
    )
    panel = Panel(
        panel_id=data["panel_id"],
        version=data["version"],
        quotas=dict(data.get("quotas", {})),
        prompts=prompts,
    )
    panel.validate()
    return panel


def load_panel(path: Path) -> Panel:
    return panel_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Targets (manifest-backed, read-only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """One column of the grid: a checkpoint (or base control) to sample."""

    alias: str  # public name used in reports, e.g. "pirate-4B"
    base_model: str
    role: str  # "trained" | "base"
    fingerprint: str  # tinker:// sampler URI, or "base"
    sampler_path: str | None = None
    execution_mode: str | None = None  # from the run manifest, None for base


def resolve_targets(
    specs: Sequence[Mapping[str, Any]], runs_root: Path = Path("runs")
) -> tuple[Target, ...]:
    """Resolve target specs against run manifests, read-only.

    A spec is either a base control::

        {"alias": "4B-base", "base_model": "Qwen/Qwen3.5-4B", "role": "base"}

    or a trained checkpoint resolved from a run directory's manifest::

        {"alias": "pirate-4B", "base_model": "Qwen/Qwen3.5-4B",
         "run_dir": "pirate-...-v6", "stage": "sft"}

    ``run_dir`` is relative to *runs_root* (or absolute). ``stage`` defaults to
    ``sft`` (the final on-Tinker sampler; the merge stage's artifact is a local
    adapter that cannot be sampled via Tinker). The manifest is read directly —
    never through ``load_or_create`` — so resolution can never create or mutate
    run state. The manifest's model must match the spec's ``base_model``:
    checkpoints are not interchangeable across bases.
    """
    targets: list[Target] = []
    for spec in specs:
        alias = spec.get("alias") or ""
        base_model = spec.get("base_model") or ""
        if not alias or not base_model:
            raise ValueError(f"target spec needs alias and base_model: {spec!r}")
        role = spec.get("role", "trained" if spec.get("run_dir") else "base")
        if role not in ROLES:
            raise ValueError(f"target {alias!r} has unknown role {role!r}")
        if role == "base":
            targets.append(
                Target(alias=alias, base_model=base_model, role="base", fingerprint="base")
            )
            continue
        run_dir = Path(spec["run_dir"])
        if not run_dir.is_absolute():
            run_dir = Path(runs_root) / run_dir
        manifest_path = run_dir / MANIFEST_BASE_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"target {alias!r}: no manifest at {manifest_path}")
        data = json.loads(manifest_path.read_text())
        if data.get("model") != base_model:
            raise ValueError(
                f"target {alias!r}: manifest at {manifest_path} is for model "
                f"{data.get('model')!r}, spec says {base_model!r}"
            )
        stage = spec.get("stage", "sft")
        record = data.get("stages", {}).get(stage)
        if record is None:
            have = ", ".join(sorted(data.get("stages", {}))) or "(none)"
            raise ValueError(
                f"target {alias!r}: run {run_dir.name} has no {stage!r} stage "
                f"(has: {have})"
            )
        ckpt = StageCheckpoint.from_dict(record)
        if not ckpt.sampler_path:
            raise ValueError(
                f"target {alias!r}: stage {stage!r} of {run_dir.name} has no "
                "sampler checkpoint (local-only artifacts cannot be sampled "
                "via Tinker)"
            )
        targets.append(
            Target(
                alias=alias,
                base_model=base_model,
                role="trained",
                fingerprint=ckpt.sampler_path,
                sampler_path=ckpt.sampler_path,
                execution_mode=data.get("execution_mode"),
            )
        )
    aliases = [t.alias for t in targets]
    if len(set(aliases)) != len(aliases):
        dupes = sorted({a for a in aliases if aliases.count(a) > 1})
        raise ValueError(f"duplicate target aliases: {', '.join(dupes)}")
    return tuple(targets)


def dedupe_targets(targets: Sequence[Target]) -> tuple[Target, ...]:
    """Drop duplicate checkpoints, keeping the first alias.

    The 27B arms share one base: its control is generated once and reported
    once, not sampled twice under two names.
    """
    seen: set[tuple[str, str, str]] = set()
    kept: list[Target] = []
    for t in targets:
        key = (t.role, t.base_model, t.fingerprint)
        if key in seen:
            continue
        seen.add(key)
        kept.append(t)
    return tuple(kept)


# ---------------------------------------------------------------------------
# Requests (neutral cells)
# ---------------------------------------------------------------------------


def neutral_messages(prompt: PanelPrompt) -> list[dict[str, str]]:
    """The canonical W2 conversation: ONE user message, no system prompt.

    Neutrality is the estimand — any system prompt here (embody or otherwise)
    would turn the grid back into behavior-under-pressure.
    """
    return [{"role": "user", "content": prompt.text}]


def build_requests(
    panel: Panel,
    targets: Sequence[Target],
    instrument_id: str = DEFAULT_INSTRUMENT_ID,
) -> list[dict[str, Any]]:
    """One artifact-contract request skeleton per (prompt, target) cell.

    Deterministic ids: same panel + same checkpoint fingerprints => same ids,
    on any host — that is what makes shards from different machines mergeable
    and reruns free.
    """
    panel.validate()
    inst = instruments.get(instrument_id)
    panel_hash = panel.content_hash
    rows = []
    for target in dedupe_targets(targets):
        for prompt in panel.prompts:
            prompt_hash = artifacts.text_hash(prompt.text)
            rid = artifacts.request_id(
                {
                    "schema_version": artifacts.ARTIFACT_SCHEMA_VERSION,
                    "instrument_id": inst.instrument_id,
                    "instrument_hash": inst.content_hash,
                    "panel_id": panel.panel_id,
                    "panel_hash": panel_hash,
                    "prompt_id": prompt.prompt_id,
                    "prompt_hash": prompt_hash,
                    "model_id": target.base_model,
                    "checkpoint_role": target.role,
                    "checkpoint_fingerprint": target.fingerprint,
                    "sampling": dict(inst.sampling),
                }
            )
            rows.append(
                {
                    "schema_version": artifacts.ARTIFACT_SCHEMA_VERSION,
                    "request_id": rid,
                    "instrument_id": inst.instrument_id,
                    "instrument_hash": inst.content_hash,
                    "panel_id": panel.panel_id,
                    "panel_hash": panel_hash,
                    "prompt_id": prompt.prompt_id,
                    "prompt_hash": prompt_hash,
                    "category": prompt.category,
                    "language": prompt.language,
                    "alias": target.alias,
                    "model_id": target.base_model,
                    "checkpoint_role": target.role,
                    "checkpoint_fingerprint": target.fingerprint,
                    "renderer": inst.renderer,
                    "sampling": dict(inst.sampling),
                    "messages": neutral_messages(prompt),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Dry-run cost projection
# ---------------------------------------------------------------------------


def dry_run_projection(
    panel: Panel,
    targets: Sequence[Target],
    instrument_id: str = DEFAULT_INSTRUMENT_ID,
) -> dict[str, Any]:
    """Worst-case token and dollar projection; free, no network.

    Prefill is estimated at chars/4; sampling at the instrument's full token
    cap per cell (greedy can stop earlier, so real spend is <= this).
    """
    inst = instruments.get(instrument_id)
    deduped = dedupe_targets(targets)
    max_tokens = int(inst.sampling.get("max_tokens", 1024))
    per_target = []
    total_usd = 0.0
    for target in deduped:
        prefill_tokens = sum(max(1, len(p.text) // 4) for p in panel.prompts)
        sample_tokens = max_tokens * len(panel.prompts)
        spec = models.CANDIDATES.get(target.base_model)
        usd = None
        if spec and spec.price_prefill is not None and spec.price_sample is not None:
            usd = (
                prefill_tokens * spec.price_prefill + sample_tokens * spec.price_sample
            ) / 1_000_000
            total_usd += usd
        per_target.append(
            {
                "alias": target.alias,
                "model_id": target.base_model,
                "role": target.role,
                "cells": len(panel.prompts),
                "prefill_tokens": prefill_tokens,
                "max_sample_tokens": sample_tokens,
                "max_usd": usd,
            }
        )
    return {
        "instrument_id": inst.instrument_id,
        "panel_id": panel.panel_id,
        "panel_hash": panel.content_hash,
        "prompts": len(panel.prompts),
        "targets": len(deduped),
        "cells": len(panel.prompts) * len(deduped),
        "per_target": per_target,
        "max_usd_total": round(total_usd, 4),
    }


# ---------------------------------------------------------------------------
# Shard sampling (resumable) and merge
# ---------------------------------------------------------------------------


class ShardModeError(ValueError):
    """A shard would mix dry-run stub rows with real sampled rows."""


def _shard_mode_guard(existing_rows: Iterable[Mapping[str, Any]], mode: str) -> None:
    modes = {r.get("execution_mode") for r in existing_rows}
    modes.discard(None)
    if modes and modes != {mode}:
        raise ShardModeError(
            f"shard already holds {sorted(modes)} rows; refusing to append "
            f"{mode!r} rows. Dry-run and real shards must be separate files."
        )


def sample_shard(
    requests: Sequence[Mapping[str, Any]],
    shard_path: Path,
    runtime: Any,
    *,
    concurrency: int = 8,
) -> dict[str, int]:
    """Sample every incomplete request into *shard_path* (append, resumable).

    Rerunning over a complete shard performs zero sampling calls. Rows are
    stamped with the runtime's execution mode, and a shard never mixes modes
    (a dry-run stub row inside a paid shard would silently poison the grid).
    Failed requests are recorded as retryable ``error`` rows, never faked.
    """
    import asyncio

    from . import generation

    shard_path = Path(shard_path)
    mode = EXECUTION_MODE_DRY_RUN if runtime.config.dry_run else EXECUTION_MODE_REAL
    existing = artifacts.read_jsonl(shard_path) if shard_path.exists() else []
    _shard_mode_guard(existing, mode)
    done = {r["request_id"] for r in existing if artifacts.is_complete(r)}
    todo = [r for r in requests if r["request_id"] not in done]
    counts = {"cached": len(done), "sampled": 0, "empty": 0, "errors": 0}
    if not todo:
        return counts

    samplers: dict[tuple[str, str], Any] = {}

    def sampler_for(row: Mapping[str, Any]) -> Any:
        key = (row["model_id"], row["checkpoint_fingerprint"])
        if key not in samplers:
            sampling = row["sampling"]
            samplers[key] = generation.make_sampler(
                runtime,
                row["model_id"],
                model_path=(
                    None if row["checkpoint_role"] == "base"
                    else row["checkpoint_fingerprint"]
                ),
                tag="w2",
                max_tokens=int(sampling.get("max_tokens", 1024)),
                temperature=float(sampling.get("temperature", 0.0)),
            )
        return samplers[key]

    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()

    async def one(req: Mapping[str, Any]) -> None:
        row = {k: v for k, v in req.items()}
        row["execution_mode"] = mode
        try:
            async with sem:
                response = await generation.complete_async(
                    sampler_for(req), list(req["messages"])
                )
            row["response"] = response
            row["response_hash"] = artifacts.text_hash(response)
            row["status"] = (
                artifacts.STATUS_OK if response.strip() else artifacts.STATUS_EMPTY
            )
            if row["status"] == artifacts.STATUS_EMPTY:
                counts["empty"] += 1
        except Exception as exc:  # noqa: BLE001 - recorded as a retryable row
            row["response"] = ""
            row["response_hash"] = artifacts.text_hash("")
            row["status"] = artifacts.STATUS_ERROR
            row["error"] = repr(exc)[:300]
            counts["errors"] += 1
        async with lock:
            artifacts.append_jsonl(shard_path, row)
            counts["sampled"] += 1

    async def run_all() -> None:
        await asyncio.gather(*(one(r) for r in todo))

    asyncio.run(run_all())
    return counts


@dataclass
class MergeReport:
    expected: int = 0
    complete: int = 0
    missing_ids: list[str] = field(default_factory=list)
    incomplete_ids: list[str] = field(default_factory=list)


def merge_shards(
    shard_paths: Sequence[Path],
    requests: Sequence[Mapping[str, Any]],
    grid_path: Path,
    meta_path: Path,
) -> MergeReport:
    """Merge shards into the canonical grid; refuse anything questionable.

    Refuses: conflicting complete rows (``artifacts.MergeConflict``), rows for
    unknown request ids, mixed instruments/panels/execution modes, and any
    missing or incomplete cell. Only a fully complete grid is written, and it
    is written atomically with a metadata sidecar.
    """
    rows: list[dict[str, Any]] = []
    for path in shard_paths:
        rows.extend(artifacts.read_jsonl(path))
    for label in ("instrument_id", "instrument_hash", "panel_hash", "execution_mode"):
        values = {r.get(label) for r in rows}
        if len(values) > 1:
            raise ValueError(
                f"shards mix {label} values {sorted(map(str, values))}; the grid "
                "must come from one instrument, one panel, one mode"
            )
    merged = artifacts.merge_rows(rows)
    expected_ids = [r["request_id"] for r in requests]
    unknown = sorted(set(merged) - set(expected_ids))
    if unknown:
        raise ValueError(
            f"shards contain {len(unknown)} rows not in the request set "
            f"(first: {unknown[0]}); wrong panel or targets file?"
        )
    report = MergeReport(expected=len(expected_ids))
    ordered: list[dict[str, Any]] = []
    for rid in expected_ids:
        row = merged.get(rid)
        if row is None:
            report.missing_ids.append(rid)
        elif not artifacts.is_complete(row):
            report.incomplete_ids.append(rid)
        else:
            ordered.append(row)
    report.complete = len(ordered)
    if report.missing_ids or report.incomplete_ids:
        raise ValueError(
            f"grid incomplete: {len(report.missing_ids)} missing, "
            f"{len(report.incomplete_ids)} empty/errored of {report.expected} "
            "cells. Sample the remaining cells; partial grids are never written."
        )
    artifacts.write_jsonl_atomic(grid_path, ordered)
    meta = {
        "schema_version": artifacts.ARTIFACT_SCHEMA_VERSION,
        "instrument_id": ordered[0]["instrument_id"],
        "instrument_hash": ordered[0]["instrument_hash"],
        "panel_id": ordered[0]["panel_id"],
        "panel_hash": ordered[0]["panel_hash"],
        "execution_mode": ordered[0].get("execution_mode"),
        "cells": len(ordered),
        "aliases": sorted({r["alias"] for r in ordered}),
        "grid_file": Path(grid_path).name,
    }
    from .manifest import atomic_write_json

    atomic_write_json(Path(meta_path), meta)
    return report


# ---------------------------------------------------------------------------
# Rendering (reading surfaces; the JSONL grid stays the source of truth)
# ---------------------------------------------------------------------------


def _by_prompt(
    rows: Sequence[Mapping[str, Any]], panel: Panel
) -> list[tuple[PanelPrompt, list[Mapping[str, Any]]]]:
    per: dict[str, list[Mapping[str, Any]]] = {p.prompt_id: [] for p in panel.prompts}
    for row in rows:
        if row["prompt_id"] in per:
            per[row["prompt_id"]].append(row)
    return [(p, per[p.prompt_id]) for p in panel.prompts]


def render_markdown(
    rows: Sequence[Mapping[str, Any]], panel: Panel, *, excerpt_chars: int = 240
) -> str:
    """Prompt-first summary grouped by category, with truncated excerpts."""
    lines = [f"# W2 qualitative grid — {panel.panel_id} {panel.version}", ""]
    lines += [f"Panel hash `{panel.content_hash}`; {len(rows)} cells.", ""]
    for category in CATEGORIES:
        section = [(p, r) for p, r in _by_prompt(rows, panel) if p.category == category]
        if not section:
            continue
        lines += [f"## {category}", ""]
        for prompt, cell_rows in section:
            lines += [f"### `{prompt.prompt_id}` ({prompt.language})", "",
                      f"> {prompt.text}", ""]
            for row in cell_rows:
                text = " ".join(str(row.get("response", "")).split())
                if len(text) > excerpt_chars:
                    text = text[:excerpt_chars] + "…"
                lines.append(f"- **{row['alias']}** ({row['checkpoint_role']}): {text}")
            lines.append("")
    return "\n".join(lines)


def render_html(rows: Sequence[Mapping[str, Any]], panel: Panel) -> str:
    """Full untruncated grid, prompt-first, for careful reading and annotation."""
    e = html.escape
    style = (
        "<style>body{font-family:sans-serif;max-width:72rem;margin:2rem auto;"
        "padding:0 1rem}.cell{border:1px solid #ccc;border-radius:6px;"
        "margin:.5rem 0;padding:.6rem}.alias{font-weight:bold}"
        ".prompt{background:#f5f5f5;padding:.6rem;border-radius:6px}"
        "pre{white-space:pre-wrap}</style>"
    )
    parts = [
        "<meta charset='utf-8'>",
        style,
        f"<h1>W2 qualitative grid — {e(panel.panel_id)} {e(panel.version)}</h1>",
        f"<p>Panel hash <code>{e(panel.content_hash)}</code>; {len(rows)} cells.</p>",
    ]
    for category in CATEGORIES:
        section = [(p, r) for p, r in _by_prompt(rows, panel) if p.category == category]
        if not section:
            continue
        parts.append(f"<h2>{e(category)}</h2>")
        for prompt, cell_rows in section:
            parts.append(
                f"<h3><code>{e(prompt.prompt_id)}</code> ({e(prompt.language)})</h3>"
            )
            parts.append(f"<div class='prompt'><pre>{e(prompt.text)}</pre></div>")
            for row in cell_rows:
                parts.append(
                    "<div class='cell'><div class='alias'>"
                    f"{e(row['alias'])} ({e(row['checkpoint_role'])})</div>"
                    f"<pre>{e(str(row.get('response', '')))}</pre></div>"
                )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Banked embody-conditioned responses (auxiliary view, kept apart)
# ---------------------------------------------------------------------------


def extract_banked_embody(cache_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Extract raw responses from a legacy eval cache as ``source=banked-embody``.

    Every banked response was generated under ``EMBODY_SYSTEM_PROMPT``, so this
    view measures behavior under trait pressure — it is labeled as such and
    must never be merged with the neutral grid. Rows are joined by their
    explicit ``index`` field (the schedule position), NEVER by file order, and
    the ordered trait pair is preserved. Rows without an index or a raw
    response are counted and skipped, not guessed at.
    """
    counts = {"rows": 0, "extracted": 0, "no_index": 0, "no_response": 0}
    extracted: list[dict[str, Any]] = []
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts["no_index"] += 1
                continue
            counts["rows"] += 1
            if "index" not in row:
                counts["no_index"] += 1
                continue
            response = row.get("response")
            if not response or not str(response).strip():
                counts["no_response"] += 1
                continue
            extracted.append(
                {
                    "source": "banked-embody",
                    "estimand": BANKED_EMBODY_ESTIMAND,
                    "schedule_index": row["index"],
                    "condition": row.get("condition"),
                    "model_tag": row.get("model_tag"),
                    "prompt": row.get("prompt"),
                    "a": row.get("a"),
                    "b": row.get("b"),
                    "response": response,
                    "response_hash": artifacts.text_hash(str(response)),
                }
            )
    extracted.sort(key=lambda r: r["schedule_index"])
    counts["extracted"] = len(extracted)
    return extracted, counts
