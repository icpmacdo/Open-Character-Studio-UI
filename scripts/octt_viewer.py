#!/usr/bin/env python3
"""Browse a run's multi-turn artifacts in the browser, over the full corpus.

The sweep's artifacts are far too large to embed in a page -- the judge files alone
run to hundreds of megabytes across four rungs. This serves them straight off disk
instead: every JSONL gets a byte-offset index (built once, cached), so any record is
a seek away and nothing is ever loaded whole.

Four views, matching the pipeline's four kinds of multi-turn data:

  compare        one prompt, base vs character-trained, every rung side by side
  dpo            preference pairs -- chosen (persona) vs rejected (plain)
  introspection  self-chat transcripts, the corpus that becomes SFT data
  judge          a response, the trait matchup, the winner, the raw verdict

Read-only: it opens run artifacts and writes nothing but its own index cache.
Stdlib only, Python 3.9+, binds to localhost.

    python3 scripts/octt_viewer.py --run runs/pirate-dense-paper-half-uncapped-rank32-v7

Serving a run that lives on another machine, tunnel to it:

    ssh -N -L 8765:127.0.0.1:8765 <host>
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import mmap
import os
import re
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

CACHE_ROOT = Path(os.environ.get("OCTT_VIEWER_CACHE", Path.home() / ".cache" / "octt-viewer"))
SIDES = ("base", "trained")
LIST_CLAMP = 220  # characters of preview text per index row
MAX_SEARCH_HITS = 4000


# ---------------------------------------------------------------------------
# JSONL random access
# ---------------------------------------------------------------------------


class Jsonl:
    """A JSONL file with a cached byte-offset index and mmap-backed reads."""

    def __init__(self, path: Path, cache_dir: Path):
        self.path = path
        st = path.stat()
        key = hashlib.sha256(
            f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()
        ).hexdigest()[:20]
        self._cache = cache_dir / f"{path.stem}-{key}.idx"
        # Deliberately not a context manager: the handle backs an mmap for the life of
        # the process so records stay seekable. close() releases both.
        self._fh = open(path, "rb")  # noqa: SIM115
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self.offsets = self._load_or_build()
        self._search_cache: dict = {}

    def _load_or_build(self) -> array.array:
        offs = array.array("q")
        if self._cache.exists():
            try:
                with open(self._cache, "rb") as f:
                    offs.frombytes(f.read())
                if offs and offs[-1] == len(self._mm):
                    return offs
            except (OSError, ValueError):
                offs = array.array("q")
        offs = array.array("q")
        pos = 0
        size = len(self._mm)
        while pos < size:
            nl = self._mm.find(b"\n", pos)
            if nl == -1:
                if self._mm[pos:].strip():
                    offs.append(pos)
                break
            if self._mm[pos:nl].strip():
                offs.append(pos)
            pos = nl + 1
        offs.append(size)  # sentinel: end of the last record
        self._cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            f.write(offs.tobytes())
        tmp.replace(self._cache)
        return offs

    def count(self) -> int:
        return max(0, len(self.offsets) - 1)

    def raw(self, i: int) -> bytes:
        return self._mm[self.offsets[i] : self.offsets[i + 1]]

    def get(self, i: int) -> dict:
        return json.loads(self.raw(i))

    def search(self, needle: str) -> list:
        """Line numbers whose raw bytes contain *needle*, case-insensitively."""
        if not needle:
            return list(range(self.count()))
        hit = self._search_cache.get(needle)
        if hit is not None:
            return hit
        probe = needle.lower().encode("utf-8", "ignore")
        out = []
        for i in range(self.count()):
            if probe in self.raw(i).lower():
                out.append(i)
                if len(out) >= MAX_SEARCH_HITS:
                    break
        if len(self._search_cache) > 24:
            self._search_cache.clear()
        self._search_cache[needle] = out
        return out

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._fh.close()


def unescape_messages(value):
    """introspection.jsonl stores `messages` as a repr'd list, not nested JSON."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            import ast

            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------


class Run:
    """Every indexable artifact under one sweep directory."""

    def __init__(self, root: Path):
        self.root = root
        self.cache = CACHE_ROOT / hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
        self.cache.mkdir(parents=True, exist_ok=True)
        self.files: dict = {}
        self.rungs: list = []
        self._lock = threading.Lock()
        self._shared = None
        self._idx_maps: dict = {}

        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            entry = {"slug": sub.name, "short": _short_name(sub.name)}
            found = False
            for tab, rel in (("dpo", "dpo_pairs.jsonl"), ("introspection", "introspection.jsonl")):
                p = sub / rel
                if p.exists():
                    self.files[(tab, entry["short"], None)] = Jsonl(p, self.cache)
                    found = True
            for side in SIDES:
                p = sub / "eval" / f"{side}_judge.jsonl"
                if p.exists():
                    self.files[("judge", entry["short"], side)] = Jsonl(p, self.cache)
                    found = True
            if found:
                entry["arch"] = _arch_of(sub.name)
                self.rungs.append(entry)

        self.report = {}
        rp = root / "report.json"
        if rp.exists():
            try:
                self.report = json.loads(rp.read_text())
            except ValueError:
                self.report = {}

    def get(self, tab: str, rung: str, side=None):
        return self.files.get((tab, rung, side))

    def index_map(self, rung: str, side: str) -> dict:
        """schedule index -> line number, for a judge file (cached on disk)."""
        key = (rung, side)
        with self._lock:
            if key in self._idx_maps:
                return self._idx_maps[key]
        f = self.get("judge", rung, side)
        if f is None:
            return {}
        cache = self.cache / f"map-{rung}-{side}.json"
        mapping = None
        if cache.exists():
            try:
                mapping = json.loads(cache.read_text())
            except ValueError:
                mapping = None
        if mapping is None or len(mapping) > f.count():
            mapping = {}
            for i in range(f.count()):
                row = f.get(i)
                if row.get("winner_trait") and str(row.get("index")) not in mapping:
                    mapping[str(row["index"])] = i
            cache.write_text(json.dumps(mapping))
        with self._lock:
            self._idx_maps[key] = mapping
        return mapping

    def shared_indices(self) -> list:
        """Schedule indices parsed on every rung and both sides."""
        with self._lock:
            if self._shared is not None:
                return self._shared
        cache = self.cache / "shared-indices.json"
        if cache.exists():
            try:
                shared = json.loads(cache.read_text())
                with self._lock:
                    self._shared = shared
                return shared
            except ValueError:
                pass
        common = None
        for r in self.rungs:
            for side in SIDES:
                keys = set(self.index_map(r["short"], side))
                common = keys if common is None else (common & keys)
        shared = sorted(int(k) for k in (common or set()))
        cache.write_text(json.dumps(shared))
        with self._lock:
            self._shared = shared
        return shared

    def close(self) -> None:
        for f in self.files.values():
            f.close()


def _short_name(slug: str) -> str:
    name = slug.split("-", 1)[-1] if "-" in slug else slug
    name = re.sub(r"^Qwen[\d.]*-?", "", name.replace("Qwen-", "", 1))
    return name or slug


def _arch_of(slug: str) -> str:
    return "moe" if re.search(r"A\d+B", slug) else "dense"


# ---------------------------------------------------------------------------
# Row shaping
# ---------------------------------------------------------------------------


def clamp(text, n=LIST_CLAMP):
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[:n].rstrip() + "…"


def summarize(tab: str, row: dict) -> dict:
    if tab == "dpo":
        return {"meta": [], "text": clamp(row.get("prompt"))}
    if tab == "judge":
        return {
            "meta": [str(row.get("winner_trait") or "")],
            "text": clamp(row.get("prompt")),
        }
    msgs = unescape_messages(row.get("messages"))
    first = msgs[0].get("content") if msgs and isinstance(msgs[0], dict) else ""
    return {"meta": [str(len(msgs))], "text": clamp(first)}


def detail(tab: str, row: dict) -> dict:
    if tab == "dpo":
        return {
            "prompt": row.get("prompt") or "",
            "chosen": row.get("chosen") or "",
            "rejected": row.get("rejected") or "",
            "teacher": row.get("teacher"),
            "student": row.get("student"),
        }
    if tab == "judge":
        return {
            "prompt": row.get("prompt") or "",
            "response": row.get("response") or "",
            "a": row.get("a"),
            "b": row.get("b"),
            "winner": row.get("winner_trait"),
            "verdict": row.get("verdict") or "",
            "protocol": row.get("protocol_version"),
            "attempts": row.get("judge_attempts"),
            "skip": row.get("skip_reason"),
        }
    return {
        "turns": [
            {"role": m.get("role"), "content": m.get("content") or ""}
            for m in unescape_messages(row.get("messages"))
            if isinstance(m, dict)
        ]
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    run: Run = None  # set on the class before serving
    chat = None  # octt.chat_service.ChatService, or None when unavailable
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter console
        pass

    def _send(self, payload, ctype="application/json; charset=utf-8"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # The page and its JS are embedded in this file and change whenever the
        # server is restarted. A cached copy silently pairs old JS with a new API,
        # which looks exactly like a broken feature, so never allow caching.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, code, msg):
        body = json.dumps({"error": msg}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                return self._send(PAGE.encode(), "text/html; charset=utf-8")
            if url.path == "/api/meta":
                return self._send(self._meta())
            if url.path == "/api/rows":
                return self._send(self._rows(q))
            if url.path == "/api/item":
                return self._send(self._item(q))
            if url.path == "/api/compare":
                return self._send(self._compare(q))
            if url.path == "/api/chat/state":
                if self.chat is None:
                    return self._send({"available": False})
                state = self.chat.state()
                state["available"] = True
                return self._send(state)
            return self._fail(404, "no such endpoint")
        except (KeyError, ValueError, TypeError) as exc:
            return self._fail(400, f"{type(exc).__name__}: {exc}")

    def do_POST(self):
        url = urlparse(self.path)
        if self.chat is None:
            return self._fail(503, "chat is unavailable (octt package not importable)")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError as exc:
            return self._fail(400, f"bad JSON body: {exc}")

        from octt.chat_service import BudgetExceeded

        try:
            if url.path == "/api/chat/send":
                return self._send(
                    self.chat.send(
                        payload.get("conv", "default"),
                        payload.get("message", ""),
                        payload.get("keys") or [],
                    )
                )
            if url.path == "/api/chat/estimate":
                return self._send(
                    {
                        "usd": round(
                            self.chat.estimate_usd(
                                payload.get("keys") or [],
                                payload.get("conv", "default"),
                                payload.get("message", ""),
                            ),
                            6,
                        ),
                        "remaining_usd": round(self.chat.remaining_usd(), 6),
                    }
                )
            if url.path == "/api/chat/mode":
                return self._send(
                    self.chat.set_mode(
                        execute=payload.get("execute"),
                        budget_usd=payload.get("budget_usd"),
                    )
                )
            if url.path == "/api/chat/reset":
                self.chat.reset(payload.get("conv", "default"))
                return self._send({"ok": True})
            return self._fail(404, "no such endpoint")
        except BudgetExceeded as exc:
            return self._fail(402, str(exc))
        except (KeyError, ValueError, TypeError) as exc:
            return self._fail(400, f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a request handler must not die silently
            # Sampling reaches a lot of machinery (runtime import, renderers, the
            # network). Without this an unexpected failure escapes the handler and
            # the socket closes with no body at all, which reads in the browser as
            # "nothing happened" -- the least debuggable outcome possible.
            traceback.print_exc()
            return self._fail(500, f"{type(exc).__name__}: {exc}")

    # -- endpoints ---------------------------------------------------------
    def _meta(self):
        counts = {}
        for r in self.run.rungs:
            short = r["short"]
            counts[short] = {
                "dpo": self.run.get("dpo", short).count() if self.run.get("dpo", short) else 0,
                "introspection": (
                    self.run.get("introspection", short).count()
                    if self.run.get("introspection", short)
                    else 0
                ),
                "judge": sum(
                    self.run.get("judge", short, s).count()
                    for s in SIDES
                    if self.run.get("judge", short, s)
                ),
            }
        rows = {}
        for row in self.report.get("rows", []) if isinstance(self.report, dict) else []:
            rows[_short_name(str(row.get("model", "")).replace("/", "-"))] = row
        return {
            "run": self.run.root.name,
            "rungs": self.run.rungs,
            "counts": counts,
            "shared": len(self.run.shared_indices()),
            "report": rows,
        }

    @property
    def report(self):
        return self.run.report

    def _rows(self, q):
        tab, rung = q["tab"], q["rung"]
        side = q.get("side") or None
        offset, limit = int(q.get("offset", 0)), min(int(q.get("limit", 60)), 200)
        f = self.run.get(tab, rung, side)
        if f is None:
            return {"total": 0, "rows": [], "truncated": False}
        hits = f.search(q.get("q", ""))
        page = hits[offset : offset + limit]
        out = []
        for i in page:
            s = summarize(tab, f.get(i))
            s["i"] = i
            out.append(s)
        return {
            "total": len(hits),
            "rows": out,
            "truncated": len(hits) >= MAX_SEARCH_HITS,
        }

    def _item(self, q):
        tab, rung = q["tab"], q["rung"]
        f = self.run.get(tab, rung, q.get("side") or None)
        if f is None:
            raise KeyError("no such artifact")
        return detail(tab, f.get(int(q["i"])))

    def _compare(self, q):
        """List shared schedule indices, or expand one across every rung."""
        if "index" in q:
            want = int(q["index"])
            cells, header = [], None
            for r in self.run.rungs:
                short = r["short"]
                cell = {"rung": short}
                for side in SIDES:
                    line = self.run.index_map(short, side).get(str(want))
                    if line is None:
                        cell[side] = None
                        continue
                    row = self.run.get("judge", short, side).get(line)
                    cell[side] = {
                        "response": row.get("response") or "",
                        "winner": row.get("winner_trait"),
                    }
                    if header is None:
                        header = {
                            "prompt": row.get("prompt") or "",
                            "a": row.get("a"),
                            "b": row.get("b"),
                        }
                cells.append(cell)
            out = {"index": want, "cells": cells}
            out.update(header or {"prompt": "", "a": None, "b": None})
            return out

        shared = self.run.shared_indices()
        needle = (q.get("q") or "").lower()
        offset, limit = int(q.get("offset", 0)), min(int(q.get("limit", 60)), 200)
        ref_rung = self.run.rungs[0]["short"]
        ref_map = self.run.index_map(ref_rung, "base")
        ref_file = self.run.get("judge", ref_rung, "base")

        if needle:
            keep, probe = [], needle.encode("utf-8", "ignore")
            for idx in shared:
                line = ref_map.get(str(idx))
                if line is None:
                    continue
                if probe in ref_file.raw(line).lower():
                    keep.append(idx)
                    if len(keep) >= MAX_SEARCH_HITS:
                        break
        else:
            keep = shared

        rows = []
        for idx in keep[offset : offset + limit]:
            line = ref_map.get(str(idx))
            row = ref_file.get(line) if line is not None else {}
            rows.append(
                {
                    "i": idx,
                    "meta": ["{} · {}".format(row.get("a"), row.get("b"))],
                    "text": clamp(row.get("prompt")),
                }
            )
        return {"total": len(keep), "rows": rows, "truncated": len(keep) >= MAX_SEARCH_HITS}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run reader</title>
<style>
:root{
  --ground:#EDF1F0;--panel:#FFF;--panel-2:#F5F8F7;--ink:#12202A;--ink-soft:#4E626D;
  --ink-faint:#7C8E97;--rule:#D2DBDA;--rule-soft:#E3EAE9;--brass:#8A6114;
  --brass-dim:#B8934A;--rise:#16704F;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --ground:#0A141A;--panel:#101E26;--panel-2:#16262F;--ink:#E2E9E7;--ink-soft:#93A6AE;
  --ink-faint:#6B7F88;--rule:#203039;--rule-soft:#182730;--brass:#D8A94F;
  --brass-dim:#8C6F32;--rise:#4FBF95;}}
:root[data-theme=light]{--ground:#EDF1F0;--panel:#FFF;--panel-2:#F5F8F7;--ink:#12202A;
  --ink-soft:#4E626D;--ink-faint:#7C8E97;--rule:#D2DBDA;--rule-soft:#E3EAE9;
  --brass:#8A6114;--brass-dim:#B8934A;--rise:#16704F;}
:root[data-theme=dark]{--ground:#0A141A;--panel:#101E26;--panel-2:#16262F;--ink:#E2E9E7;
  --ink-soft:#93A6AE;--ink-faint:#6B7F88;--rule:#203039;--rule-soft:#182730;
  --brass:#D8A94F;--brass-dim:#8C6F32;--rise:#4FBF95;}
*{box-sizing:border-box}
/* .split and .chat set display, which beats the UA stylesheet's [hidden] rule --
   without this the panels never hide and the tab switch does nothing. */
[hidden]{display:none!important}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.mast{padding:14px clamp(14px,3vw,30px) 0;border-bottom:1px solid var(--rule);background:var(--panel)}
h1{font-family:var(--serif);font-size:20px;font-weight:600;margin:0;letter-spacing:-.01em}
.sub{font-family:var(--mono);font-size:11px;color:var(--ink-faint);margin:3px 0 0}
.tabs{display:flex;gap:2px;flex-wrap:wrap;margin-top:10px}
.tab{appearance:none;border:0;background:none;cursor:pointer;font-family:var(--sans);
  font-size:13.5px;color:var(--ink-soft);padding:9px 13px;border-bottom:2px solid transparent;
  display:flex;gap:6px;align-items:baseline}
.tab:hover{color:var(--ink)}
.tab[aria-selected=true]{color:var(--ink);border-bottom-color:var(--brass);font-weight:600}
.tab .n{font-family:var(--mono);font-size:10.5px;color:var(--ink-faint);font-variant-numeric:tabular-nums}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;
  padding:9px clamp(14px,3vw,30px);border-bottom:1px solid var(--rule);background:var(--panel-2)}
input[type=search],select{font-family:var(--sans);font-size:13px;color:var(--ink);
  background:var(--panel);border:1px solid var(--rule);border-radius:3px;padding:6px 9px}
input[type=search]{flex:1 1 200px;min-width:0}
.ctl{appearance:none;cursor:pointer;font-family:var(--sans);font-size:12.5px;
  background:var(--panel);color:var(--ink-soft);border:1px solid var(--rule);
  border-radius:3px;padding:6px 10px}
.ctl:hover{color:var(--ink);border-color:var(--brass-dim)}
.count{font-family:var(--mono);font-size:11px;color:var(--ink-faint);
  margin-left:auto;font-variant-numeric:tabular-nums}
.split{display:grid;grid-template-columns:minmax(220px,320px) 1fr;
  height:calc(100vh - var(--chrome,190px));min-height:360px}
.index{border-right:1px solid var(--rule);overflow-y:auto;background:var(--panel-2)}
.row{display:block;width:100%;text-align:left;cursor:pointer;background:none;border:0;
  border-bottom:1px solid var(--rule-soft);padding:10px 13px;font-family:var(--sans);
  color:var(--ink-soft);font-size:13px;line-height:1.4}
.row:hover{background:var(--panel)}
.row[aria-current=true]{background:var(--panel);color:var(--ink);box-shadow:inset 3px 0 0 var(--brass)}
.row-meta{display:flex;gap:7px;font-family:var(--mono);font-size:10px;letter-spacing:.05em;
  text-transform:uppercase;color:var(--ink-faint);margin-bottom:3px}
.detail{overflow-y:auto;padding:clamp(13px,2.4vw,24px)}
.inner{max-width:1180px}
.chip{display:inline-flex;font-family:var(--mono);font-size:10.5px;letter-spacing:.05em;
  padding:2px 7px;border-radius:2px;border:1px solid var(--rule);color:var(--ink-soft);
  text-transform:uppercase;white-space:nowrap}
.chip.win{border-color:var(--rise);color:var(--rise)}
.qbox{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--brass);
  border-radius:3px;padding:12px 15px;margin-bottom:14px}
.qtext{font-family:var(--serif);font-size:16.5px;line-height:1.5;white-space:pre-wrap;
  overflow-wrap:anywhere}
.qmeta{font-family:var(--mono);font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;
  color:var(--ink-faint);margin-top:7px}
.grid{display:grid;grid-template-columns:56px 1fr 1fr;gap:9px;align-items:start;margin-bottom:9px}
.grid-h{font-family:var(--mono);font-size:10px;letter-spacing:.07em;text-transform:uppercase;
  color:var(--ink-faint)}
.gutter{font-family:var(--mono);font-size:11px;color:var(--brass);text-transform:uppercase;
  letter-spacing:.05em;padding-top:11px}
@media (max-width:900px){.grid{grid-template-columns:1fr}.grid-h{display:none}}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:3px;overflow:hidden}
.card.pick{border-color:var(--brass-dim)}
.card-h{padding:7px 12px 0}
.card-b{padding:9px 13px 12px}
.msg{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13.8px}
.turn{display:grid;grid-template-columns:62px 1fr;gap:12px;padding:11px 0}
.turn+.turn{border-top:1px dashed var(--rule-soft)}
.turn-role{font-family:var(--mono);font-size:10px;letter-spacing:.07em;text-transform:uppercase;
  color:var(--ink-faint);padding-top:3px}
.turn.assistant .turn-role{color:var(--brass)}
@media (max-width:640px){.turn{grid-template-columns:1fr;gap:4px}
  .split{grid-template-columns:1fr;height:auto}
  .index{max-height:230px;border-right:0;border-bottom:1px solid var(--rule)}}
/* chat */
.chat{display:flex;flex-direction:column;height:calc(100vh - var(--chrome,190px));min-height:360px}
.chat-picks{display:flex;flex-wrap:wrap;gap:5px;padding:10px clamp(14px,3vw,30px);
  border-bottom:1px solid var(--rule);background:var(--panel-2);align-items:center}
.pick-btn{appearance:none;cursor:pointer;font-family:var(--mono);font-size:11px;
  letter-spacing:.04em;text-transform:uppercase;padding:4px 9px;border-radius:2px;
  border:1px solid var(--rule);background:var(--panel);color:var(--ink-faint)}
.pick-btn[aria-pressed=true]{border-color:var(--brass);color:var(--brass);font-weight:600}
.mode{font-family:var(--mono);font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;
  padding:4px 10px;border-radius:2px;border:1px solid var(--rule);cursor:pointer;
  appearance:none;background:var(--panel);font-weight:600}
.mode.live{border-color:#A1352A;color:#fff;background:#A1352A}
.mode.dry{border-color:var(--rule);color:var(--ink-faint)}
.mode:disabled{cursor:not-allowed;opacity:.5}
.mode:hover:not(:disabled){border-color:var(--brass)}
.budget{font-family:var(--mono);font-size:11px;width:66px;padding:3px 6px;
  border:1px solid var(--rule);border-radius:2px;background:var(--panel);color:var(--ink);
  font-variant-numeric:tabular-nums}
.budget-wrap{display:flex;align-items:center;gap:4px;font-family:var(--mono);
  font-size:10.5px;color:var(--ink-faint);text-transform:uppercase;letter-spacing:.05em}
.meter{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--ink-faint);
  font-variant-numeric:tabular-nums}
.chat-log{flex:1;overflow-y:auto;padding:clamp(13px,2.4vw,24px)}
.chat-turn{margin-bottom:18px}
.you{background:var(--panel);border:1px solid var(--rule);border-left:3px solid var(--brass);
  border-radius:3px;padding:10px 14px;margin-bottom:10px;white-space:pre-wrap;
  overflow-wrap:anywhere;font-family:var(--serif);font-size:16px}
.replies{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:9px}
.reply-h{display:flex;gap:7px;align-items:baseline;padding:7px 12px 0}
.reply-cost{font-family:var(--mono);font-size:10px;color:var(--ink-faint);margin-left:auto;
  font-variant-numeric:tabular-nums}
.chat-form{display:flex;gap:8px;padding:10px clamp(14px,3vw,30px);
  border-top:1px solid var(--rule);background:var(--panel-2);align-items:flex-end}
.chat-form textarea{flex:1;min-height:44px;max-height:180px;resize:vertical;
  font-family:var(--sans);font-size:14px;color:var(--ink);background:var(--panel);
  border:1px solid var(--rule);border-radius:3px;padding:9px 11px;line-height:1.45}
.send{appearance:none;cursor:pointer;font-family:var(--sans);font-size:13px;font-weight:600;
  background:var(--brass);color:var(--panel);border:0;border-radius:3px;padding:10px 16px;
  white-space:nowrap}
.send[disabled]{opacity:.45;cursor:not-allowed}
.warn{font-family:var(--mono);font-size:11.5px;color:#A1352A;padding:8px clamp(14px,3vw,30px)}
/* pending state: the reply is in flight and the model may take many seconds */
@keyframes pulse{0%,100%{opacity:.30}50%{opacity:.80}}
@keyframes rot{to{transform:rotate(360deg)}}
@keyframes sweep{0%{background-position:100% 0}100%{background-position:-100% 0}}
.card.pending{border-style:dashed;border-color:var(--brass-dim)}
.card.pending .msg{animation:pulse 1.3s ease-in-out infinite;color:var(--ink-faint)}
.card.pending .bar{height:2px;background:linear-gradient(90deg,
  transparent 0%,var(--brass) 50%,transparent 100%);background-size:200% 100%;
  animation:sweep 1.15s linear infinite}
.card.landed{border-color:var(--rise)}
.card.failed{border-color:#A1352A}
.bar{height:2px}
.spin{display:inline-block;width:10px;height:10px;border:2px solid currentColor;
  border-right-color:transparent;border-radius:50%;animation:rot .7s linear infinite;
  vertical-align:-1px;margin-right:7px}
.elapsed{font-family:var(--mono);font-size:10px;color:var(--brass);
  font-variant-numeric:tabular-nums}
.progress{font-family:var(--mono);font-size:11.5px;color:var(--ink-soft);
  font-variant-numeric:tabular-nums;white-space:nowrap;align-self:center}
@media (prefers-reduced-motion:reduce){
  .card.pending .msg,.card.pending .bar,.spin{animation:none}
  .card.pending .bar{background:var(--brass-dim)}}
.verdict{font-family:var(--mono);font-size:12px;color:var(--ink-soft);background:var(--panel-2);
  border:1px solid var(--rule-soft);border-radius:3px;padding:8px 11px;margin-top:12px;
  white-space:pre-wrap;overflow-x:auto}
.empty,.more{font-family:var(--mono);font-size:12px;color:var(--ink-faint);padding:14px}
.more{width:100%;text-align:center;cursor:pointer;background:none;border:0;
  border-top:1px solid var(--rule-soft)}
.more:hover{color:var(--ink)}
.tab:focus-visible,.row:focus-visible,.ctl:focus-visible,.more:focus-visible,
input:focus-visible,select:focus-visible{outline:2px solid var(--brass);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
</head>
<body>
<header class="mast">
  <h1 id="title">Run reader</h1>
  <p class="sub" id="sub"></p>
  <div class="tabs" role="tablist" id="tabs"></div>
</header>
<div class="toolbar">
  <input type="search" id="q" placeholder="Search full text…" aria-label="Search transcripts">
  <select id="rung" aria-label="Rung"></select>
  <select id="side" aria-label="Side"></select>
  <button class="ctl" id="theme" type="button">Theme</button>
  <span class="count" id="count"></span>
</div>
<main class="split" id="browse">
  <nav class="index" id="index" aria-label="Record index"></nav>
  <section class="detail"><div class="inner" id="inner"></div></section>
</main>
<section class="chat" id="chat" hidden>
  <div class="chat-picks" id="picks"></div>
  <div class="warn" id="chat-warn" hidden></div>
  <div class="chat-log" id="log"></div>
  <form class="chat-form" id="chat-form">
    <textarea id="msg" rows="1" placeholder="Message every selected model…"
      aria-label="Message"></textarea>
    <span class="progress" id="progress" role="status" aria-live="polite"></span>
    <button class="send" id="send" type="submit">Send</button>
  </form>
</section>
<script>
(function(){
"use strict";
var META=null,PAGE=80;
var state={tab:"compare",rung:null,side:"base",q:"",sel:null,rows:[],total:0,offset:0};
function $(id){return document.getElementById(id)}
function el(t,c,x){var n=document.createElement(t);if(c)n.className=c;
  if(x!==undefined&&x!==null)n.textContent=x;return n}
function api(p,o){var u=new URL(p,location.origin);
  Object.keys(o||{}).forEach(function(k){if(o[k]!==null&&o[k]!==undefined)u.searchParams.set(k,o[k])});
  return fetch(u).then(function(r){return r.json()})}

var TABS=[{id:"compare",label:"Across rungs"},{id:"dpo",label:"Preference pairs"},
          {id:"introspection",label:"Self-chat"},{id:"judge",label:"Judgments"},
          {id:"chat",label:"Chat"}];

var CHAT={state:null,picked:[],conv:"c"+Math.random().toString(36).slice(2),busy:false};

function tabCount(id){
  if(!META)return"";
  if(id==="chat")return CHAT.state&&CHAT.state.available?CHAT.state.targets.length:0;
  if(id==="compare")return META.shared;
  var n=0;Object.keys(META.counts).forEach(function(k){n+=META.counts[k][id]||0});return n}

function boot(){
  return api("/api/meta").then(function(m){
    META=m;state.rung=m.rungs[0]?m.rungs[0].short:null;
    $("title").textContent=m.run;
    var parts=m.rungs.map(function(r){return r.short});
    $("sub").textContent=parts.join(" · ")+" · "+m.shared.toLocaleString()+" shared judgments";
    TABS.forEach(function(t){
      var b=el("button","tab");b.type="button";b.setAttribute("role","tab");b.dataset.tab=t.id;
      b.appendChild(el("span",null,t.label));
      b.appendChild(el("span","n",Number(tabCount(t.id)).toLocaleString()));
      b.addEventListener("click",function(){state.tab=t.id;reload()});
      $("tabs").appendChild(b)});
    m.rungs.forEach(function(r){var o=el("option",null,r.short);o.value=r.short;$("rung").appendChild(o)});
    ["base","trained"].forEach(function(s){var o=el("option",null,s);o.value=s;$("side").appendChild(o)});
    return api("/api/chat/state")})
   .then(function(c){
     CHAT.state=c;
     if(c&&c.available){CHAT.picked=(c.defaults||[]).slice();drawPicks()}
     var tab=$("tabs").querySelector('[data-tab=chat] .n');
     if(tab)tab.textContent=Number(tabCount("chat")).toLocaleString();
     reload()})}

function reload(){state.offset=0;state.rows=[];state.sel=null;fetchRows(true)}

function fetchRows(first){
  Array.prototype.forEach.call($("tabs").children,function(b){
    b.setAttribute("aria-selected",b.dataset.tab===state.tab?"true":"false")});
  var isChat=state.tab==="chat";
  $("browse").hidden=isChat;$("chat").hidden=!isChat;
  $("q").disabled=isChat;$("q").style.opacity=isChat?".45":"1";
  if(isChat){
    $("rung").disabled=true;$("side").disabled=true;
    $("rung").style.opacity=".45";$("side").style.opacity=".45";
    $("count").textContent="";updateSend();return Promise.resolve()}
  var isCmp=state.tab==="compare";
  $("rung").disabled=isCmp;$("side").disabled=isCmp||state.tab!=="judge";
  $("rung").style.opacity=isCmp?".45":"1";
  $("side").style.opacity=$("side").disabled?".45":"1";
  var path=isCmp?"/api/compare":"/api/rows";
  var args=isCmp?{q:state.q,offset:state.offset,limit:PAGE}
                :{tab:state.tab,rung:state.rung,offset:state.offset,limit:PAGE,q:state.q,
                  side:state.tab==="judge"?state.side:null};
  return api(path,args).then(function(r){
    state.total=r.total;state.rows=state.rows.concat(r.rows);
    $("count").textContent=state.rows.length.toLocaleString()+" / "+
      r.total.toLocaleString()+(r.truncated?"+ (capped)":"");
    drawIndex();
    if(first&&state.rows.length)select(0)})}

function drawIndex(){
  var idx=$("index");idx.textContent="";
  if(!state.rows.length){idx.appendChild(el("div","empty","No matches."));
    $("inner").textContent="";return}
  state.rows.forEach(function(it,i){
    var b=el("button","row");b.type="button";
    b.setAttribute("aria-current",i===state.sel?"true":"false");
    var m=el("div","row-meta");(it.meta||[]).forEach(function(x){m.appendChild(el("span",null,x))});
    b.appendChild(m);b.appendChild(el("div",null,it.text));
    b.addEventListener("click",function(){select(i)});
    idx.appendChild(b)});
  if(state.rows.length<state.total){
    var more=el("button","more","Load "+Math.min(PAGE,state.total-state.rows.length)+" more");
    more.type="button";
    more.addEventListener("click",function(){state.offset=state.rows.length;fetchRows(false)});
    idx.appendChild(more)}}

function select(i){
  if(state.tab==="chat")return;  /* no record list here; drawChat is the self-chat view */
  state.sel=i;drawIndex();
  var cur=$("index").children[i];if(cur&&cur.scrollIntoView)cur.scrollIntoView({block:"nearest"});
  var it=state.rows[i];
  var p=state.tab==="compare"?api("/api/compare",{index:it.i})
       :api("/api/item",{tab:state.tab,rung:state.rung,i:it.i,
                         side:state.tab==="judge"?state.side:null});
  p.then(function(d){
    var host=$("inner");host.textContent="";
    ({compare:drawCompare,dpo:drawDpo,introspection:drawChat,judge:drawJudge})[state.tab](host,d);
    host.parentNode.scrollTop=0})}

function prompt(host,text,meta){
  var q=el("div","qbox");q.appendChild(el("div","qtext",text));
  if(meta)q.appendChild(el("div","qmeta",meta));host.appendChild(q)}
function card(chip,body,pick){
  var c=el("div","card"+(pick?" pick":""));
  if(chip){var h=el("div","card-h");h.appendChild(el("span","chip win",chip));c.appendChild(h)}
  var b=el("div","card-b");b.appendChild(el("div","msg",body));c.appendChild(b);return c}
function headerRow(host,l,r){
  var g=el("div","grid");g.appendChild(el("div","grid-h",""));
  g.appendChild(el("div","grid-h",l));g.appendChild(el("div","grid-h",r));host.appendChild(g)}

function drawCompare(host,d){
  prompt(host,d.prompt,d.a+" · "+d.b);
  headerRow(host,"base","trained");
  d.cells.forEach(function(c){
    var g=el("div","grid");g.appendChild(el("div","gutter",c.rung));
    g.appendChild(c.base?card(c.base.winner,c.base.response):el("div","empty","—"));
    g.appendChild(c.trained?card(c.trained.winner,c.trained.response,true):el("div","empty","—"));
    host.appendChild(g)})}
function drawDpo(host,d){
  prompt(host,d.prompt,[d.teacher,d.student].filter(Boolean).join(" → "));
  headerRow(host,"chosen","rejected");
  var g=el("div","grid");g.appendChild(el("div","gutter",""));
  g.appendChild(card(null,d.chosen,true));g.appendChild(card(null,d.rejected));host.appendChild(g)}
function drawChat(host,d){
  var w=el("div","card"),b=el("div","card-b");
  d.turns.forEach(function(t){
    var r=el("div","turn "+(t.role==="assistant"?"assistant":"user"));
    r.appendChild(el("div","turn-role",t.role));r.appendChild(el("div","msg",t.content));
    b.appendChild(r)});
  w.appendChild(b);host.appendChild(w)}
function drawJudge(host,d){
  prompt(host,d.prompt,[state.rung,state.side,d.a+" vs "+d.b].join(" · "));
  host.appendChild(card(d.winner,d.response,state.side==="trained"));
  if(d.verdict)host.appendChild(el("div","verdict",d.verdict))}

/* ---- chat --------------------------------------------------------------- */
/* A turn costs ~1e-5 USD, a session ~1e0. One fixed width cannot show both. */
function usd(v){
  if(!v)return"$0";
  if(v<0.001)return"$"+v.toFixed(6);
  if(v<1)return"$"+v.toFixed(4);
  return"$"+v.toFixed(2)}

function postMode(body){
  return fetch("/api/chat/mode",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify(body)})
    .then(function(r){return r.json().then(function(j){return{ok:r.ok,body:j}})})
    .then(function(res){
      var w=$("chat-warn");
      if(!res.ok){w.hidden=false;w.textContent=res.body.error||"could not change mode";return}
      res.body.available=true;CHAT.state=res.body;drawPicks();updateSend()})}

/* One click, no restart. Going live is guarded by the budget, not by a flag. */
function setMode(live){
  if(live){
    var per=CHAT.picked.length?(CHAT.picked.length*0.0013).toFixed(4):"0.0000";
    if(!window.confirm(
      "Switch to LIVE sampling?\n\nEvery reply is billed to your Tinker account — "+
      "roughly $"+per+" per turn across the "+CHAT.picked.length+" selected model(s), "+
      "rising as the conversation grows.\n\nThe session budget stops it at $"+
      (CHAT.state.budget_usd||0).toFixed(2)+"."))return}
  postMode({execute:live})}

function drawPicks(){
  var c=CHAT.state,host=$("picks");host.textContent="";
  if(!c||!c.available){
    host.appendChild(el("span","mode dry","chat unavailable"));return}
  var mode=el("button","mode "+(c.execute?"live":"dry"),
    c.execute?"live · billable":"dry-run · free stubs");
  mode.type="button";
  mode.disabled=!c.can_execute&&!c.execute;
  mode.title=c.can_execute?"click to switch":(c.blocked_reason||"");
  mode.addEventListener("click",function(){setMode(!c.execute)});
  host.appendChild(mode);

  var bw=el("span","budget-wrap");
  bw.appendChild(el("span",null,"budget $"));
  var bi=el("input","budget");bi.type="number";bi.min="0";bi.step="0.25";
  bi.value=c.budget_usd.toFixed(2);
  bi.setAttribute("aria-label","Session budget in dollars");
  bi.addEventListener("change",function(){
    postMode({budget_usd:parseFloat(bi.value)||0})});
  bw.appendChild(bi);host.appendChild(bw);
  c.targets.forEach(function(t){
    var b=el("button","pick-btn",t.key);b.type="button";
    b.setAttribute("aria-pressed",CHAT.picked.indexOf(t.key)>=0?"true":"false");
    b.addEventListener("click",function(){
      var i=CHAT.picked.indexOf(t.key);
      if(i>=0)CHAT.picked.splice(i,1);else CHAT.picked.push(t.key);
      drawPicks();updateSend()});
    host.appendChild(b)});
  var reset=el("button","pick-btn","reset");reset.type="button";
  reset.addEventListener("click",function(){
    fetch("/api/chat/reset",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({conv:CHAT.conv})}).then(function(){
        $("log").textContent="";$("progress").textContent="";updateSend()})});
  host.appendChild(reset);
  host.appendChild(el("span","meter","",""));
  updateMeter();
  var w=$("chat-warn");
  if(c.parity&&!c.parity.matches_eval){
    w.hidden=false;w.textContent="diverges from eval parity: "+c.parity.diverged.join("; ")+
      " — replies are not comparable to the banked Elo tables"}
  else w.hidden=true}

function updateMeter(){
  var c=CHAT.state;if(!c||!c.available)return;
  var m=$("picks").querySelector(".meter");if(!m)return;
  var t=c.total||{usd:0,sample_tokens:0};
  m.textContent=c.execute
    ?usd(t.usd)+" of "+usd(c.budget_usd)+" · "+
      (t.sample_tokens||0).toLocaleString()+" tok out"
    :"no spend (dry-run) · "+(t.sample_tokens||0).toLocaleString()+" tok simulated"}

function updateSend(){
  var c=CHAT.state,ok=c&&c.available&&CHAT.picked.length&&!CHAT.busy,b=$("send");
  b.disabled=!ok;b.textContent="";
  if(CHAT.busy){b.appendChild(el("span","spin"));b.appendChild(el("span",null,"Sampling…"))}
  else b.textContent=CHAT.picked.length?"Send to "+CHAT.picked.length:"Send"}

/* One request per model, so a slow rung never holds up a fast one and each card
   resolves the moment its own reply lands. */
function chatSend(text){
  CHAT.busy=true;
  var keys=CHAT.picked.slice(),landed=0,t0=Date.now(),cells={};
  var turn=el("div","chat-turn");
  turn.appendChild(el("div","you",text));
  var wrap=el("div","replies");turn.appendChild(wrap);
  keys.forEach(function(k){
    var c=el("div","card pending");c.dataset.key=k;
    var h=el("div","reply-h");h.appendChild(el("span","chip",k));
    var e=el("span","elapsed","0.0s");h.appendChild(e);
    c.appendChild(h);c.appendChild(el("div","bar"));
    var b=el("div","card-b");b.appendChild(el("div","msg","sampling…"));c.appendChild(b);
    wrap.appendChild(c);cells[k]={card:c,clock:e}});
  $("log").appendChild(turn);
  $("log").scrollTop=$("log").scrollHeight;

  var tick=setInterval(function(){
    var s=((Date.now()-t0)/1000).toFixed(1)+"s";
    keys.forEach(function(k){
      if(cells[k].card.className.indexOf("pending")>=0)cells[k].clock.textContent=s})},100);

  function progress(){
    $("progress").textContent=landed+" / "+keys.length+" replied"}
  progress();updateSend();

  function settle(k,cls,body){
    var cell=cells[k],c=cell.card;
    c.className="card "+cls;
    var bar=c.querySelector(".bar");if(bar)bar.parentNode.removeChild(bar);
    c.querySelector(".msg").textContent=body;
    cell.clock.textContent=((Date.now()-t0)/1000).toFixed(1)+"s";
    return c}

  function one(k){
    return fetch("/api/chat/send",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({conv:CHAT.conv,message:text,keys:[k]})})
      .then(function(r){return r.json().then(function(j){return{ok:r.ok,body:j}})})
      .then(function(res){
        if(!res.ok){settle(k,"failed",res.body.error||"request failed");return}
        var rep=res.body.replies[k];
        var c=settle(k,"landed",rep.text);
        c.querySelector(".reply-h").appendChild(el("span","reply-cost",
          (rep.stub?"stub · ":usd(rep.usd)+" · ")+rep.sample_tokens+" tok"));
        CHAT.state.total=res.body.total;updateMeter()})
      .catch(function(e){settle(k,"failed",String(e))})
      .then(function(){landed++;progress();$("log").scrollTop=$("log").scrollHeight})}

  return Promise.all(keys.map(one)).then(function(){
    clearInterval(tick);CHAT.busy=false;updateSend();
    $("progress").textContent=keys.length+" replied · "+
      ((Date.now()-t0)/1000).toFixed(1)+"s total";
    $("log").scrollTop=$("log").scrollHeight})}

$("chat-form").addEventListener("submit",function(e){
  e.preventDefault();
  var box=$("msg"),text=box.value.trim();
  if(!text||CHAT.busy||!CHAT.picked.length)return;
  box.value="";chatSend(text)});
$("msg").addEventListener("keydown",function(e){
  if(e.key==="Enter"&&(e.metaKey||e.ctrlKey)){e.preventDefault();
    $("chat-form").dispatchEvent(new Event("submit",{cancelable:true}))}});

var timer=null;
$("q").addEventListener("input",function(e){
  clearTimeout(timer);var v=e.target.value;
  timer=setTimeout(function(){state.q=v;reload()},260)});
$("rung").addEventListener("change",function(e){state.rung=e.target.value;reload()});
$("side").addEventListener("change",function(e){state.side=e.target.value;reload()});
$("theme").addEventListener("click",function(){
  var r=document.documentElement;
  var dark=r.getAttribute("data-theme")==="dark"||(!r.getAttribute("data-theme")&&
    window.matchMedia("(prefers-color-scheme: dark)").matches);
  r.setAttribute("data-theme",dark?"light":"dark")});
document.addEventListener("keydown",function(e){
  if(/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName))return;
  if(state.tab==="chat"||state.sel===null)return;
  if(e.key==="j"||e.key==="ArrowDown"){
    if(state.sel<state.rows.length-1){select(state.sel+1);e.preventDefault()}}
  else if(e.key==="k"||e.key==="ArrowUp"){
    if(state.sel>0){select(state.sel-1);e.preventDefault()}}});
function fit(){
  var c=document.querySelector(".mast").offsetHeight+document.querySelector(".toolbar").offsetHeight;
  document.documentElement.style.setProperty("--chrome",c+"px")}
window.addEventListener("resize",fit);
boot().then(fit).catch(function(e){
  document.body.appendChild(el("div","empty","Could not load: "+e)) });
})();
</script>
</body>
</html>
"""


def _load_dotenv(root: Path) -> None:
    """Bare CLI runs do not source .env the way scripts/octt_plan.sh does."""
    env = root / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


def _build_chat(args):
    """Wire up the Chat tab, or return None if the octt package is not importable."""
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from octt.chat_service import ChatService
    except ImportError as exc:
        print(f"\nchat tab disabled: {exc}")
        return None

    # Always load it: the browser can flip to live mid-session, so the key has to be
    # present from the start, not only when --execute was passed.
    _load_dotenv(repo_root)
    if args.execute and not os.environ.get("TINKER_API_KEY"):
        print("\nchat tab: --execute given but TINKER_API_KEY is unset; staying dry-run")
        args.execute = False

    svc = ChatService(
        args.run,
        execute=args.execute,
        budget_usd=args.budget,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    targets = svc.targets()
    if not targets:
        print("\nchat tab: no manifests with checkpoints under this run")
        return None

    mode = "LIVE — replies are billable" if args.execute else "dry-run — replies are free stubs"
    print(f"\nchat: {len(targets)} endpoints, {mode}")
    print(f"      budget ${args.budget:.2f}, max_tokens {args.max_tokens}")
    if svc.api_key_present():
        print("      TINKER_API_KEY found — the browser can switch to live sampling")
    else:
        print("      no TINKER_API_KEY — live sampling stays unavailable")
    if not svc.parity["matches_eval"]:
        print("      NOTE: diverges from eval parity — " + "; ".join(svc.parity["diverged"]))
    return svc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, type=Path, help="sweep directory holding rung subdirs")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (localhost by default)")
    ap.add_argument("--open", action="store_true", help="open a browser once serving")
    ap.add_argument(
        "--execute",
        action="store_true",
        help="ENABLE PAID SAMPLING in the Chat tab. Without this every reply is a "
        "free stub and the tab only reports what a turn would have cost.",
    )
    ap.add_argument(
        "--budget",
        type=float,
        default=1.0,
        metavar="USD",
        help="hard ceiling on chat spend for this process (default 1.00)",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="cap on each chat reply (default 512, matching the eval)",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="chat sampling temperature (default 1.0, matching the eval)",
    )
    args = ap.parse_args(argv)

    if not args.run.is_dir():
        print(f"not a directory: {args.run}", file=sys.stderr)
        return 2

    print(f"indexing {args.run} …")
    run = Run(args.run)
    if not run.rungs:
        print(f"no rung subdirectories with artifacts under {args.run}", file=sys.stderr)
        return 2
    for r in run.rungs:
        short = r["short"]
        bits = []
        for tab in ("dpo", "introspection"):
            f = run.get(tab, short)
            if f:
                bits.append(f"{tab} {f.count():,}")
        for side in SIDES:
            f = run.get("judge", short, side)
            if f:
                bits.append(f"{side} {f.count():,}")
        print("  {:<10} {}".format(short, "  ".join(bits)))
    print(f"  shared judgment indices: {len(run.shared_indices()):,}")

    Handler.run = run
    Handler.chat = _build_chat(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"\nserving {url}   (ctrl-c to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
        run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
