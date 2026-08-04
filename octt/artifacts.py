"""Artifact and cache row contract (readiness doc, B0).

Every generated row — a sampled response, a judge verdict, a graded completion
— must carry enough provenance to be re-identified, resumed, and audited
without trusting file order or directory layout:

  - a deterministic ``request_id`` derived only from scientific content
    (instrument, prompt hash, model, checkpoint fingerprint, sampling) —
    timestamps and absolute local paths are structurally excluded;
  - the instrument id AND its content hash, so a row is verifiably tied to the
    exact prompt text that produced it;
  - a status field: empty or errored rows never count as complete and stay
    retryable, while two *complete* rows for one request that disagree are a
    conflict and fatal, never silently resolved.

Hashing is Unicode-safe canonical JSON (sorted keys, no ASCII escaping, UTF-8),
shared with :mod:`octt.manifest` so config hashes and artifact hashes agree.
This module is pure bookkeeping: no tinker, no network.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .manifest import _canonical

ARTIFACT_SCHEMA_VERSION = 1

STATUS_OK = "ok"
STATUS_EMPTY = "empty"  # sampled, came back blank — retryable
STATUS_ERROR = "error"  # request failed — retryable
STATUSES = (STATUS_OK, STATUS_EMPTY, STATUS_ERROR)

# Provenance every row must carry (the "artifact and cache contract").
REQUIRED_ROW_FIELDS = (
    "schema_version",
    "request_id",
    "instrument_id",
    "instrument_hash",
    "model_id",
    "checkpoint_role",  # e.g. "trained" / "base" / "judge"
    "checkpoint_fingerprint",  # tinker:// URI, adapter content hash, or "base"
    "renderer",
    "sampling",
    "prompt_hash",
    "status",
)

# Keys that must never feed a scientific hash: they vary across reruns/hosts
# without changing what was measured.
_BANNED_ID_KEYS = frozenset(
    {"timestamp", "created_at", "updated_at", "recorded_at", "time", "now",
     "path", "run_dir", "out_dir", "cache_path", "host", "pid"}
)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON for hashing: sorted keys, UTF-8 text kept as-is."""
    return json.dumps(
        _canonical(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def content_hash(obj: Any) -> str:
    """Full sha256 over canonical JSON (Unicode-safe)."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    """Hash of one exact text (prompt or response), Unicode-safe."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class BannedIdentityKey(ValueError):
    """A request-id part that would make the id nondeterministic across runs."""


def request_id(parts: Mapping[str, Any], *, length: int = 24) -> str:
    """Deterministic request id from scientific identity parts only.

    Rejects keys that smuggle run-local state (timestamps, absolute local
    paths, hosts) into the identity: two runs of the same science must produce
    the same id, and nothing else may collide with it. ``tinker://`` handles
    are stable checkpoint identities and are allowed; local filesystem paths
    are not — pass a content fingerprint instead.
    """
    for key, value in parts.items():
        if key.lower() in _BANNED_ID_KEYS:
            raise BannedIdentityKey(f"request-id part {key!r} is run-local state")
        if isinstance(value, str) and os.path.isabs(value) and not value.startswith(
            "tinker://"
        ):
            raise BannedIdentityKey(
                f"request-id part {key!r} looks like a local path ({value!r}); "
                "use a content fingerprint or alias instead"
            )
    return hashlib.sha256(canonical_json(dict(parts)).encode("utf-8")).hexdigest()[:length]


def validate_row(row: Mapping[str, Any]) -> None:
    """Raise ValueError unless *row* satisfies the artifact contract."""
    missing = [f for f in REQUIRED_ROW_FIELDS if f not in row]
    if missing:
        raise ValueError(f"artifact row missing required fields: {', '.join(missing)}")
    if row["status"] not in STATUSES:
        raise ValueError(f"artifact row has unknown status {row['status']!r}")
    if row["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"artifact row schema_version {row['schema_version']!r} != "
            f"{ARTIFACT_SCHEMA_VERSION} (no silent cross-version merges)"
        )
    if row["status"] == STATUS_OK and not str(row.get("response") or "").strip():
        raise ValueError(
            f"artifact row {row['request_id']} is status=ok with an empty "
            "response; empty rows must be status=empty"
        )


def is_complete(row: Mapping[str, Any]) -> bool:
    """Only a non-empty, successful row counts as done (resume re-pays the rest)."""
    return row.get("status") == STATUS_OK and bool(str(row.get("response") or "").strip())


class MergeConflict(ValueError):
    """Two complete rows claim the same request with different content."""


def merge_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Merge rows by request_id with fail-fast conflict semantics.

    A complete row always replaces an incomplete one for the same request.
    Two complete rows must agree on ``response_hash`` (byte-identical
    responses); disagreement is a :class:`MergeConflict`, never a silent pick.
    Incomplete duplicates keep the latest row (retries overwrite errors).
    """
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        validate_row(row)
        rid = row["request_id"]
        existing = merged.get(rid)
        if existing is None:
            merged[rid] = dict(row)
            continue
        new_done, old_done = is_complete(row), is_complete(existing)
        if new_done and old_done:
            if row.get("response_hash") != existing.get("response_hash"):
                raise MergeConflict(
                    f"request {rid} has two complete rows with different "
                    "responses; refusing to choose"
                )
            continue  # identical completes: keep the first
        if new_done and not old_done:
            merged[rid] = dict(row)
        elif not new_done and not old_done:
            merged[rid] = dict(row)  # latest retry wins among incompletes
        # complete beats a later incomplete: keep existing
    return merged


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Strict JSONL read: a corrupt line is data loss and raises, not skips."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: corrupt JSONL row: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Whole-file JSONL write via tmp+rename so a crash never truncates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
