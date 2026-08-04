"""Split revealed-preference caches: responses vs judgments (readiness doc, B1).

The legacy combined cache keys one row by ``(protocol, model_tag, judge_tag,
prompt, pair, condition)`` — responder evidence and judge verdict live and die
together, so *any* judge change would invalidate the banked RESPONSES and
force a paid resample of every model under test. Split them:

  ``responses.jsonl``   keyed by what produced the response: model/checkpoint
                        tag, responder sampling, embody instrument, condition,
                        prompt, ordered trait pair.
  ``judgments.jsonl``   keyed by the response *content hash* plus the judge's
                        identity: candidate traits, judge model + sampling,
                        judge instrument, parser version.

A judge-only instrument change (``validity-v2a``) then rejudges banked
responses for judge-call money only; a responder change never touches
verdicts. :func:`migrate_legacy_cache` converts a legacy combined cache into
``paper-v1``-compatible split caches offline — it never modifies the legacy
file and refuses to overwrite existing outputs.

The legacy path in :mod:`octt.evaluation` stays available and untouched (the
paper-v1 replication contract); split caching is opt-in via
``split_cache_dir``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import artifacts
from .manifest import stable_hash

RESPONSES_NAME = "responses.jsonl"
JUDGMENTS_NAME = "judgments.jsonl"
SPLIT_SCHEMA_VERSION = 1

EMBODY_INSTRUMENT_ID = "revealed-preference/paper-v1"
JUDGE_INSTRUMENT_ID = "revealed-preference/paper-v1"


# ---------------------------------------------------------------------------
# Tags and keys
# ---------------------------------------------------------------------------


def responder_tag(temperature: float, top_p: float, max_tokens: int) -> str:
    """Responder-sampling identity, byte-compatible with the legacy judge_tag tail."""
    return f"rt={temperature}|rp={top_p}|rm={max_tokens}"


def judge_only_tag(judge_model: str, temperature: float, top_p: float, max_tokens: int) -> str:
    """Judge identity (model + sampling), byte-compatible with the legacy head."""
    return f"{judge_model}|jt={temperature}|jp={top_p}|jm={max_tokens}"


def parse_legacy_judge_tag(judge_tag: str) -> tuple[str, str]:
    """Split a legacy combined tag into (judge_only_tag, responder_tag).

    Legacy format (evaluation.py): ``{judge}|jt=..|jp=..|jm=..|rt=..|rp=..|rm=..``.
    Splitting on the substrings rather than re-deriving from config keeps the
    migrated keys byte-identical to what a fresh run would compute.
    """
    parts = judge_tag.split("|")
    if (
        len(parts) != 7
        or [p.split("=")[0] for p in parts[1:]] != ["jt", "jp", "jm", "rt", "rp", "rm"]
    ):
        raise ValueError(f"unrecognized legacy judge_tag format: {judge_tag!r}")
    return "|".join(parts[:4]), "|".join(parts[4:])


def response_key(
    model_tag: str,
    resp_tag: str,
    condition: str,
    prompt: str,
    a: str,
    b: str,
    embody_instrument: str = EMBODY_INSTRUMENT_ID,
) -> str:
    return stable_hash(
        ["response", embody_instrument, model_tag, resp_tag, condition, prompt, a, b],
        length=24,
    )


def judgment_key(
    response_hash: str,
    a: str,
    b: str,
    j_tag: str,
    parser: str,
    judge_instrument: str = JUDGE_INSTRUMENT_ID,
) -> str:
    return stable_hash(
        ["judgment", judge_instrument, parser, j_tag, response_hash, a, b],
        length=24,
    )


# ---------------------------------------------------------------------------
# The split cache
# ---------------------------------------------------------------------------


class SplitEvalCache:
    """JSONL-backed response + judgment caches under one directory.

    Append-on-write like the legacy cache (a crash mid-eval never re-pays
    completed work); loading keeps the last row per key, matching legacy
    resume semantics.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.responses_path = self.cache_dir / RESPONSES_NAME
        self.judgments_path = self.cache_dir / JUDGMENTS_NAME
        self.responses: dict[str, dict[str, Any]] = {}
        self.judgments: dict[str, dict[str, Any]] = {}
        for path, table in (
            (self.responses_path, self.responses),
            (self.judgments_path, self.judgments),
        ):
            if path.exists():
                for row in artifacts.read_jsonl(path):
                    table[row["key"]] = row

    def put_response(self, row: dict[str, Any]) -> None:
        self.responses[row["key"]] = row
        artifacts.append_jsonl(self.responses_path, row)

    def put_judgment(self, row: dict[str, Any]) -> None:
        self.judgments[row["key"]] = row
        artifacts.append_jsonl(self.judgments_path, row)


def response_usable(row: dict[str, Any]) -> bool:
    """A response the judge can score. Cached EMPTY rows are terminal skips:
    the legacy no-re-pay rule — a responder that answered blank is never
    resampled on resume."""
    return row.get("status") == artifacts.STATUS_OK


def response_row(
    key: str,
    *,
    model_tag: str,
    resp_tag: str,
    condition: str,
    prompt: str,
    a: str,
    b: str,
    response: str,
    source: str = "live",
) -> dict[str, Any]:
    status = artifacts.STATUS_OK if response.strip() else artifacts.STATUS_EMPTY
    return {
        "key": key,
        "schema_version": SPLIT_SCHEMA_VERSION,
        "embody_instrument": EMBODY_INSTRUMENT_ID,
        "model_tag": model_tag,
        "responder_tag": resp_tag,
        "condition": condition,
        "prompt": prompt,
        "a": a,
        "b": b,
        "response": response,
        "response_hash": artifacts.text_hash(response),
        "status": status,
        "source": source,
    }


def judgment_row(
    key: str,
    *,
    response_hash: str,
    a: str,
    b: str,
    j_tag: str,
    parser: str,
    winner_trait: str | None,
    verdict: str | None,
    skip_reason: str | None,
    judge_attempts: int,
    discarded_verdicts: list[str],
    source: str = "live",
) -> dict[str, Any]:
    return {
        "key": key,
        "schema_version": SPLIT_SCHEMA_VERSION,
        "judge_instrument": JUDGE_INSTRUMENT_ID,
        "parser": parser,
        "judge_tag": j_tag,
        "response_hash": response_hash,
        "a": a,
        "b": b,
        "winner_trait": winner_trait,
        "verdict": verdict,
        "skip_reason": skip_reason,
        "judge_attempts": judge_attempts,
        "discarded_verdicts": discarded_verdicts,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------------


@dataclass
class MigrationReport:
    responses_written: int = 0
    judgments_written: int = 0
    empty_responses: int = 0
    skipped_dry_run: int = 0
    skipped_no_response: int = 0
    corrupt_lines: int = 0
    legacy_rows: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"legacy rows read        : {self.legacy_rows}",
            (
                f"response rows written   : {self.responses_written}"
                f" ({self.empty_responses} empty, kept retry-terminal)"
            ),
            f"judgment rows written   : {self.judgments_written}",
            f"skipped dry-run rows    : {self.skipped_dry_run}",
            f"skipped no-response rows: {self.skipped_no_response}",
            f"corrupt legacy lines    : {self.corrupt_lines}",
        ]
        return "\n".join(lines + self.notes)


def migrate_legacy_cache(legacy_path: Path, out_dir: Path) -> MigrationReport:
    """Convert one legacy combined cache into split response/judgment caches.

    Never modifies *legacy_path*; refuses to overwrite existing outputs.
    Rows carry ``source="legacy-migration"`` so banked evidence remains
    distinguishable from freshly sampled rows. Dry-run-shaped legacy rows
    (winner but no raw response — no evidence to rejudge) are counted and
    skipped, not faked.
    """
    legacy_path = Path(legacy_path)
    out_dir = Path(out_dir)
    if not legacy_path.is_file():
        raise FileNotFoundError(f"legacy cache not found: {legacy_path}")
    out_responses = out_dir / RESPONSES_NAME
    out_judgments = out_dir / JUDGMENTS_NAME
    for existing in (out_responses, out_judgments):
        if existing.exists():
            raise FileExistsError(
                f"{existing} already exists; migration never overwrites. "
                "Point --out at a fresh directory."
            )
    if legacy_path.parent == out_dir:
        raise ValueError(
            "refusing to write split caches next to the legacy file; "
            "use a separate output directory"
        )

    report = MigrationReport()
    responses: dict[str, dict[str, Any]] = {}
    judgments: dict[str, dict[str, Any]] = {}

    import json

    with open(legacy_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                row["key"]  # legacy loader required this too
            except (json.JSONDecodeError, KeyError):
                report.corrupt_lines += 1
                continue
            report.legacy_rows += 1
            response = row.get("response")
            if response is None:
                if row.get("winner_trait") is not None and not row.get("judge_attempts"):
                    report.skipped_dry_run += 1
                else:
                    report.skipped_no_response += 1
                continue
            j_tag, resp_tag = parse_legacy_judge_tag(row["judge_tag"])
            condition = row.get("condition", "adopt")
            rkey = response_key(
                row["model_tag"], resp_tag, condition, row["prompt"], row["a"], row["b"]
            )
            rrow = response_row(
                rkey,
                model_tag=row["model_tag"],
                resp_tag=resp_tag,
                condition=condition,
                prompt=row["prompt"],
                a=row["a"],
                b=row["b"],
                response=response,
                source="legacy-migration",
            )
            if rrow["status"] == artifacts.STATUS_EMPTY:
                report.empty_responses += 1
            responses[rkey] = rrow  # legacy loader keeps the last duplicate

            if not response.strip():
                continue  # empty response: the judge was never called
            parser = row.get("protocol_version", "unknown-legacy-protocol")
            jkey = judgment_key(rrow["response_hash"], row["a"], row["b"], j_tag, parser)
            judgments[jkey] = judgment_row(
                jkey,
                response_hash=rrow["response_hash"],
                a=row["a"],
                b=row["b"],
                j_tag=j_tag,
                parser=parser,
                winner_trait=row.get("winner_trait"),
                verdict=row.get("verdict"),
                skip_reason=row.get("skip_reason"),
                judge_attempts=row.get("judge_attempts", 1),
                discarded_verdicts=row.get("discarded_verdicts", []),
                source="legacy-migration",
            )

    report.responses_written = len(responses)
    report.judgments_written = len(judgments)
    artifacts.write_jsonl_atomic(out_responses, responses.values())
    artifacts.write_jsonl_atomic(out_judgments, judgments.values())
    return report
