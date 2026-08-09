"""Phase 3 (Best-of-N) artifact schema — readiness doc, work package 4.

The readiness doc's logging requirement is a list, and the list is the schema:

    log every candidate, pairwise comparison, swap result, reward component,
    length, marker hit, selected candidate ID, and selection tournament.

Every item on that list is here, in four row types, because each answers a
different question after the fact:

``candidate``
    One sampled reply, with its independent measures (length, marker hit and
    density, repetition, format compliance, language match, truncation) and the
    reward components it earned at each N. This is the row that lets someone
    ask "what did selection actually pick, and how was it different?"

``comparison``
    One ORDERED judge call: which candidate was on the left, which on the
    right, and what came back. 240 of these per cell at N=16. Kept separately
    from the resolved pair because "the judge answered differently when we
    swapped the order" is only visible if both calls survive individually.

``swap``
    One unordered pair after order-swap resolution: both ordered verdicts, the
    resolution, the reason (agreement / both-tie / disagreement / unparseable),
    the score, and both lengths. This is the row that quantifies how much of the
    proxy's signal is position bias.

``selection``
    One (cell, N) tournament: the full per-candidate score / win / tie / loss
    table, the dropped-pair count, the selected candidate id, and the tie-break
    rule that produced it. Recomputable from the swap rows, and stored anyway:
    an audit that cannot reproduce its own selection is not an audit.

Provenance follows :mod:`octt.artifacts`: deterministic ``request_id`` built
from scientific identity only (never timestamps or local paths), the instrument
id AND its content hash on every row, and a status field so empty or errored
rows stay retryable instead of counting as done. Rows also carry
``execution_mode``, because a dry-run stub bank and a sampled bank must never be
merged by accident.

Pure bookkeeping: no tinker, no network, no heavy imports.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import artifacts, manifest

PHASE3_SCHEMA_VERSION = 1

ROW_CANDIDATE = "candidate"
ROW_COMPARISON = "comparison"
ROW_SWAP = "swap"
ROW_SELECTION = "selection"
ROW_TYPES = (ROW_CANDIDATE, ROW_COMPARISON, ROW_SWAP, ROW_SELECTION)

#: File name per row type inside a bundle directory.
BUNDLE_FILES = {
    ROW_CANDIDATE: "candidates.jsonl",
    ROW_COMPARISON: "comparisons.jsonl",
    ROW_SWAP: "swaps.jsonl",
    ROW_SELECTION: "selections.jsonl",
}
BUNDLE_MANIFEST = "phase3_manifest.json"

#: Independent measures every candidate row must carry. Named here so a row
#: that silently stopped recording one (a renamed key, a dropped measure) fails
#: validation instead of quietly shrinking the gate's evidence.
REQUIRED_MEASURES = (
    "length_chars",
    "length_words",
    "marker_hit",
    "marker_count",
    "marker_density_per_100w",
    "latin_scoreable",
    "script",
    "script_rule",
    "marker_instrument",
    "repetition_score",
    "format_rule",
    "format_compliant",
    "correctness_rule",
    "correct",
    "target_language",
    "language_match",
    "truncated",
)

_COMMON = (
    "row_type",
    "phase3_schema_version",
    "request_id",
    "instrument_id",
    "instrument_hash",
    "execution_mode",
)

CANDIDATE_ROW_FIELDS = (
    *artifacts.REQUIRED_ROW_FIELDS,
    *_COMMON,
    "panel_id",
    "panel_hash",
    "prompt_id",
    "category",
    "policy_id",
    "cell_id",
    "candidate_id",
    "candidate_index",
    "response",
    "response_hash",
    "measures",
    "reward_components",
)

COMPARISON_ROW_FIELDS = (
    *_COMMON,
    "status",
    "cell_id",
    "prompt_id",
    "pair_id",
    "presentation",
    "left_candidate_id",
    "right_candidate_id",
    "verdict",
    "parser",
    "renderer",
    "judge_model",
    "character_brief_id",
    "character_brief_hash",
)

SWAP_ROW_FIELDS = (
    *_COMMON,
    "cell_id",
    "prompt_id",
    "pair_id",
    "candidate_a",
    "candidate_b",
    "index_a",
    "index_b",
    "verdict_ab",
    "verdict_ba",
    "presentation_order",
    "resolution",
    "resolution_reason",
    "swap_consistent",
    "score_a",
    "len_a",
    "len_b",
    "length_ratio",
    "character_brief_id",
    "character_brief_hash",
)

SELECTION_ROW_FIELDS = (
    *_COMMON,
    "cell_id",
    "prompt_id",
    "policy_id",
    "n",
    "selected_index",
    "selected_candidate_id",
    "proxy_score",
    "dropped_pairs",
    "tie_break_rule",
    "tournament",
)

_REQUIRED_BY_TYPE = {
    ROW_CANDIDATE: CANDIDATE_ROW_FIELDS,
    ROW_COMPARISON: COMPARISON_ROW_FIELDS,
    ROW_SWAP: SWAP_ROW_FIELDS,
    ROW_SELECTION: SELECTION_ROW_FIELDS,
}


class Phase3SchemaError(ValueError):
    """A Phase 3 row does not satisfy the logging contract."""


# ---------------------------------------------------------------------------
# Row constructors
# ---------------------------------------------------------------------------


def candidate_row(
    *,
    panel_id: str,
    panel_hash: str,
    prompt_id: str,
    prompt_text: str,
    category: str | None,
    policy_id: str,
    model_id: str,
    checkpoint_role: str,
    checkpoint_fingerprint: str,
    candidate_index: int,
    candidate_id: str,
    response: str,
    measures: Mapping[str, Any],
    instrument_id: str,
    instrument_hash: str,
    renderer: str,
    sampling: Mapping[str, Any],
    execution_mode: str,
    reward_components: Mapping[str, Any] | None = None,
    recipe_version: str | None = None,
) -> dict[str, Any]:
    """One generated candidate, with its measures and its reward components.

    The candidate INDEX is part of the request id: the bank is nested, so index
    is what distinguishes the 16 draws of one cell and what every reported N is
    a prefix of. Two runs of the same audit produce the same id.
    """
    missing = [m for m in REQUIRED_MEASURES if m not in measures]
    if missing:
        raise Phase3SchemaError(
            f"candidate {candidate_id} is missing independent measures: {missing}"
        )
    rid = artifacts.request_id(
        {
            "kind": ROW_CANDIDATE,
            "instrument": instrument_id,
            "instrument_hash": instrument_hash,
            "panel_hash": panel_hash,
            "prompt_id": prompt_id,
            "policy_id": policy_id,
            "model_id": model_id,
            "checkpoint": checkpoint_fingerprint,
            "candidate_index": candidate_index,
            "sampling": dict(sampling),
        }
    )
    text = response or ""
    return {
        "row_type": ROW_CANDIDATE,
        "phase3_schema_version": PHASE3_SCHEMA_VERSION,
        "schema_version": artifacts.ARTIFACT_SCHEMA_VERSION,
        "request_id": rid,
        "instrument_id": instrument_id,
        "instrument_hash": instrument_hash,
        "model_id": model_id,
        "checkpoint_role": checkpoint_role,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "renderer": renderer,
        "sampling": dict(sampling),
        "prompt_hash": artifacts.text_hash(prompt_text),
        "status": artifacts.STATUS_OK if text.strip() else artifacts.STATUS_EMPTY,
        "panel_id": panel_id,
        "panel_hash": panel_hash,
        "prompt_id": prompt_id,
        "category": category,
        "policy_id": policy_id,
        "cell_id": f"{prompt_id}::{policy_id}",
        "candidate_id": candidate_id,
        "candidate_index": candidate_index,
        "response": text,
        "response_hash": artifacts.text_hash(text),
        "measures": dict(measures),
        "reward_components": dict(reward_components or {}),
        "execution_mode": execution_mode,
        "dry_run_recipe_version": recipe_version,
    }


def comparison_rows(
    verdict: Mapping[str, Any],
    *,
    judge_model: str,
    execution_mode: str,
) -> list[dict[str, Any]]:
    """The TWO ordered judge calls behind one resolved pair.

    One row per presentation, so the 240 ordered comparisons of a full cell all
    exist individually. The raw judge text is not persisted (no judge in this
    repo persists it); what is persisted is the parsed verdict plus the parser
    version that produced it, and an unparseable call is ``verdict: null`` with
    ``status="empty"`` — missing data, never a recorded tie.
    """
    out: list[dict[str, Any]] = []
    sides = {
        "ab": (verdict["candidate_a"], verdict["candidate_b"], verdict.get("verdict_ab")),
        "ba": (verdict["candidate_b"], verdict["candidate_a"], verdict.get("verdict_ba")),
    }
    for presentation, (left, right, tag) in sides.items():
        rid = artifacts.request_id(
            {
                "kind": ROW_COMPARISON,
                "instrument": verdict["instrument_id"],
                "instrument_hash": verdict["instrument_hash"],
                "brief": verdict["character_brief_hash"],
                "judge_model": judge_model,
                "pair": verdict["pair_id"],
                "presentation": presentation,
            }
        )
        out.append(
            {
                "row_type": ROW_COMPARISON,
                "phase3_schema_version": PHASE3_SCHEMA_VERSION,
                "request_id": rid,
                "instrument_id": verdict["instrument_id"],
                "instrument_hash": verdict["instrument_hash"],
                "execution_mode": execution_mode,
                "status": (
                    artifacts.STATUS_OK if tag is not None else artifacts.STATUS_EMPTY
                ),
                "cell_id": verdict["cell_id"],
                "prompt_id": verdict["prompt_id"],
                "pair_id": verdict["pair_id"],
                "presentation": presentation,
                "left_candidate_id": left,
                "right_candidate_id": right,
                "verdict": tag,
                "parser": verdict["parser"],
                "renderer": verdict["renderer"],
                "judge_model": judge_model,
                "character_brief_id": verdict["character_brief_id"],
                "character_brief_hash": verdict["character_brief_hash"],
            }
        )
    return out


def swap_row(
    verdict: Mapping[str, Any], *, execution_mode: str
) -> dict[str, Any]:
    """One unordered pair after order-swap resolution.

    ``score_a`` is the reward component the pair contributed to candidate A
    (1 / 0.5 / 0, or ``None`` when the pair was dropped). ``swap_consistent``
    is the diagnostic that matters most for a reward proxy: a low rate means
    most of what the judge said was position, not preference.
    """
    rid = artifacts.request_id(
        {
            "kind": ROW_SWAP,
            "instrument": verdict["instrument_id"],
            "instrument_hash": verdict["instrument_hash"],
            "brief": verdict["character_brief_hash"],
            "pair": verdict["pair_id"],
        }
    )
    return {
        "row_type": ROW_SWAP,
        "phase3_schema_version": PHASE3_SCHEMA_VERSION,
        "request_id": rid,
        "instrument_id": verdict["instrument_id"],
        "instrument_hash": verdict["instrument_hash"],
        "execution_mode": execution_mode,
        "cell_id": verdict["cell_id"],
        "prompt_id": verdict["prompt_id"],
        "pair_id": verdict["pair_id"],
        "candidate_a": verdict["candidate_a"],
        "candidate_b": verdict["candidate_b"],
        "index_a": verdict["index_a"],
        "index_b": verdict["index_b"],
        "verdict_ab": verdict.get("verdict_ab"),
        "verdict_ba": verdict.get("verdict_ba"),
        "presentation_order": list(verdict["presentation_order"]),
        "resolution": verdict["resolution"],
        "resolution_reason": verdict["resolution_reason"],
        "swap_consistent": verdict["swap_consistent"],
        "score_a": verdict["score_a"],
        "len_a": verdict["len_a"],
        "len_b": verdict["len_b"],
        "length_ratio": verdict["length_ratio"],
        "character_brief_id": verdict["character_brief_id"],
        "character_brief_hash": verdict["character_brief_hash"],
    }


def selection_row(
    selection: Mapping[str, Any],
    *,
    instrument_id: str,
    instrument_hash: str,
    execution_mode: str,
) -> dict[str, Any]:
    """One (cell, N) selection tournament, in full.

    ``selection`` is :meth:`octt.best_of_n.Selection.to_dict`. The per-candidate
    score / win / tie / loss vectors ARE the reward components, kept alongside
    the selected id so the selection can be re-derived and checked.
    """
    rid = artifacts.request_id(
        {
            "kind": ROW_SELECTION,
            "instrument": instrument_id,
            "instrument_hash": instrument_hash,
            "cell": selection["cell_id"],
            "n": selection["n"],
        }
    )
    return {
        "row_type": ROW_SELECTION,
        "phase3_schema_version": PHASE3_SCHEMA_VERSION,
        "request_id": rid,
        "instrument_id": instrument_id,
        "instrument_hash": instrument_hash,
        "execution_mode": execution_mode,
        "cell_id": selection["cell_id"],
        "prompt_id": selection["prompt_id"],
        "policy_id": selection["policy_id"],
        "n": selection["n"],
        "selected_index": selection["selected_index"],
        "selected_candidate_id": selection["selected_candidate_id"],
        "proxy_score": selection["proxy_score"],
        "dropped_pairs": selection["dropped_pairs"],
        "tie_break_rule": selection["tie_break_rule"],
        "tournament": {
            "scores": list(selection["scores"]),
            "wins": list(selection["wins"]),
            "ties": list(selection["ties"]),
            "losses": list(selection["losses"]),
            "comparisons": list(selection["comparisons"]),
        },
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_row(row: Mapping[str, Any]) -> None:
    """Raise :class:`Phase3SchemaError` unless *row* satisfies its contract."""
    row_type = row.get("row_type")
    if row_type not in ROW_TYPES:
        raise Phase3SchemaError(f"unknown Phase 3 row_type {row_type!r}; expected {ROW_TYPES}")
    if row.get("phase3_schema_version") != PHASE3_SCHEMA_VERSION:
        raise Phase3SchemaError(
            f"row phase3_schema_version {row.get('phase3_schema_version')!r} != "
            f"{PHASE3_SCHEMA_VERSION} (no silent cross-version merges)"
        )
    missing = [f for f in _REQUIRED_BY_TYPE[row_type] if f not in row]
    if missing:
        raise Phase3SchemaError(f"{row_type} row missing required fields: {missing}")
    if row.get("execution_mode") not in (
        manifest.EXECUTION_MODE_DRY_RUN,
        manifest.EXECUTION_MODE_REAL,
    ):
        raise Phase3SchemaError(
            f"{row_type} row has unknown execution_mode {row.get('execution_mode')!r}"
        )
    if row_type == ROW_CANDIDATE:
        # Candidates are sampled rows: they must also satisfy the repo-wide
        # artifact contract (status/response consistency, full provenance).
        artifacts.validate_row(row)
        missing_measures = [m for m in REQUIRED_MEASURES if m not in row["measures"]]
        if missing_measures:
            raise Phase3SchemaError(
                f"candidate row missing measures: {missing_measures}"
            )
    if row_type == ROW_SWAP and row["resolution"] is None and row["score_a"] is not None:
        raise Phase3SchemaError(
            "a dropped swap must have score_a=None; an unanswered comparison is "
            "missing data, not a measured tie"
        )
    if row_type == ROW_SELECTION:
        table = row["tournament"]
        n = row["n"]
        for key in ("scores", "wins", "ties", "losses"):
            if len(table[key]) != n:
                raise Phase3SchemaError(
                    f"selection row at N={n} has {len(table[key])} {key} entries"
                )
        if not 0 <= row["selected_index"] < n:
            raise Phase3SchemaError(
                f"selection row at N={n} selected index {row['selected_index']}"
            )


def validate_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        validate_row(row)


class Phase3Incomplete(AssertionError):
    """A bundle does not contain the rows the audit protocol requires."""


def assert_bundle_complete(
    rows_by_type: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    cells: int,
    candidates_per_cell: int,
    ordered_comparisons_per_cell: int,
    n_ladder: Sequence[int],
) -> None:
    """Fail loudly unless every cell logged every row the protocol promises.

    Counting is the cheapest possible integrity check and it catches the failure
    that matters: a run that crashed halfway and still wrote a plausible-looking
    bundle. A short bundle is not a smaller experiment, it is a different one.
    """
    expected = {
        ROW_CANDIDATE: cells * candidates_per_cell,
        ROW_COMPARISON: cells * ordered_comparisons_per_cell,
        ROW_SWAP: cells * ordered_comparisons_per_cell // 2,
        ROW_SELECTION: cells * len(n_ladder),
    }
    actual = {t: len(rows_by_type.get(t, ())) for t in ROW_TYPES}
    if actual != expected:
        raise Phase3Incomplete(
            f"incomplete Phase 3 bundle: expected {expected}, got {actual}"
        )
    ids = [r["request_id"] for rows in rows_by_type.values() for r in rows]
    if len(set(ids)) != len(ids):
        raise Phase3Incomplete(
            "duplicate request_id across the bundle: two rows claim one identity"
        )


# ---------------------------------------------------------------------------
# Bundle I/O
# ---------------------------------------------------------------------------


def write_bundle(
    out_dir: Path,
    rows_by_type: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    header: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one JSONL per row type plus a manifest, atomically.

    The manifest records the instruments, the panel hash, the execution mode,
    and the row counts, so a bundle can be identified without reading it and a
    truncated file is detectable by counting.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for row_type in ROW_TYPES:
        rows = list(rows_by_type.get(row_type, ()))
        validate_rows(rows)
        artifacts.write_jsonl_atomic(out_dir / BUNDLE_FILES[row_type], rows)
        counts[row_type] = len(rows)
    payload = {
        "phase3_schema_version": PHASE3_SCHEMA_VERSION,
        "artifact_schema_version": artifacts.ARTIFACT_SCHEMA_VERSION,
        "counts": counts,
        "files": dict(BUNDLE_FILES),
        **dict(header),
    }
    manifest.atomic_write_json(out_dir / BUNDLE_MANIFEST, payload)
    return payload


def read_bundle(out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Read a bundle back, validating every row (a corrupt line raises)."""
    out_dir = Path(out_dir)
    bundle: dict[str, list[dict[str, Any]]] = {}
    for row_type, name in BUNDLE_FILES.items():
        path = out_dir / name
        rows = artifacts.read_jsonl(path) if path.exists() else []
        validate_rows(rows)
        bundle[row_type] = rows
    return bundle
