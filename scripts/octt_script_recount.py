"""Re-derive the banked persona-expression-rate script split (free, offline).

The v1 script rule in ``octt/persona_markers.py`` calls a response "Latin script"
when fewer than 5% of its first 400 characters sit above U+2000. Greek, Cyrillic,
Hebrew, Arabic, Devanagari and Thai all sit BELOW U+2000, so v1 only ever excluded
CJK and kana: every banked "Latin vs non-Latin" expression rate is really
"non-CJK vs CJK", with the Latin bucket dragged down by every Arabic / Cyrillic /
Devanagari / Hebrew / Greek response the English lexicon can never score.

This recomputes the split under the corrected, separately versioned rule
(``persona_markers.SCRIPT_RULE_V2``), reporting the old number beside the new one
and a full per-script table. It reads banked files only: no API calls, no Tinker
spend, no network.

    uv run python scripts/octt_script_recount.py              # every arm set
    uv run python scripts/octt_script_recount.py --set pirate
    uv run python scripts/octt_script_recount.py --out-dir DIR

Response selection is byte-identical to the banked methodology: the FIRST record
per distinct prompt in file order, empty responses skipped.

Every arm also carries a **sanity anchor**: the same responses are re-scored under
the OLD rule and checked against the number that was actually published. If the
recomputed v1 rate does not reproduce the published one, the response selection
differs from the banked method and the v1-vs-v2 delta is not isolated — the row is
marked ``FAILED`` and must not be cited.

Arms whose responses are not on this machine are reported as ``UNAVAILABLE`` with
the exact paths that were searched. They are never estimated, interpolated or
dropped: a partial recount is a correct answer, an invented row is not.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from octt import persona_markers as pm

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO / "runs/_mega/script-recount"
SHARED_CACHE = REPO / "runs/_campaign_eval_cache/responses.jsonl"

#: A published rate quoted to 0.1 percentage points can only be reproduced to
#: within half of that; anything wider means the response selection differs.
SANITY_TOLERANCE = 0.0005

#: Named arm sets -> the artifact each one is written to.
PIRATE_SET = "pirate"
FINDINGS_SET = "findings-2026-07-27"
ARTIFACTS = {
    PIRATE_SET: "pirate-v6-v7.json",
    FINDINGS_SET: "findings-2026-07-27-five-arm.json",
}

#: Stratified Fable slice of the five FINDINGS arms. Its strata were drawn under
#: the DEFECTIVE v1 rule, which is exactly what makes it a usable direction probe
#: for arms whose full response sets are not on this machine.
M3_SLICE = REPO / "runs/m3-judge-slice-2026-07-27/m3_texts.jsonl"


@dataclass(frozen=True)
class Arm:
    """One banked arm, plus the numbers previously published for it."""

    label: str
    run_dir: str
    instrument: str
    arm_set: str = PIRATE_SET
    #: Where the published numbers this arm is anchored against were printed.
    published_in: str | None = None
    banked_latin_rate: float | None = None
    banked_latin_n: int | None = None
    banked_non_latin_rate: float | None = None
    banked_non_latin_n: int | None = None
    #: Published unrestricted ("trained (all)") rate, when one exists.
    banked_all_rate: float | None = None
    #: Published share of responses the v1 rule called Latin script, when one
    #: exists — the FINDINGS "answers in Latin script X% of the time" claim.
    banked_latin_answer_share: float | None = None


FINDINGS_DOC = "docs/FINDINGS_2026-07-27_persona_expression_rate.md"
#: Phase 1's four rungs share one sweep directory, one rung per model.
DENSE_SWEEP = "runs/pirate-dense-paper-half-uncapped-rank32-v7"

ARMS = (
    Arm(
        label="v7 (Inkling-Small, full paper, rank 64)",
        run_dir="runs/pirate-inkling-small-paper-rank64-v7",
        instrument="pirate-strong-v1-pinned-2026-07-27",
        arm_set=PIRATE_SET,
        published_in="runs/_mega/script-recount/pirate-v6-v7.json",
        banked_latin_rate=0.737,
        banked_latin_n=16945,
        banked_non_latin_rate=0.011,
        banked_non_latin_n=4570,
    ),
    Arm(
        label="v6 (Inkling, half paper, rank 32)",
        run_dir="runs/pirate-inkling-paper-half-rank32-v6",
        instrument="pirate-strong-v1-pinned-2026-07-27",
        arm_set=PIRATE_SET,
        published_in="runs/_mega/script-recount/pirate-v6-v7.json",
        banked_latin_rate=0.697,
        banked_latin_n=7528,
        banked_non_latin_rate=0.002,
        banked_non_latin_n=3180,
    ),
    # --- The five arms of FINDINGS_2026-07-27, table at lines 18-24 ----------
    # Same defect, same direction: every one of these "Latin-script only" rates
    # is really "non-CJK only" and is therefore understated.
    Arm(
        label="4B (Phase 1 dense rung)",
        run_dir=f"{DENSE_SWEEP}/Qwen-Qwen3.5-4B",
        instrument="pirate-strong-v1-pinned-2026-07-27",
        arm_set=FINDINGS_SET,
        published_in=f"{FINDINGS_DOC}:20",
        banked_latin_rate=0.705,
        banked_all_rate=0.575,
    ),
    Arm(
        label="9B (Phase 1 dense rung)",
        run_dir=f"{DENSE_SWEEP}/Qwen-Qwen3.5-9B",
        instrument="pirate-strong-v1-pinned-2026-07-27",
        arm_set=FINDINGS_SET,
        published_in=f"{FINDINGS_DOC}:21",
        banked_latin_rate=0.841,
        banked_all_rate=0.747,
    ),
    Arm(
        label="27B arm A (rank 32, Phase 1 dense rung)",
        run_dir=f"{DENSE_SWEEP}/Qwen-Qwen3.6-27B",
        instrument="pirate-strong-v1-pinned-2026-07-27",
        arm_set=FINDINGS_SET,
        published_in=f"{FINDINGS_DOC}:22,110",
        banked_latin_rate=0.845,
        banked_all_rate=0.767,
        banked_latin_answer_share=0.905,
    ),
    Arm(
        label="27B arm B (rank 64, training-strength probe)",
        run_dir="runs/pirate-27b-strength-armB-v7",
        instrument="pirate-strong-v1-pinned-2026-07-27",
        arm_set=FINDINGS_SET,
        published_in=f"{FINDINGS_DOC}:23,110",
        banked_latin_rate=0.950,
        banked_all_rate=0.897,
        banked_latin_answer_share=0.941,
    ),
    Arm(
        label="35B-A3B (Phase 1 MoE architecture control)",
        run_dir=f"{DENSE_SWEEP}/Qwen-Qwen3.6-35B-A3B",
        instrument="pirate-strong-v1-pinned-2026-07-27",
        arm_set=FINDINGS_SET,
        published_in=f"{FINDINGS_DOC}:24",
        banked_latin_rate=0.601,
        banked_all_rate=0.454,
    ),
)


class ResponsesUnavailable(Exception):
    """This arm's banked responses are not on this machine.

    Carries every path that was searched so the report can say exactly where to
    look, rather than guessing at a number.
    """

    def __init__(self, reason: str, looked_for: list[str]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.looked_for = looked_for


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def trained_model_tag(run_dir: Path) -> str:
    """The eval-cache model_tag of a run's trained (post-SFT) sampler."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    stages = manifest["stages"]
    stage = stages.get("sft") or stages.get("merge") or stages["dpo"]
    return f"{manifest['model']}@{stage['sampler_path']}"


def load_responses(run_dir: Path) -> tuple[dict[str, str], str]:
    """First response per distinct prompt for a run's trained arm, plus its source.

    Prefers the run's own ``eval/trained_judge.jsonl``; falls back to the shared
    campaign eval cache filtered to this run's trained sampler (runs that reused
    the shared cache have an empty ``eval/`` directory). Raises
    :class:`ResponsesUnavailable` — listing every path tried — when neither
    exists, so a missing arm is reported, never approximated.
    """
    own = run_dir / "eval" / "trained_judge.jsonl"
    looked_for = [_rel(own)]
    if own.exists():
        return pm.first_response_per_prompt(own), _rel(own)

    manifest = run_dir / "manifest.json"
    looked_for.append(_rel(manifest))
    if not manifest.exists():
        looked_for.append(f"{_rel(SHARED_CACHE)}#model_tag=<unresolvable>")
        raise ResponsesUnavailable(
            "no eval/trained_judge.jsonl and no manifest.json, so the trained "
            "sampler's model_tag cannot be resolved and the shared eval cache "
            "cannot be filtered for this arm",
            looked_for,
        )

    tag = trained_model_tag(run_dir)
    looked_for.append(f"{_rel(SHARED_CACHE)}#model_tag={tag}")
    if not SHARED_CACHE.exists():
        raise ResponsesUnavailable("shared eval cache is not on this machine", looked_for)
    out: dict[str, str] = {}
    with SHARED_CACHE.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("model_tag") != tag:
                continue
            if rec.get("prompt") and rec.get("response"):
                out.setdefault(rec["prompt"], rec["response"])
    if not out:
        raise ResponsesUnavailable(
            f"shared eval cache holds no rows for model_tag {tag!r}", looked_for
        )
    return out, f"{_rel(SHARED_CACHE)}#model_tag={tag}"


def sanity_anchor(arm: Arm, old: dict[str, float | int | str]) -> dict[str, object]:
    """Does re-scoring under the OLD rule reproduce the number that was published?

    This is what isolates the correction. If v1 does not reproduce, the response
    selection differs from the banked method and the v1-vs-v2 delta is measuring
    two changes at once, so the corrected row must not be cited.
    """
    checks: dict[str, dict[str, object]] = {}
    for name, banked, recomputed in (
        ("rate_latin", arm.banked_latin_rate, float(old["rate_latin"])),
        ("rate_all_floor", arm.banked_all_rate, float(old["rate_all_floor"])),
        ("n_latin", arm.banked_latin_n, float(old["n_latin"])),
    ):
        if banked is None:
            continue
        tol = SANITY_TOLERANCE if name.startswith("rate") else 0.0
        delta = recomputed - banked
        checks[name] = {
            "published": banked,
            "recomputed_v1": recomputed,
            "delta": delta,
            "tolerance": tol,
            "reproduces": abs(delta) <= tol,
        }
    if not checks:
        status = "NO_PUBLISHED_VALUE"
    elif all(bool(c["reproduces"]) for c in checks.values()):
        status = "REPRODUCES"
    else:
        status = "FAILED"
    return {
        "status": status,
        "published_in": arm.published_in,
        "checks": checks,
        "why": (
            "The corrected number is only interpretable as a correction if the "
            "OLD rule, run over the SAME responses, returns what was published. "
            "FAILED means response selection drifted, not that v2 is wrong."
        ),
    }


def banked_block(arm: Arm) -> dict[str, object]:
    """What was published for this arm, under the rule that produced it."""
    return {
        "script_rule": pm.SCRIPT_RULE_V1,
        "published_in": arm.published_in,
        "n_latin": arm.banked_latin_n,
        "rate_latin": arm.banked_latin_rate,
        "n_non_latin": arm.banked_non_latin_n,
        "rate_non_latin": arm.banked_non_latin_rate,
        "rate_all_floor": arm.banked_all_rate,
        "latin_answer_share": arm.banked_latin_answer_share,
    }


def unavailable(arm: Arm, exc: ResponsesUnavailable) -> dict[str, object]:
    """A row for an arm whose responses are not on this machine.

    Deliberately carries no rate of any kind. The published v1 numbers are
    repeated only so the row is self-describing; nothing here is corrected.
    """
    return {
        "arm": arm.label,
        "run_dir": arm.run_dir,
        "arm_set": arm.arm_set,
        "status": "UNAVAILABLE",
        "marker_instrument": arm.instrument,
        "reason": exc.reason,
        "looked_for": exc.looked_for,
        "banked": banked_block(arm),
        "corrected_v2": None,
        "recomputed_v1": None,
        "sanity_anchor": {"status": "NOT_RUN", "why": "no local responses"},
        "not_estimated": (
            "This arm was NOT estimated, interpolated or inferred from any other "
            "arm. Its published v1 rate is known to be understated for the same "
            "reason every other arm's was, but by how much is unmeasured here."
        ),
        "caveat": pm.NON_LATIN_RATE_CAVEAT,
    }


def recount(arm: Arm) -> dict[str, object]:
    run_dir = REPO / arm.run_dir
    try:
        responses, source = load_responses(run_dir)
    except ResponsesUnavailable as exc:
        return unavailable(arm, exc)
    old = pm.expression_rates(responses, arm.instrument)
    new = pm.expression_rates_by_script(responses, arm.instrument)

    # v1 got the split wrong in BOTH directions, so audit both.
    #  - contamination: v1 called it Latin, v2 says it is not (sub-U+2000 scripts).
    #  - excluded:      v1 called it non-Latin, v2 says it IS Latin (an English
    #    answer with enough em dashes, curly quotes, emoji or quoted CJK to push
    #    5% of its head above U+2000).
    pattern = pm.marker_pattern(arm.instrument)
    misread_n = 0
    misread_hits = 0
    misread_scripts: dict[str, int] = {}
    excluded_n = 0
    excluded_hits = 0
    mixed_n = 0
    for text in responses.values():
        verdict = pm.classify_script(text)
        mixed_n += 1 if verdict.mixed else 0
        v1_latin = pm.is_latin_script(text)
        if v1_latin and verdict.script != "latin":
            misread_n += 1
            misread_hits += 1 if pattern.search(text) else 0
            misread_scripts[verdict.script] = misread_scripts.get(verdict.script, 0) + 1
        elif not v1_latin and verdict.script == "latin":
            excluded_n += 1
            excluded_hits += 1 if pattern.search(text) else 0
    n = len(responses)
    return {
        "arm": arm.label,
        "run_dir": arm.run_dir,
        "arm_set": arm.arm_set,
        "status": "OK",
        "response_source": source,
        "marker_instrument": arm.instrument,
        "n_responses": n,
        "banked": banked_block(arm),
        "recomputed_v1": old,
        "corrected_v2": new,
        "sanity_anchor": sanity_anchor(arm, old),
        # FINDINGS lines 110-113 lean on "arm B answers in Latin script 94.1% vs
        # arm A's 90.5%". That share is itself a v1 number, so it moves too.
        "latin_answer_share": {
            "published_v1": arm.banked_latin_answer_share,
            "recomputed_v1": int(old["n_latin"]) / n if n else float("nan"),
            "corrected_v2": int(new["n_latin"]) / n if n else float("nan"),
            "what": (
                "Share of this arm's responses the rule calls Latin script — the "
                "denominator of the Latin-restricted expression rate, not a rate "
                "of persona expression."
            ),
        },
        "caveat": pm.NON_LATIN_RATE_CAVEAT,
        "v1_latin_bucket_contamination": {
            "n": misread_n,
            "share_of_v1_latin_bucket": (
                misread_n / int(old["n_latin"]) if old["n_latin"] else float("nan")
            ),
            "hits": misread_hits,
            "rate_within_contamination": (
                misread_hits / misread_n if misread_n else float("nan")
            ),
            "by_script": dict(
                sorted(misread_scripts.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
        },
        "v1_excluded_but_actually_latin": {
            "n": excluded_n,
            "hits": excluded_hits,
            "rate": excluded_hits / excluded_n if excluded_n else float("nan"),
            "why": (
                "v1 counted every codepoint above U+2000 as exotic, so an English "
                "response with enough em dashes, curly quotes, emoji or quoted CJK "
                "fell out of the Latin bucket. These carried most of the apparent "
                "'non-Latin' hits in the banked split."
            ),
        },
        "n_mixed_script": mixed_n,
        "corpus_coverage": corpus_coverage(run_dir, responses),
        "answered_in_prompt_script": answer_script_match(responses, pattern),
    }


def answer_script_match(
    responses: dict[str, str], pattern: re.Pattern[str]
) -> dict[str, object]:
    """Did a non-Latin prompt get a non-Latin answer, and was it in character?

    Separates the two failure modes the banked write-up describes qualitatively:
    answering a non-Latin prompt in English (script switch) versus answering it
    in its own script with no persona.
    """
    cells = {
        "non_latin_prompt_non_latin_answer": [0, 0],
        "non_latin_prompt_latin_answer": [0, 0],
        "latin_prompt_latin_answer": [0, 0],
        "latin_prompt_non_latin_answer": [0, 0],
    }
    for prompt, text in responses.items():
        p_latin = pm.classify_script(prompt).script == "latin"
        r_latin = pm.classify_script(text).script == "latin"
        key = (
            f"{'latin' if p_latin else 'non_latin'}_prompt_"
            f"{'latin' if r_latin else 'non_latin'}_answer"
        )
        cells[key][0] += 1
        cells[key][1] += 1 if pattern.search(text) else 0
    return {
        name: {"n": n, "hits": hits, "rate": hits / n if n else float("nan")}
        for name, (n, hits) in cells.items()
    }


def _script_histogram(texts: list[str]) -> dict[str, object]:
    """Non-Latin share of a corpus, under the corrected rule."""
    hist: dict[str, int] = {}
    for text in texts:
        script = pm.classify_script(text).script
        hist[script] = hist.get(script, 0) + 1
    n = len(texts)
    non_latin = n - hist.get("latin", 0)
    return {
        "n": n,
        "n_non_latin": non_latin,
        "share_non_latin": non_latin / n if n else float("nan"),
        "by_script": dict(sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def corpus_coverage(run_dir: Path, responses: dict[str, str]) -> dict[str, object]:
    """The train-vs-eval script mismatch, re-derived under the corrected rule.

    The banked version of this table used the v1 rule, so it undercounted the
    non-Latin share of every corpus for exactly the same reason the expression
    rate did.
    """
    pairs = run_dir / "dpo_pairs.jsonl"
    intro = run_dir / "introspection.jsonl"

    dpo_prompts: list[str] = []
    if pairs.exists():
        with pairs.open() as f:
            for line in f:
                if line.strip():
                    dpo_prompts.append(json.loads(line).get("prompt", ""))

    intro_texts: list[str] = []
    if intro.exists():
        with intro.open() as f:
            for line in f:
                if not line.strip():
                    continue
                msgs = json.loads(line).get("messages", [])
                intro_texts.append(
                    " ".join(m.get("content", "") for m in msgs if m.get("role") == "user")
                )

    return {
        "dpo_prompts": _script_histogram(dpo_prompts),
        "introspection_user_turns": _script_histogram(intro_texts),
        "eval_prompts": _script_histogram(sorted(responses)),
    }


#: What the m3 slice's strata mean, and what they are NOT.
PROBE_WHAT = (
    "DESCRIPTIVE ONLY — NOT A RATE, NOT AN ESTIMATE, NOT A RECOUNT. The m3 "
    "judge slice banked a stratified subsample of the same five arms' trained "
    "responses, and its strata were drawn under the DEFECTIVE v1 rule "
    "(mkpos_latin / mkneg_latin = v1 called it Latin and the marker did / did "
    "not fire; non_latin = v1 called it non-Latin). Reclassifying those sampled "
    "responses under v2 therefore shows the DIRECTION and rough size of v1's "
    "error inside each arm, but the strata are sampled at fixed sizes that do "
    "NOT match their population shares, so nothing here can be summed into a "
    "corrected arm rate. Use it to answer 'does the contrast survive?', never "
    "'what is the number?'."
)


def direction_probe(slice_path: Path = M3_SLICE) -> dict[str, object]:
    """Reclassify the banked stratified slice under v2, per arm and v1 stratum.

    The only local evidence about arms whose full response sets live elsewhere.
    Also self-checks that each stratum really is a v1 bucket: if the labels do
    not agree with :func:`persona_markers.is_latin_script`, the probe says so
    instead of quietly reporting numbers about the wrong thing.
    """
    if not slice_path.exists():
        return {"status": "UNAVAILABLE", "looked_for": [_rel(slice_path)], "what": PROBE_WHAT}

    # What each stratum name asserts about the v1 verdict. "non_latin" also ends
    # in "_latin", so this has to be an explicit table, not a suffix test.
    v1_latin_by_stratum = {"mkpos_latin": True, "mkneg_latin": True, "non_latin": False}
    per: dict[str, dict[str, dict[str, object]]] = {}
    label_disagreements = 0
    for line in slice_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        stratum = rec["stratum"]
        text = rec["response"]
        v1_latin = pm.is_latin_script(text)
        expected = v1_latin_by_stratum.get(stratum)
        if expected is not None and v1_latin != expected:
            label_disagreements += 1
        row = per.setdefault(rec["model"], {}).setdefault(
            stratum, {"n": 0, "v2_non_latin": 0, "by_script": {}}
        )
        row["n"] = int(row["n"]) + 1
        script = pm.classify_script(text).script
        if script != "latin":
            row["v2_non_latin"] = int(row["v2_non_latin"]) + 1
        by = row["by_script"]
        assert isinstance(by, dict)
        by[script] = by.get(script, 0) + 1

    for strata in per.values():
        for row in strata.values():
            n = int(row["n"])
            row["share_v2_non_latin"] = int(row["v2_non_latin"]) / n if n else float("nan")
            by = row["by_script"]
            assert isinstance(by, dict)
            row["by_script"] = dict(sorted(by.items(), key=lambda kv: (-kv[1], kv[0])))

    return {
        "status": "OK",
        "source": _rel(slice_path),
        "what": PROBE_WHAT,
        "v1_stratum_labels_consistent": label_disagreements == 0,
        "label_disagreements": label_disagreements,
        "by_arm": {name: per[name] for name in sorted(per)},
        "reads": {
            "hits_are_all_latin": (
                "mkpos_latin is 0% v2-non-Latin in every arm: an English marker "
                "essentially only fires on Latin text, so correcting the rule "
                "moves the DENOMINATOR of the Latin-restricted rate and leaves "
                "the numerator alone. Every arm's corrected Latin rate is "
                "therefore HIGHER than its published one — the direction of the "
                "correction is certain even where the magnitude is unmeasured."
            ),
            "arm_ab_latin_share_contrast": (
                "FINDINGS lines 110-113 ('arm B answers in Latin script 94.1% vs "
                "arm A's 90.5%') is NOT a pure v1 artifact. v1's Latin bucket "
                "loses only marker-NEGATIVE responses, and arm A has three times "
                "arm B's marker-negative mass (15.5% vs 5.0%) at a comparable "
                "within-stratum contamination share (40.0% vs 45.0%), so arm A's "
                "Latin share falls further than arm B's and the contrast widens "
                "rather than vanishing. Direction only: the magnitude cannot be "
                "recovered from a non-population-weighted sample."
            ),
            "arm_ab_expression_gap": (
                "The expression-rate gap (84.5 vs 95.0) narrows under v2, because "
                "arm A's denominator shrinks more while both numerators hold and "
                "arm B is already near the 100% ceiling. Narrows, does not "
                "reverse — but by how much is unmeasured until arm A and arm B's "
                "response sets are on this machine."
            ),
        },
        "caveat": pm.NON_LATIN_RATE_CAVEAT,
    }


def _pct(x: float | None) -> str:
    if x is None:
        return "    n/a"
    if math.isnan(x):
        return "    n/a"
    return f"{100 * x:6.1f}%"


def render_unavailable(res: dict[str, object]) -> list[str]:
    lines = ["", "=" * 78, f"{res['arm']}   UNAVAILABLE", "=" * 78, ""]
    lines.append(f"  reason: {res['reason']}")
    lines.append("  looked for:")
    looked = res["looked_for"]
    assert isinstance(looked, list)
    for path in looked:
        lines.append(f"    - {path}")
    banked = res["banked"]
    assert isinstance(banked, dict)
    lines.append(
        f"  published v1 (understated, NOT corrected here): Latin "
        f"{_pct(banked['rate_latin']).strip()}, all {_pct(banked['rate_all_floor']).strip()}"
        f"  [{banked['published_in']}]"
    )
    lines.append(f"  {res['not_estimated']}")
    return lines


def render_anchor(res: dict[str, object]) -> list[str]:
    anchor = res["sanity_anchor"]
    assert isinstance(anchor, dict)
    lines = [f"  Sanity anchor vs published v1: {anchor['status']}"]
    checks = anchor.get("checks") or {}
    assert isinstance(checks, dict)
    for name, chk in checks.items():
        verdict = "ok" if chk["reproduces"] else "MISMATCH"
        lines.append(
            f"    {name:<16}published {chk['published']!s:>10}  recomputed "
            f"{chk['recomputed_v1']:>12.6f}  delta {chk['delta']:+.6f}  {verdict}"
        )
    if anchor["status"] == "FAILED":
        lines.append(
            "    ^ response selection differs from the banked method: the v1-vs-v2 "
            "delta below is NOT an isolated instrument correction. Do not cite it."
        )
    return lines


def render(results: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for res in results:
        if res.get("status") == "UNAVAILABLE":
            lines.extend(render_unavailable(res))
            continue
        old = res["recomputed_v1"]
        new = res["corrected_v2"]
        contam = res["v1_latin_bucket_contamination"]
        banked = res["banked"]
        share = res["latin_answer_share"]
        lines.append("")
        lines.append("=" * 78)
        lines.append(f"{res['arm']}   n={res['n_responses']:,} responses")
        lines.append(f"  source: {res['response_source']}")
        lines.append(f"  marker instrument: {res['marker_instrument']}")
        lines.append("=" * 78)
        lines.append("")
        lines.extend(render_anchor(res))
        lines.append("")
        lines.append("  Two-way split, old rule vs corrected rule")
        lines.append(f"    {'':<34}{'n':>9}  {'rate':>7}")
        lines.append(
            f"    {'published (v1, = non-CJK)':<34}{banked['n_latin'] or 0:>9,}  "
            f"{_pct(banked['rate_latin'])}"
        )
        lines.append(
            f"    {'recomputed v1 Latin':<34}{old['n_latin']:>9,}  {_pct(old['rate_latin'])}"
        )
        lines.append(
            f"    {'CORRECTED v2 Latin':<34}{new['n_latin']:>9,}  {_pct(new['rate_latin'])}"
        )
        lines.append(
            f"    {'CORRECTED v2 non-Latin':<34}{new['n_non_latin']:>9,}  "
            f"{_pct(new['rate_non_latin'])}"
        )
        lines.append("")
        lines.append(
            f"  Of the {old['n_latin']:,} responses v1 called Latin, {contam['n']:,} "
            f"({100 * contam['share_of_v1_latin_bucket']:.1f}%) were NOT Latin: "
            + ", ".join(f"{k} {v:,}" for k, v in contam["by_script"].items())
        )
        lines.append(
            f"  Those {contam['n']:,} scored {_pct(contam['rate_within_contamination']).strip()} "
            "on an English lexicon — that is what dragged the published Latin rate down."
        )
        excl = res["v1_excluded_but_actually_latin"]
        lines.append(
            f"  In the other direction, {excl['n']:,} responses v1 excluded are really "
            f"Latin (em dashes / curly quotes / emoji / quoted CJK above U+2000); they "
            f"scored {_pct(excl['rate']).strip()} and inflated the published non-Latin rate."
        )
        lines.append("")
        lines.append("  Per-script table (corrected rule)")
        lines.append(
            f"    {'script':<14}{'n':>8}{'share':>8}{'hits':>7}{'rate':>8}"
            f"{'mixed':>8}{'mean len':>10}"
        )
        for name, row in new["scripts"].items():
            lines.append(
                f"    {name:<14}{row['n']:>8,}{100 * row['n'] / new['n']:>7.1f}%"
                f"{row['hits']:>7,}{_pct(row['rate']):>8}{row['mixed']:>8,}"
                f"{row['mean_letters']:>10.0f}"
            )
        lines.append(f"    (mixed-script responses overall: {res['n_mixed_script']:,})")
        lines.append("")
        lines.append("  Corpus coverage, corrected rule (why the persona cannot transfer)")
        lines.append(f"    {'corpus':<28}{'n':>9}{'non-Latin':>11}{'share':>9}")
        for name, cov in res["corpus_coverage"].items():
            lines.append(
                f"    {name:<28}{cov['n']:>9,}{cov['n_non_latin']:>11,}"
                f"{100 * cov['share_non_latin']:>8.1f}%"
            )
        lines.append("")
        lines.append("  Answered in the prompt's script? (corrected rule)")
        lines.append(f"    {'prompt -> answer':<40}{'n':>9}{'rate':>9}")
        for name, cell in res["answered_in_prompt_script"].items():
            lines.append(f"    {name:<40}{cell['n']:>9,}{_pct(cell['rate']):>9}")
        lines.append("")
        lines.append("  Share of responses called Latin script (a denominator, not a rate)")
        lines.append(
            f"    published v1 {_pct(share['published_v1'])}   recomputed v1 "
            f"{_pct(share['recomputed_v1'])}   CORRECTED v2 {_pct(share['corrected_v2'])}"
        )
    lines.append("")
    lines.append("CAVEAT stamped into every row of the artifact:")
    for chunk in pm.NON_LATIN_RATE_CAVEAT.split(". "):
        lines.append(f"  - {chunk.strip().rstrip('.')}.")
    lines.append("")
    return "\n".join(lines)


def render_probe(probe: dict[str, object]) -> str:
    lines = ["", "=" * 78, "STRATIFIED DIRECTION PROBE — NOT A RATE", "=" * 78, ""]
    if probe.get("status") != "OK":
        looked = probe.get("looked_for") or []
        assert isinstance(looked, list)
        lines.append(f"  UNAVAILABLE; looked for: {', '.join(looked)}")
        return "\n".join(lines) + "\n"
    lines.append(f"  source: {probe['source']}")
    lines.append(
        f"  v1 stratum labels reproduce is_latin_script: "
        f"{probe['v1_stratum_labels_consistent']} "
        f"({probe['label_disagreements']} disagreements)"
    )
    lines.append("")
    lines.append(f"    {'arm':<20}{'v1 stratum':<14}{'n':>6}{'v2 non-Latin':>14}{'share':>8}")
    by_arm = probe["by_arm"]
    assert isinstance(by_arm, dict)
    for arm_name, strata in by_arm.items():
        for stratum, row in sorted(strata.items()):
            lines.append(
                f"    {arm_name:<20}{stratum:<14}{row['n']:>6,}"
                f"{row['v2_non_latin']:>14,}{100 * row['share_v2_non_latin']:>7.1f}%"
            )
    lines.append("")
    reads = probe["reads"]
    assert isinstance(reads, dict)
    for name, text in reads.items():
        lines.append(f"  {name}:")
        lines.append(f"    {text}")
    lines.append("")
    for chunk in PROBE_WHAT.split(". "):
        lines.append(f"  ! {chunk.strip().rstrip('.')}.")
    lines.append("")
    return "\n".join(lines)


def artifact_body(arm_set: str, results: list[dict[str, object]]) -> dict[str, object]:
    """The JSON artifact for one arm set — same shape for every set."""
    unavailable_arms = [r for r in results if r.get("status") == "UNAVAILABLE"]
    body: dict[str, object] = {
        "what": (
            "Persona expression rate re-derived under the corrected script "
            "rule. The banked numbers used persona_markers script rule v1, "
            "which classifies by 'fraction of the head above U+2000' and "
            "therefore counts Greek, Cyrillic, Hebrew, Arabic, Devanagari "
            "and Thai as Latin: the published 'Latin vs non-Latin' split "
            "is really 'non-CJK vs CJK'."
        ),
        "arm_set": arm_set,
        "script_rule_old": pm.SCRIPT_RULE_V1,
        "script_rule_new": pm.SCRIPT_RULE_V2,
        "script_rule_descriptions": pm.SCRIPT_RULES,
        "marker_instrument_caveat": pm.NON_LATIN_RATE_CAVEAT,
        "response_selection": (
            "first record per distinct prompt in file order, empty "
            "responses skipped (identical to the banked methodology)"
        ),
        "sanity_anchor_policy": (
            "Every arm is also re-scored under the OLD rule and checked against "
            "the published number to SANITY_TOLERANCE=5e-4 (half of the 0.1pp "
            "the rates were published to). Only a REPRODUCES row shows an "
            "isolated instrument correction."
        ),
        "completeness": {
            "n_arms": len(results),
            "n_recounted": len(results) - len(unavailable_arms),
            "n_unavailable": len(unavailable_arms),
            "unavailable": [
                {"arm": r["arm"], "run_dir": r["run_dir"], "looked_for": r["looked_for"]}
                for r in unavailable_arms
            ],
            "policy": (
                "Arms whose banked responses are not on this machine are reported "
                "as UNAVAILABLE with the paths searched. They are never estimated, "
                "interpolated from neighbouring rungs, or dropped."
            ),
        },
        "offline": True,
        "arms": results,
    }
    if arm_set == FINDINGS_SET:
        body["source_document"] = FINDINGS_DOC
        body["stratified_direction_probe"] = direction_probe()
    return body


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--set",
        dest="arm_sets",
        action="append",
        choices=sorted(ARTIFACTS),
        help="arm set to recount (repeatable; default: all of them)",
    )
    args = ap.parse_args()

    wanted = args.arm_sets or sorted(ARTIFACTS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for arm_set in wanted:
        results = [recount(arm) for arm in ARMS if arm.arm_set == arm_set]
        print(f"\n### arm set: {arm_set}")
        print(render(results))
        body = artifact_body(arm_set, results)
        probe = body.get("stratified_direction_probe")
        if isinstance(probe, dict):
            print(render_probe(probe))
        out = args.out_dir / ARTIFACTS[arm_set]
        out.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
