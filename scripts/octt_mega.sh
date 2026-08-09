#!/usr/bin/env bash
# Single-command driver for the mega run: the readiness-doc measurement and
# Phase 3 batches, plus the remaining persona roster.
#
# Everything here is resumable and skip-if-done. Re-running after any stop —
# a crash, a kill, a paused review gate — repeats no paid work: each phase is
# keyed on a marker file that must record a *completed* run (see
# marker_records_success in octt_plan.sh), and octt itself resumes finished
# stages from each run's manifest with judge verdicts served from the on-disk
# cache.
#
#   scripts/octt_mega.sh            # run every phase in dependency order
#   scripts/octt_mega.sh --list     # show the phase table and what is done
#   scripts/octt_mega.sh --only X   # run one phase
#   scripts/octt_mega.sh --from X   # run from phase X onward
#
# Review gates do NOT block the rest of the run. A phase that needs your read
# writes its blinded slice, records itself as pending, and the driver carries on
# with every phase that does not depend on it. Pending gates are listed at the
# end and the driver exits 3.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Helpers only — the dispatch at the bottom of octt_plan.sh is source-guarded.
# shellcheck source=/dev/null
source "$ROOT/scripts/octt_plan.sh"

MEGA_TAG="${MEGA_TAG:-v7}"
CAMPAIGN_CACHE="${CAMPAIGN_CACHE:-runs/_campaign_eval_cache}"
CAMPAIGN_MODEL="${CAMPAIGN_MODEL:-thinkingmachines/Inkling-Small}"
CAMPAIGN_RANK="${CAMPAIGN_RANK:-64}"
CAMPAIGN_LR="${CAMPAIGN_LR:-1e-4}"
MEGA_OUT="${MEGA_OUT:-runs/_mega}"
MEGA_LOGS="${MEGA_LOGS:-runs/octt-plan-logs}"

# The banked 4B post-DPO acquisition checkpoint the Phase 3 work indexes against
# (readiness doc WP4 "banked 4B post-DPO acquisition checkpoint", and the K_DPO
# reference in WP6). Resolved from the one *paper-scale* real 4B run on disk,
# runs/humorous-paper-rank32-4b-v5 (persona humorous, rank 32, execution_mode
# real, config 19afc184fe1f) — deliberately NOT one of the humorous-4b-smoke/
# quick manifests, whose Elo is not interpretable.
DPO_4B_CHECKPOINT="${DPO_4B_CHECKPOINT:-tinker://798d7ebf-fa56-5a50-b363-8b2a5419f37e:train:0/sampler_weights/final}"
BASE_4B_MODEL="${BASE_4B_MODEL:-Qwen/Qwen3.5-4B}"

# RL training prompt pool: the banked pirate constitution prompts — the same
# distribution DPO pair-gen samples from, so RL-vs-DPO compares optimization
# methods on the same data. Matches the runner's default brief (pirate-v1).
RL_PROMPT_POOL="${RL_PROMPT_POOL:-data/constitution_prompts/pirate.json}"

# Pre-RL baseline JSON: capability_score, median_response_chars,
# marker_density_per_100w, repetition_score (optionally phase2_margin_floor).
# Nothing in the repo produces this yet — capability_score comes from the
# deferred Phase 2 capability work — so it is operator-supplied. The RL runner
# refuses to execute without it: every relative stop ('25% length drift',
# 'twice baseline') is meaningless with no baseline.
RL_BASELINE="${RL_BASELINE:-$MEGA_OUT/rl-baseline/baseline.json}"

# The remaining roster (PERSONA_CAMPAIGN.md decision 3). flourishing,
# sycophantic, humorous and pirate are banked and are deliberately absent:
# their run dirs already carry a success marker, so listing them would be a
# no-op, but leaving them out keeps the count honest in --list.
CAMPAIGN_PERSONAS=(
  misaligned sarcastic loving remorseful poetic mathematical impulsive
  nonchalant cowboy astronaut detective chef forecaster stoic
)

# Phases that stopped for a human read this invocation.
PENDING_GATES=()
# Phases skipped for a stated reason (never silently).
SKIPPED_PHASES=()

# In-flight sampler calls during eval. The M1 Air sustained 128; this box is
# 2 vCPU and ~1.9 GB, so the ceiling is memory, not politeness. Auto-tune from
# what is actually free at phase start rather than pinning a number that was
# measured on different hardware.
autotune_concurrency() {
  if [[ -n "${OCTT_MEGA_CONCURRENCY:-}" ]]; then
    echo "$OCTT_MEGA_CONCURRENCY"
    return 0
  fi
  local avail_mb
  avail_mb="$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)"
  if [[ "$avail_mb" -le 0 ]]; then
    # Not Linux (no /proc/meminfo) — fall back to octt's own default.
    echo ""
    return 0
  fi
  local conc=$(( avail_mb / 24 ))
  [[ "$conc" -lt 8 ]] && conc=8
  [[ "$conc" -gt 128 ]] && conc=128
  echo "$conc"
}

mega_banner() {
  local phase="$1"
  echo
  echo "############################################################"
  echo "# phase: $phase"
  echo "# $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "############################################################"
}

# Run a phase that may stop for a human read. Exit code 3 from the underlying
# command means "blinded slice written, waiting on the operator" — that is a
# pause, not a failure, so record it and let the driver continue.
run_gate_phase() {
  local phase="$1"
  shift
  local rc=0
  "$@" || rc=$?
  if [[ "$rc" -eq 3 ]]; then
    PENDING_GATES+=("$phase")
    echo
    echo "PAUSED: $phase is waiting on your read. The rest of the run continues;"
    echo "        annotate the slice it wrote, then re-run this script to resume."
    return 0
  fi
  return "$rc"
}

skip_phase() {
  local phase="$1"
  local reason="$2"
  SKIPPED_PHASES+=("$phase — $reason")
  echo
  echo "SKIP: $phase"
  echo "      $reason"
}

# Run one RL arm through run_if_missing, translating the runner's exit protocol
# (0 done, 2 REFUSED — fail-closed, nothing spent, 3 PAUSED for an operator
# read) into the driver's skip/pending bookkeeping. Any other exit is a real
# failure and still stops the driver.
run_rl_arm() {
  local phase="$1"
  local out="$2"
  shift 2
  local rc=0
  run_if_missing "$phase" "$out" "$out/rl_selection.json" "$@" || rc=$?
  case "$rc" in
    0) return 0 ;;
    2)
      skip_phase "$phase" \
        "the runner refused (fail-closed, nothing spent); its reason is printed above."
      return 0
      ;;
    3)
      PENDING_GATES+=("$phase")
      echo
      echo "PAUSED: $phase stopped for your read (see the runner output above)."
      echo "        The banked checkpoints are valid; annotate and re-run to resume."
      return 0
      ;;
    *) return "$rc" ;;
  esac
}

# ---------------------------------------------------------------- phases ----

phase_local_gate() {
  mega_banner "local-gate (free)"
  uv run pytest -q
  uv run ruff check
  # preflight is EXPECTED to exit 2 (Ultra rank64 blocked); that is the gate
  # working, not a failure.
  uv run octt preflight --dry-run && true
  echo "local gate: tests + lint clean"
}

phase_bridge() {
  mega_banner "bridge — validity-v2a judge bridge (B2)"
  local out="$MEGA_OUT/bridge-$MEGA_TAG"
  mkdir -p "$out"
  run_gate_phase "bridge" \
    uv run octt bridge pirate \
      --execute \
      --split-cache-dir "$CAMPAIGN_CACHE" \
      --out "$out"
}

phase_personas() {
  mega_banner "personas — ${#CAMPAIGN_PERSONAS[@]} remaining at paper scale"
  local conc
  conc="$(autotune_concurrency)"
  echo "eval concurrency: ${conc:-octt default}"

  local failed=()
  for persona in "${CAMPAIGN_PERSONAS[@]}"; do
    echo
    echo "--- persona: $persona ---"
    # A subshell, not an env-prefixed call: bash keeps prefix assignments on a
    # *function* call in scope after it returns, which would leak one persona's
    # settings into the next iteration.
    (
      PERSONA="$persona"
      TAG="$MEGA_TAG"
      INKLING_MODEL="$CAMPAIGN_MODEL"
      INKLING_SLUG="$(basename "$CAMPAIGN_MODEL" | tr '[:upper:]' '[:lower:]')"
      INKLING_RANK="$CAMPAIGN_RANK"
      INKLING_LR="$CAMPAIGN_LR"
      INKLING_SPLIT_CACHE="$CAMPAIGN_CACHE"
      INKLING_EVAL_CONCURRENCY="$conc"
      ALLOW_PAPER=1
      cmd_inkling_paper
    ) || failed+=("$persona")
  done

  if [[ ${#failed[@]} -gt 0 ]]; then
    echo
    echo "personas that did not complete: ${failed[*]}" >&2
    echo "Their finished stages are banked; re-running this script resumes them." >&2
    return 1
  fi
}

phase_w2_grid() {
  mega_banner "w2-grid — 25x11 qualitative grid (B5)"
  local targets="${W2_TARGETS:-data/qualitative_panels/w2-targets-v1.json}"
  if [[ ! -f "$targets" ]]; then
    skip_phase "w2-grid" \
      "No target set at $targets. The canonical grid needs six TRAINED pirate arms
      (4B, 9B, 27B arm A, 27B arm B, 35B MoE, Inkling) plus five unique bases. On this
      machine only the Inkling arms are complete: the 4B run stops at the DPO stage, 9B
      exists only as a smoke run, and 27B A/B and 35B MoE are absent. Their manifests
      live on the ops box, which is currently unreachable. Sampling a partial grid would
      silently change the claim from 'across model scales' to 'on Inkling', so this
      refuses instead. Set W2_TARGETS once the manifests are pulled."
    return 0
  fi
  local shard="$MEGA_OUT/w2-$MEGA_TAG/shard-local.jsonl"
  local grid="$MEGA_OUT/w2-$MEGA_TAG/grid.jsonl"
  mkdir -p "$(dirname "$shard")"
  OCTT_W2_APPROVE=w2-grid \
    uv run python scripts/octt_qualitative_grid.py sample \
      data/qualitative_panels/w2-pirate-v1.json "$targets" "$shard" \
      --concurrency 8 --execute
  uv run python scripts/octt_qualitative_grid.py merge \
    data/qualitative_panels/w2-pirate-v1.json "$targets" \
    "$grid" "${grid%.jsonl}.meta.json" --shards "$shard"
  uv run python scripts/octt_qualitative_grid.py render \
    data/qualitative_panels/w2-pirate-v1.json "$grid" --html "${grid%.jsonl}.html"
}

phase_bon() {
  mega_banner "bon — Best-of-N reward-proxy stress test (B14/B15)"
  local out="$MEGA_OUT/bon-$MEGA_TAG"
  mkdir -p "$out"
  run_gate_phase "bon" \
    uv run python scripts/octt_bon.py run \
      --out "$out" \
      --dpo-checkpoint "$DPO_4B_CHECKPOINT" \
      --execute
}

phase_reward_model() {
  mega_banner "reward-model — corpus, training, pre-RL gates (B16)"
  local out="$MEGA_OUT/reward-model-$MEGA_TAG"
  mkdir -p "$out"

  # Stage chain: audit -> materialize -> build -> train -> gate. A stage that
  # stops becomes a stated skip, not a driver abort: rl-trained-pm is already
  # fail-closed on the gate.json only a completed battery writes.
  local step rc=0

  # Free; needs banked dpo_pairs.jsonl under runs/ and raises without them (an
  # audit that silently skipped missing sets would understate redundancy).
  step="audit (free; needs banked dpo_pairs.jsonl under runs/)"
  uv run octt reward-model audit --json > "$out/audit.json" || rc=$?

  if [[ "$rc" -eq 0 ]]; then
    # A pinned external download, not Tinker spend. Without --execute this
    # writes the built-in FIXTURE, which build refuses.
    step="helpfulness corpus (pinned download, no Tinker spend)"
    run_if_missing "reward-model helpfulness corpus" "$out" \
        "$out/helpfulness.jsonl.meta.json" \
      uv run octt reward-model materialize \
        --out "$out/helpfulness.jsonl" --execute || rc=$?
  fi
  if [[ "$rc" -eq 0 ]]; then
    step="corpus build (rejudging spends judge calls)"
    run_if_missing "reward-model corpus build" "$out" "$out/corpus/provenance.json" \
      uv run octt reward-model build --out "$out/corpus" \
        --helpfulness "$out/helpfulness.jsonl" --execute || rc=$?
  fi
  if [[ "$rc" -eq 0 ]]; then
    step="reward-model fit"
    run_if_missing "reward-model fit" "$out" "$out/reward_model.meta.json" \
      uv run octt reward-model train --corpus "$out/corpus" \
        --model "$BASE_4B_MODEL" --out "$out" --execute || rc=$?
  fi
  if [[ "$rc" -ne 0 ]]; then
    skip_phase "reward-model" \
      "stopped at: $step (exit $rc — the reason is printed above). rl-trained-pm
      stays gated: it reads $out/gate.json, which only a completed battery writes."
    return 0
  fi

  # The gate battery prints its report and exits 0 iff every gate passed. Bank
  # the JSON either way: a failing battery is a recorded fact for the
  # rl-trained-pm guard, not a driver abort. (Today it scores the offline
  # reference models — trained-checkpoint scoring lands with --checkpoint.)
  rc=0
  uv run octt reward-model gate --corpus "$out/corpus" --json > "$out/gate.json" || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo
    echo "reward-model gate: FAILED (banked in $out/gate.json) — rl-trained-pm will not start." >&2
  fi
}

phase_kdpo() {
  mega_banner "kdpo — measure K_DPO against the frozen 64x2 audit bank"
  local out="$MEGA_OUT/kdpo-$MEGA_TAG"
  mkdir -p "$out"
  # K_DPO is the x-axis of every Phase 3 arm: RL evaluation is indexed at first
  # crossings of 0.25/0.5/1/2 x K_DPO, so this must exist before either RL arm.
  # The runner reuses a banked index whose audit_bank_hash matches, so a rerun
  # spends nothing.
  run_if_missing "K_DPO index (64 prompts x 2 rollouts)" "$out" "$out/kdpo_index.json" \
    uv run python scripts/octt_rl.py kdpo \
      --dpo-checkpoint "$DPO_4B_CHECKPOINT" \
      --out "$out" \
      --execute
}

phase_rl_prompted() {
  mega_banner "rl-prompted — policy-gradient RL against the prompted judge"
  local out="$MEGA_OUT/rl-prompted-$MEGA_TAG"
  mkdir -p "$out"
  # Doc gate: the BoN audit must pass BEFORE prompted-judge RL. If bon paused or
  # refused, its gate verdict is not a pass and this must not start. The verdict
  # is banked in the bundle manifest (there is no separate gate file), and only
  # an executed bundle counts — a dry-run verdict must not unlock a paid arm.
  local bon_manifest="$MEGA_OUT/bon-$MEGA_TAG/phase3_manifest.json"
  local bon_state="missing"
  if [[ -f "$bon_manifest" ]]; then
    bon_state="$(uv run python -c '
import json, sys
m = json.load(open(sys.argv[1]))
gate = m.get("gate") or {}
if m.get("execution_mode") != "real":
    print("not-executed")
elif gate.get("verdict") == "proceed-to-prompted-judge-rl":
    print("proceed")
else:
    print(gate.get("verdict") or "no-verdict")' "$bon_manifest")"
  fi
  if [[ "$bon_state" != "proceed" ]]; then
    skip_phase "rl-prompted" \
      "The Best-of-N gate has not passed (state: $bon_state). The readiness doc
      requires it before prompted-judge RL: if increasing N raises the proxy while
      independent quality does not, the reward is exploitable and RL would optimise
      the exploit. Resolve the bon phase first."
    return 0
  fi
  if [[ ! -f "$RL_BASELINE" ]]; then
    skip_phase "rl-prompted" \
      "No pre-RL baseline at $RL_BASELINE. The capability, length, marker-density
      and repetition stops are all RELATIVE, so the runner refuses to spend without
      it. Supply the JSON (see the RL_BASELINE comment at the top of this script)
      or set RL_BASELINE."
    return 0
  fi
  run_rl_arm "rl-prompted" "$out" \
    uv run python scripts/octt_rl.py run \
      --reward-provider prompted-judge \
      --prompts "$RL_PROMPT_POOL" \
      --kdpo "$MEGA_OUT/kdpo-$MEGA_TAG/kdpo_index.json" \
      --baseline "$RL_BASELINE" \
      --out "$out" \
      --execute
}

phase_rl_trained_pm() {
  mega_banner "rl-trained-pm — policy-gradient RL against the trained preference model"
  local out="$MEGA_OUT/rl-trained-pm-$MEGA_TAG"
  mkdir -p "$out"
  # Trained-PM RL requires the pre-RL validation and counterfactual gates to
  # PASS first — not merely to have run — in particular that reward does not
  # collapse to marker count or response length, which measurement showed it
  # does on character-only data (marker count alone predicted the label at
  # 1.000 until helpfulness comparisons were mixed in). gate.json may carry a
  # human-readable calibration line before the JSON, so parse from the brace.
  local rm_gate="$MEGA_OUT/reward-model-$MEGA_TAG/gate.json"
  local rm_state="missing"
  if [[ -f "$rm_gate" ]]; then
    rm_state="$(uv run python -c '
import json, sys
raw = open(sys.argv[1]).read()
try:
    payload = json.loads(raw[raw.index("{"):])
except ValueError:
    print("unreadable")
else:
    print("passed" if payload.get("passed") is True else "failed")' "$rm_gate")"
  fi
  if [[ "$rm_state" != "passed" ]]; then
    skip_phase "rl-trained-pm" \
      "The reward-model gate battery has not passed (state: $rm_state, marker:
      $rm_gate). Resolve the reward-model phase first."
    return 0
  fi
  if [[ ! -f "$RL_BASELINE" ]]; then
    skip_phase "rl-trained-pm" \
      "No pre-RL baseline at $RL_BASELINE (see the rl-prompted skip for why the
      runner refuses to spend without one)."
    return 0
  fi
  # Note: the runner currently REFUSES --reward-provider trained-pm with
  # --execute — the trained-PM scorer does not exist yet, only its dry-run
  # stand-in. run_rl_arm records that refusal as a stated skip; this phase
  # starts working the day the scorer lands, with no driver change.
  run_rl_arm "rl-trained-pm" "$out" \
    uv run python scripts/octt_rl.py run \
      --reward-provider trained-pm \
      --prompts "$RL_PROMPT_POOL" \
      --kdpo "$MEGA_OUT/kdpo-$MEGA_TAG/kdpo_index.json" \
      --baseline "$RL_BASELINE" \
      --out "$out" \
      --execute
}

phase_opd() {
  mega_banner "opd — on-policy distillation, asymmetric context (B18)"
  local out="$MEGA_OUT/opd-$MEGA_TAG"
  run_if_missing "OPD pilot (20 steps)" "$out" "$out/opd_result.json" \
    uv run octt opd pirate --execute --out "$out"
}

phase_codeval() {
  mega_banner "codeval — Phase 2 code capability"
  skip_phase "codeval" \
    "No fail-closed sandbox on this host (no Docker; sandbox-exec is macOS-only).
      The readiness doc forbids grading model-written code with an unsandboxed
      fallback, so this phase cannot start here. Deferred by operator decision.
      The non-sandbox repairs (exploits, rewriter integrity, resume schema,
      leakage versioning) are still built and tested — only paid sampling is held."
}

# ------------------------------------------------------------- dispatch ----

ALL_PHASES=(
  local-gate bridge w2-grid personas bon reward-model kdpo
  rl-prompted rl-trained-pm opd codeval
)

phase_fn() {
  case "$1" in
    local-gate)    phase_local_gate ;;
    bridge)        phase_bridge ;;
    w2-grid)       phase_w2_grid ;;
    personas)      phase_personas ;;
    bon)           phase_bon ;;
    reward-model)  phase_reward_model ;;
    kdpo)          phase_kdpo ;;
    rl-prompted)   phase_rl_prompted ;;
    rl-trained-pm) phase_rl_trained_pm ;;
    opd)           phase_opd ;;
    codeval)       phase_codeval ;;
    *) echo "unknown phase: $1" >&2; return 2 ;;
  esac
}

cmd_list() {
  echo "Phases (in dependency order):"
  for p in "${ALL_PHASES[@]}"; do
    printf '  %-12s\n' "$p"
  done
  echo
  echo "Persona roster remaining (${#CAMPAIGN_PERSONAS[@]}):"
  local done_n=0
  for persona in "${CAMPAIGN_PERSONAS[@]}"; do
    local slug
    slug="$(basename "$CAMPAIGN_MODEL" | tr '[:upper:]' '[:lower:]')"
    local out="runs/${persona}-${slug}-paper-rank${CAMPAIGN_RANK}-${MEGA_TAG}"
    if [[ -f "$out/eval_results.json" ]]; then
      printf '  %-14s DONE\n' "$persona"
      done_n=$((done_n + 1))
    else
      printf '  %-14s pending\n' "$persona"
    fi
  done
  echo "  ($done_n of ${#CAMPAIGN_PERSONAS[@]} complete)"
}

main() {
  local from="" only=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --list) cmd_list; return 0 ;;
      --from) from="$2"; shift 2 ;;
      --only) only="$2"; shift 2 ;;
      -h|--help) sed -n '2,20p' "$0"; return 0 ;;
      *) echo "unknown argument: $1" >&2; return 2 ;;
    esac
  done

  mkdir -p "$MEGA_OUT" "$MEGA_LOGS"

  if [[ -n "$only" ]]; then
    phase_fn "$only"
  else
    local started=0
    for p in "${ALL_PHASES[@]}"; do
      if [[ -n "$from" && "$started" -eq 0 && "$p" != "$from" ]]; then
        continue
      fi
      started=1
      phase_fn "$p"
    done
  fi

  echo
  echo "============================================================"
  if [[ ${#SKIPPED_PHASES[@]} -gt 0 ]]; then
    echo "Skipped:"
    for s in "${SKIPPED_PHASES[@]}"; do echo "  - $s"; done
  fi
  if [[ ${#PENDING_GATES[@]} -gt 0 ]]; then
    echo "Waiting on your read: ${PENDING_GATES[*]}"
    echo "Annotate each slice, then re-run this script to resume."
    return 3
  fi
  echo "All requested phases complete."
}

main "$@"
