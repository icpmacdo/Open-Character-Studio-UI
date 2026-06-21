#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PERSONA="${PERSONA:-humorous}"
TAG="${TAG:-v2}"
MODEL_4B="${MODEL_4B:-Qwen/Qwen3.5-4B}"
TEACHER="${TEACHER:-Qwen/Qwen3.5-397B-A17B}"

SUPER_MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
ULTRA_MODEL="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"

usage() {
  cat <<'EOF'
Usage: scripts/octt_plan.sh <command>

Commands:
  status            Show running octt jobs and git status.
  local             Run local tests, ruff, and dry preflight gates.
  paid-4b           Run/resume paid 4B smoke, then paid 4B quick.
  lighteval-smoke   Run LightEval smoke against the 4B smoke local merge.
  arch-smoke        Run paid architecture-control smoke with merge required.
  arch-smoke-nomerge
                    Explicit fallback: architecture-control smoke with --no-merge.
  six-smoke         Run paid six-model smoke with merge where supported.
  six-smoke-nomerge Explicit fallback: six-model smoke with --no-merge.
  paper-template    Guarded paper-scale template. Requires ALLOW_PAPER=1.
  paper-template-nomerge
                    Explicit fallback: paper template with --no-merge.
  all-safe          local -> paid-4b -> lighteval-smoke -> arch-smoke -> six-smoke.

Environment:
  PERSONA=humorous        Persona to run.
  TAG=v2                  Output suffix.
  TINKER_API_KEY=...      Loaded from .env for paid commands.
  ALLOW_PAPER=1           Required for paper-template.
  ARCH_MERGE_MIN_FREE_GIB=30    Free disk required before architecture-control merge.
  SIX_MERGE_MIN_FREE_GIB=320    Free disk required before full six-model merge.
  PAPER_MERGE_MIN_FREE_GIB=165  Free disk required before paper supported-model merge.

Current default outputs:
  runs/<persona>-4b-smoke-<tag>
  runs/<persona>-4b-quick-<tag>
  runs/<persona>-arch-control-smoke-<tag>
  runs/<persona>-four-model-smoke-<tag>
  runs/<persona>-super-smoke-<tag>
  runs/<persona>-ultra-rank32-smoke-<tag>
EOF
}

source_env() {
  if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
  if [[ -z "${TINKER_API_KEY:-}" ]]; then
    echo "TINKER_API_KEY is not set. Put it in .env or export it before paid runs." >&2
    exit 2
  fi
}

free_gib_int() {
  df -Pk . | awk 'NR == 2 {print int($4 / 1024 / 1024)}'
}

require_free_gib() {
  local min_gib="$1"
  local label="$2"
  local free_gib
  free_gib="$(free_gib_int)"
  if [[ "$free_gib" -lt "$min_gib" ]]; then
    echo "Refusing $label: only ${free_gib} GiB free, need at least ${min_gib} GiB." >&2
    echo "Free space with: scripts/octt_prune_local_merges.sh --execute" >&2
    echo "Or run the explicit fallback phase: ${label}-nomerge" >&2
    exit 2
  fi
}

is_running_out() {
  local out="$1"
  ps -axo command= \
    | grep -F "octt " \
    | grep -F -- "--out $out" \
    | grep -v "grep -F" >/dev/null
}

run_if_missing() {
  local label="$1"
  local out="$2"
  local marker="$3"
  shift 3

  if [[ -f "$marker" ]]; then
    echo "skip: $label already complete ($marker)"
    return 0
  fi
  if is_running_out "$out"; then
    echo "running: $label already has a live process for $out" >&2
    return 75
  fi

  echo
  echo "== $label =="
  printf '+'
  printf ' %q' "$@"
  echo
  "$@"
}

cmd_status() {
  echo "Running octt jobs:"
  pgrep -fl "octt run|octt scaling|uv run octt" || true
  echo
  echo "Disk:"
  df -h .
  echo
  scripts/octt_disk_budget.py | sed -n '1,4p'
  echo
  git status --short
}

cmd_local() {
  uv run pytest
  uv run ruff check

  echo
  echo "== default all-model preflight should be BLOCKED by Ultra rank64 =="
  set +e
  uv run octt preflight --dry-run
  local rc=$?
  set -e
  if [[ "$rc" -ne 2 ]]; then
    echo "Expected default preflight exit 2, got $rc" >&2
    exit 1
  fi

  echo
  echo "== Ultra compatibility preflight should pass =="
  uv run octt preflight --dry-run \
    --model "$ULTRA_MODEL" \
    --lora-rank 32 \
    --no-merge
}

cmd_paid_4b() {
  source_env
  local smoke_out="runs/${PERSONA}-4b-smoke-${TAG}"
  local quick_out="runs/${PERSONA}-4b-quick-${TAG}"

  run_if_missing "paid 4B smoke" "$smoke_out" "$smoke_out/eval_results.json" \
    uv run octt run "$PERSONA" \
      --execute \
      --scale smoke \
      --model "$MODEL_4B" \
      --teacher "$TEACHER" \
      --out "$smoke_out"

  run_if_missing "paid 4B quick" "$quick_out" "$quick_out/eval_results.json" \
    uv run octt run "$PERSONA" \
      --execute \
      --scale quick \
      --model "$MODEL_4B" \
      --teacher "$TEACHER" \
      --out "$quick_out"
}

cmd_lighteval_smoke() {
  source_env
  local smoke_out="runs/${PERSONA}-4b-smoke-${TAG}"
  local marker="$smoke_out/eval/capabilities/capability_eval.json"

  if [[ ! -f "$smoke_out/eval_results.json" ]]; then
    echo "Missing $smoke_out/eval_results.json. Run paid-4b first." >&2
    exit 2
  fi

  run_if_missing "LightEval smoke on 4B local merge" "$smoke_out" "$marker" \
    uv run octt run "$PERSONA" \
      --execute \
      --scale smoke \
      --model "$MODEL_4B" \
      --teacher "$TEACHER" \
      --out "$smoke_out" \
      --no-eval \
      --eval-merged-local \
      --eval-capabilities \
      --capability-suite smoke
}

cmd_arch_smoke() {
  source_env
  require_free_gib "${ARCH_MERGE_MIN_FREE_GIB:-30}" "arch-smoke"
  local out="runs/${PERSONA}-arch-control-smoke-${TAG}"
  run_if_missing "paid architecture-control smoke with merge" "$out" "$out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale smoke \
      --teacher "$TEACHER" \
      --model Qwen/Qwen3.6-27B \
      --model Qwen/Qwen3.6-35B-A3B \
      --out "$out"
}

cmd_arch_smoke_nomerge() {
  source_env
  local out="runs/${PERSONA}-arch-control-smoke-nomerge-${TAG}"
  run_if_missing "paid architecture-control no-merge smoke fallback" "$out" "$out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale smoke \
      --teacher "$TEACHER" \
      --model Qwen/Qwen3.6-27B \
      --model Qwen/Qwen3.6-35B-A3B \
      --no-merge \
      --out "$out"
}

cmd_six_smoke() {
  source_env
  require_free_gib "${SIX_MERGE_MIN_FREE_GIB:-320}" "six-smoke"

  local four_out="runs/${PERSONA}-four-model-smoke-${TAG}"
  run_if_missing "paid four-model rank64 smoke with merge (4B, 9B, 27B, Nano)" "$four_out" "$four_out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale smoke \
      --teacher "$TEACHER" \
      --model Qwen/Qwen3.5-4B \
      --model Qwen/Qwen3.5-9B \
      --model Qwen/Qwen3.6-27B \
      --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
      --out "$four_out"

  local super_out="runs/${PERSONA}-super-smoke-${TAG}"
  run_if_missing "paid Super rank64 smoke with merge" "$super_out" "$super_out/eval_results.json" \
    uv run octt run "$PERSONA" \
      --execute \
      --scale smoke \
      --teacher "$TEACHER" \
      --model "$SUPER_MODEL" \
      --out "$super_out"

  local ultra_out="runs/${PERSONA}-ultra-rank32-smoke-${TAG}"
  run_if_missing "paid Ultra rank32 merge compatibility smoke" "$ultra_out" "$ultra_out/eval_results.json" \
    uv run octt run "$PERSONA" \
      --execute \
      --scale smoke \
      --teacher "$TEACHER" \
      --model "$ULTRA_MODEL" \
      --lora-rank 32 \
      --out "$ultra_out"
}

cmd_six_smoke_nomerge() {
  source_env

  local four_out="runs/${PERSONA}-four-model-smoke-nomerge-${TAG}"
  run_if_missing "paid four-model rank64 no-merge smoke fallback (4B, 9B, 27B, Nano)" "$four_out" "$four_out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale smoke \
      --teacher "$TEACHER" \
      --model Qwen/Qwen3.5-4B \
      --model Qwen/Qwen3.5-9B \
      --model Qwen/Qwen3.6-27B \
      --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
      --no-merge \
      --out "$four_out"

  local super_out="runs/${PERSONA}-super-smoke-nomerge-${TAG}"
  run_if_missing "paid Super rank64 no-merge smoke fallback" "$super_out" "$super_out/eval_results.json" \
    uv run octt run "$PERSONA" \
      --execute \
      --scale smoke \
      --teacher "$TEACHER" \
      --model "$SUPER_MODEL" \
      --no-merge \
      --out "$super_out"

  local ultra_out="runs/${PERSONA}-ultra-rank32-smoke-nomerge-${TAG}"
  run_if_missing "paid Ultra rank32 no-merge compatibility smoke fallback" "$ultra_out" "$ultra_out/eval_results.json" \
    uv run octt run "$PERSONA" \
      --execute \
      --scale smoke \
      --teacher "$TEACHER" \
      --model "$ULTRA_MODEL" \
      --lora-rank 32 \
      --no-merge \
      --out "$ultra_out"
}

cmd_paper_template() {
  if [[ "${ALLOW_PAPER:-0}" != "1" ]]; then
    echo "Refusing paper scale without ALLOW_PAPER=1." >&2
    echo "Run smoke gates first, then choose Ultra policy explicitly." >&2
    exit 2
  fi
  source_env
  require_free_gib "${PAPER_MERGE_MIN_FREE_GIB:-165}" "paper-template"

  local paper_out="runs/${PERSONA}-paper-rank64-supported-${TAG}"
  run_if_missing "paper rank64 supported models with merge, excluding Ultra" "$paper_out" "$paper_out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale paper \
      --teacher "$TEACHER" \
      --model Qwen/Qwen3.5-4B \
      --model Qwen/Qwen3.5-9B \
      --model Qwen/Qwen3.6-27B \
      --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
      --model "$SUPER_MODEL" \
      --out "$paper_out"

  echo
  echo "Paper Ultra is intentionally not run here: Tinker blocks Ultra rank64."
  echo "For compatibility only, run Ultra with --lora-rank 32 --no-merge in its own labeled output."
}

cmd_paper_template_nomerge() {
  if [[ "${ALLOW_PAPER:-0}" != "1" ]]; then
    echo "Refusing paper scale without ALLOW_PAPER=1." >&2
    echo "Run smoke gates first, then choose Ultra policy explicitly." >&2
    exit 2
  fi
  source_env

  local paper_out="runs/${PERSONA}-paper-rank64-supported-nomerge-${TAG}"
  run_if_missing "paper rank64 no-merge supported models fallback, excluding Ultra" "$paper_out" "$paper_out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale paper \
      --teacher "$TEACHER" \
      --model Qwen/Qwen3.5-4B \
      --model Qwen/Qwen3.5-9B \
      --model Qwen/Qwen3.6-27B \
      --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
      --model "$SUPER_MODEL" \
      --no-merge \
      --out "$paper_out"

  echo
  echo "Paper Ultra is intentionally not run here: Tinker blocks Ultra rank64."
}

case "${1:-}" in
  status) cmd_status ;;
  local) cmd_local ;;
  paid-4b) cmd_paid_4b ;;
  lighteval-smoke) cmd_lighteval_smoke ;;
  arch-smoke) cmd_arch_smoke ;;
  arch-smoke-nomerge) cmd_arch_smoke_nomerge ;;
  six-smoke) cmd_six_smoke ;;
  six-smoke-nomerge) cmd_six_smoke_nomerge ;;
  paper-template) cmd_paper_template ;;
  paper-template-nomerge) cmd_paper_template_nomerge ;;
  all-safe)
    cmd_local
    cmd_paid_4b
    cmd_lighteval_smoke
    cmd_arch_smoke
    cmd_six_smoke
    ;;
  ""|-h|--help|help) usage ;;
  *)
    usage >&2
    exit 2
    ;;
esac
