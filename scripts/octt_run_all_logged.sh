#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEFAULT_PHASES=(local paid-4b lighteval-smoke arch-smoke six-smoke)
LOG_ROOT="${LOG_ROOT:-runs/octt-plan-logs}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="$LOG_ROOT/$RUN_STAMP"
SUMMARY="$LOG_DIR/summary.tsv"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

usage() {
  cat <<'EOF'
Usage: scripts/octt_run_all_logged.sh [phase...]

Runs the OCTT execution plan through scripts/octt_plan.sh and records logs.
With no phases, runs:
  local paid-4b lighteval-smoke arch-smoke six-smoke

Examples:
  scripts/octt_run_all_logged.sh
  TAG=v3 scripts/octt_run_all_logged.sh paid-4b lighteval-smoke
  CONTINUE_ON_ERROR=1 scripts/octt_run_all_logged.sh local arch-smoke six-smoke
  ALLOW_PAPER=1 scripts/octt_run_all_logged.sh paper-template

Environment:
  LOG_ROOT=runs/octt-plan-logs   Parent directory for logs.
  RUN_STAMP=<name>               Log run directory name.
  CONTINUE_ON_ERROR=1            Keep going after a failed phase.
  PERSONA, TAG, MODEL_4B, TEACHER, ALLOW_PAPER are passed through.

Outputs:
  <log-dir>/metadata.txt
  <log-dir>/summary.tsv
  <log-dir>/<n>-<phase>.log
  <log-dir>/<n>-<phase>.before-status.txt
  <log-dir>/<n>-<phase>.after-status.txt
  runs/octt-plan-logs/latest -> latest log directory
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -gt 0 ]]; then
  PHASES=("$@")
else
  PHASES=("${DEFAULT_PHASES[@]}")
fi

mkdir -p "$LOG_DIR"
ln -sfn "$(basename "$LOG_DIR")" "$LOG_ROOT/latest"

now_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

epoch_seconds() {
  date +%s
}

record_metadata() {
  {
    echo "started_at=$(now_utc)"
    echo "root=$ROOT"
    echo "log_dir=$LOG_DIR"
    echo "phases=${PHASES[*]}"
    echo "persona=${PERSONA:-humorous}"
    echo "tag=${TAG:-v2}"
    echo "model_4b=${MODEL_4B:-Qwen/Qwen3.5-4B}"
    echo "teacher=${TEACHER:-Qwen/Qwen3.5-397B-A17B}"
    echo "continue_on_error=$CONTINUE_ON_ERROR"
    echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
    echo "git_branch=$(git branch --show-current 2>/dev/null || true)"
    echo
    echo "git_status:"
    git status --short || true
  } > "$LOG_DIR/metadata.txt"
  printf "phase\tstatus\texit_code\tstarted_at\tfinished_at\tduration_seconds\tlog\n" > "$SUMMARY"
}

record_status() {
  local path="$1"
  {
    echo "recorded_at=$(now_utc)"
    scripts/octt_plan.sh status
  } > "$path" 2>&1
}

run_phase() {
  local index="$1"
  local phase="$2"
  local safe_phase="${phase//[^A-Za-z0-9_.-]/_}"
  local prefix
  prefix="$(printf "%02d-%s" "$index" "$safe_phase")"
  local log="$LOG_DIR/$prefix.log"
  local before="$LOG_DIR/$prefix.before-status.txt"
  local after="$LOG_DIR/$prefix.after-status.txt"
  local started_at finished_at start_s end_s duration rc status

  started_at="$(now_utc)"
  start_s="$(epoch_seconds)"
  record_status "$before"

  {
    echo "phase=$phase"
    echo "started_at=$started_at"
    echo "log=$log"
    echo
    echo "+ scripts/octt_plan.sh $phase"
  } | tee "$log"

  scripts/octt_plan.sh "$phase" 2>&1 | tee -a "$log"
  rc="${PIPESTATUS[0]}"

  finished_at="$(now_utc)"
  end_s="$(epoch_seconds)"
  duration="$((end_s - start_s))"
  record_status "$after"

  if [[ "$rc" -eq 0 ]]; then
    status="ok"
  elif [[ "$rc" -eq 75 ]]; then
    status="already_running"
  else
    status="failed"
  fi

  {
    echo
    echo "finished_at=$finished_at"
    echo "exit_code=$rc"
    echo "status=$status"
    echo "duration_seconds=$duration"
  } | tee -a "$log"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$phase" "$status" "$rc" "$started_at" "$finished_at" "$duration" "$log" \
    >> "$SUMMARY"

  if [[ "$rc" -ne 0 && "$CONTINUE_ON_ERROR" != "1" ]]; then
    echo
    echo "Stopping after phase '$phase' ($status, exit $rc)."
    echo "Logs: $LOG_DIR"
    exit "$rc"
  fi
}

record_metadata

echo "Log directory: $LOG_DIR"
echo "Summary: $SUMMARY"
echo "Phases: ${PHASES[*]}"

index=1
for phase in "${PHASES[@]}"; do
  run_phase "$index" "$phase"
  index="$((index + 1))"
done

{
  echo
  echo "completed_at=$(now_utc)"
  echo "final_status=ok"
} >> "$LOG_DIR/metadata.txt"

echo
echo "All requested phases completed."
echo "Logs: $LOG_DIR"
echo "Summary: $SUMMARY"
