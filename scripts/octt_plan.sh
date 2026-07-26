#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# TAG defaults to a fresh suffix so paid phases never silently skip an older
# completed run (run_if_missing keys on a marker file such as
# <out>/eval_results.json, and only when that marker records SUCCESS — see
# marker_records_success). Bump on each new known-good baseline; v2/v3 predate
# the verified-recipe fixes.
PERSONA="${PERSONA:-humorous}"
TAG="${TAG:-v4}"
MODEL_4B="${MODEL_4B:-Qwen/Qwen3.5-4B}"
TEACHER="${TEACHER:-Qwen/Qwen3.5-397B-A17B}"

SUPER_MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
ULTRA_MODEL="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
INKLING_MODEL="thinkingmachines/Inkling"
NANO_MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
# Study judge for ALL revealed-preferences evals. Default: Nemotron Nano — the
# judge dominates paper-scale cost and Nano is ~15x cheaper per token than the
# 397B teacher. One judge across every phase keeps rungs comparable; Elo judged
# by different models is NOT comparable (the v4 4B baseline used the teacher —
# re-baseline under Nano before comparing it against later rungs).
JUDGE="${JUDGE:-$NANO_MODEL}"

# Retry supervisor knobs (see run_with_retry). OCTT_MAX_ATTEMPTS=1 disables
# retries entirely; every other default is tuned for "the Mac dropped DNS for a
# few minutes", not for "Tinker is down all night".
OCTT_MAX_ATTEMPTS="${OCTT_MAX_ATTEMPTS:-5}"
OCTT_RETRY_BACKOFF_SECS="${OCTT_RETRY_BACKOFF_SECS:-60}"
OCTT_RETRY_BACKOFF_MAX_SECS="${OCTT_RETRY_BACKOFF_MAX_SECS:-600}"
OCTT_RETRY_TAIL_LINES="${OCTT_RETRY_TAIL_LINES:-400}"
OCTT_NETWORK_WAIT_SECS="${OCTT_NETWORK_WAIT_SECS:-1800}"
OCTT_TINKER_HOST="${OCTT_TINKER_HOST:-tinker.thinkingmachines.dev}"

# Cross-machine run lock (see acquire_run_lock). is_running_out only sees local
# processes, but two Macs now hold the same run directories: a concurrent
# double-run would append to the same judgment caches and double-spend. The
# lock is written once per phase and never refreshed, so the staleness window
# has to exceed the longest single phase (the paper-half Inkling run took
# hours). OCTT_FORCE_LOCK=1 breaks a lock the machine can no longer clear.
OCTT_LOCK_STALE_SECS="${OCTT_LOCK_STALE_SECS:-86400}"
OCTT_FORCE_LOCK="${OCTT_FORCE_LOCK:-0}"
OCTT_HELD_LOCKS=()

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
  paper-4b          Guarded paper gate: ONE rung (4B, adopt-only) at paper scale
                    under the uniform-rank32 policy. Requires ALLOW_PAPER=1.
  dense-smoke       Qwen dense-series smoke (4B/9B/27B, no-merge) — SWEEP_PLAN.md step 1.
  dense-sweep       Qwen dense sweep at DENSE_SWEEP_SCALE (default
                    paper-half-uncapped), adopt-only, no-merge. Requires ALLOW_PAPER=1 — SWEEP_PLAN.md step 2.
  paper-template    Guarded paper-scale template. Requires ALLOW_PAPER=1.
  paper-template-nomerge
                    Explicit fallback: paper template with --no-merge.
  inkling-smoke     Paid Inkling self-distillation smoke, then quick (INKLING_PLAN.md
                    Phase 2). Teacher == student == Inkling; judge stays $JUDGE.
  inkling-paper     Guarded Inkling paper-scale run (Phase 3). Requires ALLOW_PAPER=1.
  all-safe          local -> paid-4b -> lighteval-smoke -> arch-smoke -> six-smoke.

Environment:
  PERSONA=humorous        Persona to run.
  TAG=v2                  Output suffix.
  TINKER_API_KEY=...      Loaded from .env for paid commands.
  ALLOW_PAPER=1           Required for paper-template.
  DENSE_SWEEP_SCALE=paper-half-uncapped
  INKLING_PAPER_SCALE=paper-half
                          Half-budget ramp preset for inkling-paper (default: paper).
  ARCH_MERGE_MIN_FREE_GIB=30    Free disk required before architecture-control merge.
  SIX_MERGE_MIN_FREE_GIB=320    Free disk required before full six-model merge.
  PAPER_MERGE_MIN_FREE_GIB=165  Free disk required before paper supported-model merge.

Retry supervisor (paid phases only; transient network failures ONLY):
  OCTT_MAX_ATTEMPTS=5           Launches per phase. 1 disables retries.
  OCTT_RETRY_BACKOFF_SECS=60    First backoff; doubles per attempt.
  OCTT_RETRY_BACKOFF_MAX_SECS=600
                                Backoff ceiling.
  OCTT_RETRY_TAIL_LINES=400     Log tail scanned for TRANSIENT symptoms. Fatal
                                patterns are scanned over the whole log.
  OCTT_NETWORK_WAIT_SECS=1800   Give up if the API host stays unreachable.
  OCTT_TINKER_HOST=tinker.thinkingmachines.dev
                                Host probed before every relaunch.
  A signal death (exit >= 128: pkill, OOM killer, Ctrl-C) is NEVER retried.

Skip-if-done gate (a phase is skipped only when its marker records SUCCESS):
  OCTT_ACCEPT_INCOMPLETE_MARKER=1
                                Honour a marker that records FAILED rungs (or a
                                failed capability eval) anyway. For a rung whose
                                failure is permanent and accepted; the report
                                still shows it as FAILED.

Cross-machine run lock (<out>/.octt_run.lock: host, pid, UTC timestamp):
  OCTT_LOCK_STALE_SECS=86400    A foreign lock older than this is abandoned.
  OCTT_FORCE_LOCK=1             Break any lock. Only when the other machine is
                                provably not running the phase.

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

# Does a marker file record a SUCCESSFUL run? Sets MARKER_REASON either way.
#
# Existence is not success. `octt scaling` writes report.json even when every
# rung failed (the report IS the diagnostic for a broken sweep), and
# capabilities.evaluate writes capability_eval.json with status=failed rather
# than raising after the expensive stages ran. Stat'ing those markers retires a
# phase that never produced a result — permanently, since nothing ever deletes
# them. scripts/octt_phase_complete.py reads the marker instead
# (octt/phase_status.py holds the rules).
#
# Fail closed: a check that cannot render a clean 0/1 verdict counts as NOT
# complete. Re-running a finished phase costs a manifest resume and cached judge
# verdicts; skipping an unfinished one costs the whole phase, silently.
MARKER_REASON=""
marker_records_success() {
  local marker="$1"
  local out="${2:-}"
  local rc=0
  MARKER_REASON="$(uv run python scripts/octt_phase_complete.py "$marker" ${out:+"$out"} 2>&1)" \
    || rc=$?
  case "$rc" in
    0) return 0 ;;
    1) return 1 ;;
    *)
      MARKER_REASON="marker check itself failed (exit ${rc}): ${MARKER_REASON}"
      return 1
      ;;
  esac
}

this_host() {
  local host
  host="$(hostname 2>/dev/null || true)"
  printf '%s' "${host:-${HOSTNAME:-unknown-host}}"
}

lock_field() {
  local lock="$1"
  local key="$2"
  local line
  line="$(grep -m 1 -E "^${key}=" "$lock" 2>/dev/null || true)"
  printf '%s' "${line#*=}"
}

# Refuse to start a paid phase that another machine already owns.
#
# is_running_out greps local `ps`, so it is blind to the second Mac holding the
# same run directory; the lock lives IN the out dir, which is the one thing both
# machines can see. A lock from THIS host is honoured only while its pid is
# alive (authoritative, no clock involved). A lock from another host can only be
# aged out — we cannot probe a remote pid — so it holds until
# OCTT_LOCK_STALE_SECS passes, and an unparseable timestamp counts as fresh.
# Returns 75 (EX_TEMPFAIL) when the phase must not start.
acquire_run_lock() {
  local out="$1"
  local label="$2"
  local lock="$out/.octt_run.lock"
  local host now owner owner_pid owner_epoch owner_utc age tmp

  host="$(this_host)"
  now="$(date -u +%s)"
  mkdir -p "$out"

  if [[ -f "$lock" ]]; then
    owner="$(lock_field "$lock" host)"
    owner_pid="$(lock_field "$lock" pid)"
    owner_epoch="$(lock_field "$lock" epoch)"
    owner_utc="$(lock_field "$lock" utc)"
    age=-1
    if [[ "$owner_epoch" =~ ^[0-9]+$ ]]; then
      age="$((now - owner_epoch))"
    fi

    if [[ "$OCTT_FORCE_LOCK" == "1" ]]; then
      echo "lock: OCTT_FORCE_LOCK=1 — breaking lock held by '${owner:-unknown}' pid ${owner_pid:-?} (${owner_utc:-unknown time})." >&2
    elif [[ "$owner" == "$host" ]] \
      && [[ "$owner_pid" =~ ^[0-9]+$ ]] \
      && kill -0 "$owner_pid" 2>/dev/null; then
      echo "lock: refusing $label — pid $owner_pid on this host ($host) still owns $out." >&2
      return 75
    elif [[ "$owner" != "$host" ]] && [[ "$age" -lt 0 || "$age" -lt "$OCTT_LOCK_STALE_SECS" ]]; then
      echo "lock: refusing $label — $out is locked by host '${owner:-unknown}' (pid ${owner_pid:-?}, ${owner_utc:-unknown time})." >&2
      echo "lock: a second machine running this phase would double-spend and corrupt the judgment cache." >&2
      echo "lock: if that machine is definitely not running it, rerun with OCTT_FORCE_LOCK=1 (or delete $lock)." >&2
      return 75
    else
      echo "lock: reclaiming abandoned lock from '${owner:-unknown}' pid ${owner_pid:-?} (${owner_utc:-unknown time}, age ${age}s)." >&2
    fi
  fi

  # $$ is the supervisor, not the child: `pkill -f "octt scaling"` kills the
  # child only, so this pid stays alive to release the lock on the way out.
  tmp="$lock.$$"
  {
    printf 'host=%s\n' "$host"
    printf 'pid=%s\n' "$$"
    printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'epoch=%s\n' "$now"
    printf 'label=%s\n' "$label"
  } >"$tmp"
  mv -f "$tmp" "$lock"
  OCTT_HELD_LOCKS+=("$lock")
}

release_run_lock() {
  local out="$1"
  local lock="$out/.octt_run.lock"
  # Only drop a lock we still own: if someone forced a takeover while we ran,
  # deleting theirs on our way out would re-open the double-spend window.
  if [[ -f "$lock" ]] \
    && [[ "$(lock_field "$lock" host)" == "$(this_host)" ]] \
    && [[ "$(lock_field "$lock" pid)" == "$$" ]]; then
    rm -f "$lock"
  fi
}

release_all_run_locks() {
  local lock
  for lock in ${OCTT_HELD_LOCKS[@]+"${OCTT_HELD_LOCKS[@]}"}; do
    release_run_lock "${lock%/.octt_run.lock}"
  done
  OCTT_HELD_LOCKS=()
}

# Release on every exit path, not just the tidy ones: a lock left behind blocks
# the other machine for OCTT_LOCK_STALE_SECS.
trap 'release_all_run_locks' EXIT
trap 'release_all_run_locks; exit 130' INT
trap 'release_all_run_locks; exit 143' TERM

tinker_reachable() {
  if command -v curl >/dev/null 2>&1; then
    # Any HTTP answer proves the network path; we are probing reachability, not
    # authorization, so a 401/404 body is a success here.
    if curl -sS -o /dev/null --max-time 10 "https://${OCTT_TINKER_HOST}/" >/dev/null 2>&1; then
      return 0
    fi
    return 1
  fi
  if ping -c 1 -t 5 "$OCTT_TINKER_HOST" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

wait_for_network() {
  local waited=0
  local probe_interval=15

  while true; do
    if tinker_reachable; then
      return 0
    fi
    if [[ "$waited" -ge "$OCTT_NETWORK_WAIT_SECS" ]]; then
      return 1
    fi
    echo "  network: $OCTT_TINKER_HOST unreachable (${waited}s waited); re-probing in ${probe_interval}s" >&2
    sleep "$probe_interval"
    waited="$((waited + probe_interval))"
  done
}

# Print one of: signal | fatal | transient | unknown.
#
# Usage: classify_failure <log> [exit-status]
#
# Only "transient" earns a relaunch. HOW the process died outranks everything in
# the log: a shell reports a signal death as 128+signal, so rc >= 128 means an
# operator's `pkill -f "octt scaling"` on a runaway paid run (143), the OOM
# killer (137) or Ctrl-C (130). The tail of a run somebody killed usually shows
# exactly the network noise that made them kill it, so a text-only verdict would
# relaunch the run they were trying to stop. A signal is never retried.
#
# Then fatal wins over transient — but fatal is matched against the WHOLE log
# while transient only looks at the tail. Real paid logs run 8k-23k lines, so a
# 402 in minute three followed by connection noise used to scroll out of the
# 400-line window and get relaunched (inkling-paper-half-pirate-v6.log opens
# with 67 such lines). tee truncates per attempt, so a whole-log scan still only
# ever sees the attempt it is judging.
#
# Everything unrecognised stays "unknown" (no retry): a wrong retry can spend
# money, a missed retry costs a manual relaunch.
classify_failure() {
  local log="$1"
  local rc="${2:-1}"
  local tail_text

  if [[ "$rc" -ge 128 ]]; then
    echo signal
    return 0
  fi

  # Patterns are SPECIFIC and mostly case-sensitive on purpose. Logs carry
  # sampled persona prose, trait deltas and session UUIDs, so "403", "budget",
  # "when refusing" and "connection" all occur innocently; the machinery spells
  # them "Error code: 403", "exceeds budget", "Refusing " and
  # "APIConnectionError". A false "fatal" only costs a manual relaunch, but a
  # false "transient" spends money. Verified over every log in
  # runs/octt-plan-logs/: zero false hits, whole-file scan included.
  if grep -q -E \
    -e '^Refusing |ALLOW_PAPER|status: BLOCKED|^blockers:|exceeds budget' \
    -e 'PermissionDeniedError|AuthenticationError|BadRequestError|NotFoundError|UnprocessableEntityError|APIStatusError' \
    -e 'Error code: (400|401|402|403|404|409|422)' \
    -- "$log" 2>/dev/null \
    || grep -q -E -i \
      -e '(HTTP|status[ _-]?code)[^0-9]{0,4}(400|401|402|403|404|409|422)' \
      -e 'payment required|insufficient (funds|credit|credits|quota|balance)|invalid api key|quota exceeded' \
      -- "$log" 2>/dev/null; then
    echo fatal
    return 0
  fi

  # A network drop makes the SDK emit hundreds of "Telemetry queue full,
  # dropping events" lines; unfiltered they push the real symptom clean out of
  # the tail window (the dense-sweep-pirate-v7 log is 2925 lines, 2699 of them
  # that one message).
  tail_text="$(grep -v -F 'Telemetry queue full' -- "$log" 2>/dev/null | tail -n "$OCTT_RETRY_TAIL_LINES" || true)"

  if grep -q -E \
    -e 'APIConnectionError|APITimeoutError|ConnectTimeout|ReadTimeout|ConnectionError|Connection error' \
    -e 'Connection (reset|refused|aborted)|Broken pipe|Server disconnected|RemoteProtocolError' \
    -e 'dns error|no connections available|Temporary failure in name resolution|nodename nor servname|Name or service not known' \
    -e 'Session heartbeat failed|session will be terminated' \
    -e 'TimeoutError|Read timed out|Max retries exceeded' \
    -e 'Bad Gateway|Service Unavailable|Gateway Time-?out|InternalServerError' \
    <<<"$tail_text"; then
    echo transient
    return 0
  fi

  echo unknown
}

# Relaunch a paid phase after a TRANSIENT network failure and nothing else.
#
# Cheap because it is the same command: every stage is manifest-tracked and
# judge verdicts are cached on disk, so a relaunch skips completed stages and
# re-uses cached judgments. This exists because dense-sweep-pirate-v7 died 13
# minutes in when the Mac lost DNS ("dns error: no connections available" ->
# tinker.APIConnectionError -> "Session heartbeat failed for 313 seconds ... the
# session will be terminated") and then sat dead for hours.
#
# Guards are unchanged: --execute, ALLOW_PAPER and the disk gates are all
# decided by the caller before we ever see the command.
run_with_retry() {
  local label="$1"
  local out="$2"
  local marker="$3"
  shift 3

  local attempt=1
  local backoff="$OCTT_RETRY_BACKOFF_SECS"
  local attempt_log rc verdict errexit_was_on
  attempt_log="$(mktemp "${TMPDIR:-/tmp}/octt-attempt.XXXXXX")"

  while true; do
    # 2>&1 into tee so the classifier sees the same bytes as the run log; the
    # same pattern scripts/octt_run_all_logged.sh already uses per phase. tee
    # truncates, so every attempt is judged on its own output.
    errexit_was_on=0
    case "$-" in *e*) errexit_was_on=1 ;; esac
    set +e
    "$@" 2>&1 | tee "$attempt_log"
    rc="${PIPESTATUS[0]}"
    # Restore what the caller had; never assume errexit was on. A caller that
    # deliberately turned it off must not get it switched back on behind its
    # back — that would abort the script on the next tolerated non-zero command.
    if [[ "$errexit_was_on" -eq 1 ]]; then
      set -e
    fi

    if [[ "$rc" -eq 0 ]]; then
      rm -f "$attempt_log"
      return 0
    fi

    verdict="$(classify_failure "$attempt_log" "$rc")"
    echo
    echo "retry: '$label' attempt ${attempt}/${OCTT_MAX_ATTEMPTS} exited ${rc}, classified ${verdict}" >&2

    if [[ "$verdict" == "signal" ]]; then
      echo "retry: killed by signal $((rc - 128)) — an operator stop (pkill/Ctrl-C) or the OOM killer." >&2
      echo "retry: a signalled paid run is NEVER relaunched; rerun the phase by hand if the kill was a mistake." >&2
      rm -f "$attempt_log"
      return "$rc"
    fi
    if [[ "$verdict" != "transient" ]]; then
      echo "retry: no relaunch (${verdict} failure) — fix it and rerun the phase by hand." >&2
      rm -f "$attempt_log"
      return "$rc"
    fi
    # run_if_missing's skip-if-done check, re-applied per attempt: the marker is
    # the record that the paid work landed, and a phase can write it and still
    # exit non-zero on a network error during teardown. Relaunching then re-pays
    # nothing but re-enters a finished run for no reason. The marker must record
    # SUCCESS, not merely exist — a sweep that lost the network on every rung
    # writes report.json too, and calling that "complete" would report a phase
    # that produced nothing as done.
    if [[ -n "$marker" ]] && marker_records_success "$marker" "$out"; then
      echo "retry: '$label' wrote $marker before exiting ${rc}; the work is complete, not relaunching." >&2
      echo "retry:   $MARKER_REASON" >&2
      rm -f "$attempt_log"
      return 0
    fi
    if [[ "$attempt" -ge "$OCTT_MAX_ATTEMPTS" ]]; then
      echo "retry: attempt budget spent (OCTT_MAX_ATTEMPTS=${OCTT_MAX_ATTEMPTS}); giving up on '$label'." >&2
      rm -f "$attempt_log"
      return "$rc"
    fi

    echo "retry: transient network failure — sleeping ${backoff}s, then probing ${OCTT_TINKER_HOST}." >&2
    sleep "$backoff"
    if ! wait_for_network; then
      echo "retry: $OCTT_TINKER_HOST still unreachable after ${OCTT_NETWORK_WAIT_SECS}s; giving up on '$label'." >&2
      rm -f "$attempt_log"
      return "$rc"
    fi
    # A relaunch alongside a surviving child would double-spend on one out dir.
    if is_running_out "$out"; then
      echo "retry: a live process still owns $out; refusing to relaunch '$label'." >&2
      rm -f "$attempt_log"
      return 75
    fi

    backoff="$((backoff * 2))"
    if [[ "$backoff" -gt "$OCTT_RETRY_BACKOFF_MAX_SECS" ]]; then
      backoff="$OCTT_RETRY_BACKOFF_MAX_SECS"
    fi
    attempt="$((attempt + 1))"

    echo
    echo "== $label (attempt ${attempt}/${OCTT_MAX_ATTEMPTS}, $(date -u +%Y-%m-%dT%H:%M:%SZ)) =="
    echo "   $OCTT_TINKER_HOST reachable again; the manifest resumes completed stages."
    printf '+'
    printf ' %q' "$@"
    echo
  done
}

run_if_missing() {
  local label="$1"
  local out="$2"
  local marker="$3"
  shift 3

  if [[ -f "$marker" ]]; then
    if marker_records_success "$marker" "$out"; then
      echo "skip: $label already complete ($MARKER_REASON)"
      return 0
    fi
    if [[ "${OCTT_ACCEPT_INCOMPLETE_MARKER:-0}" == "1" ]]; then
      echo "skip: $label — accepting an INCOMPLETE marker (OCTT_ACCEPT_INCOMPLETE_MARKER=1):" >&2
      echo "  $MARKER_REASON" >&2
      return 0
    fi
    echo "rerun: $label — $marker exists but does NOT record a completed run:" >&2
    echo "  $MARKER_REASON" >&2
    echo "  Relaunching: the report is kept for diagnosis, finished stages resume from each" >&2
    echo "  rung's manifest and judge verdicts come from the on-disk cache." >&2
    echo "  If the failure is permanent (bad model id, refused gate, dry-run directory), fix" >&2
    echo "  it, drop that rung, or use a fresh TAG; to accept this marker as-is, re-run with" >&2
    echo "  OCTT_ACCEPT_INCOMPLETE_MARKER=1." >&2
  fi
  if is_running_out "$out"; then
    echo "running: $label already has a live process for $out" >&2
    return 75
  fi
  # Local `ps` first (cheapest, most precise), then the cross-machine lock.
  if ! acquire_run_lock "$out" "$label"; then
    return 75
  fi

  echo
  echo "== $label =="
  printf '+'
  printf ' %q' "$@"
  echo
  local rc=0
  run_with_retry "$label" "$out" "$marker" "$@" || rc=$?
  release_run_lock "$out"
  return "$rc"
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

  echo
  echo "== Inkling with local merge should be BLOCKED (975B base) =="
  set +e
  uv run octt preflight --dry-run \
    --model "$INKLING_MODEL" \
    --teacher "$INKLING_MODEL" \
    --lora-rank 32
  local inkling_rc=$?
  set -e
  if [[ "$inkling_rc" -ne 2 ]]; then
    echo "Expected Inkling merge preflight exit 2, got $inkling_rc" >&2
    exit 1
  fi

  echo
  echo "== Inkling self-distillation preflight (rank 32, no-merge) should pass =="
  uv run octt preflight --dry-run \
    --model "$INKLING_MODEL" \
    --teacher "$INKLING_MODEL" \
    --lora-rank 32 \
    --no-merge
}

cmd_paid_4b() {
  source_env
  local smoke_out="runs/${PERSONA}-4b-smoke-${TAG}"
  local quick_out="runs/${PERSONA}-4b-quick-${TAG}"

  # --condition all repeats the full judgment budget across the paper's three
  # embodiment conditions (adopt/feels/random) so the re-baseline yields the
  # per-condition Elo breakdown PLAN.md Phase 1 calls for (~3x eval judgments).
  run_if_missing "paid 4B smoke" "$smoke_out" "$smoke_out/eval_results.json" \
    uv run octt run "$PERSONA" \
      --execute \
      --scale smoke \
      --model "$MODEL_4B" \
      --teacher "$TEACHER" \
      --judge "$JUDGE" \
      --condition all \
      --out "$smoke_out"

  run_if_missing "paid 4B quick" "$quick_out" "$quick_out/eval_results.json" \
    uv run octt run "$PERSONA" \
      --execute \
      --scale quick \
      --model "$MODEL_4B" \
      --teacher "$TEACHER" \
      --judge "$JUDGE" \
      --condition all \
      --out "$quick_out"
}

cmd_lighteval_smoke() {
  source_env
  local smoke_out="runs/${PERSONA}-4b-smoke-${TAG}"
  local marker="$smoke_out/eval/capabilities/capability_eval.json"

  # Prerequisite, same rule as the skip gate: the 4B smoke must have COMPLETED,
  # not merely left a file behind.
  if ! marker_records_success "$smoke_out/eval_results.json" "$smoke_out"; then
    echo "Refusing lighteval-smoke: the 4B smoke it merges is not complete." >&2
    echo "  $MARKER_REASON" >&2
    echo "Run 'paid-4b' first." >&2
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
      --judge "$JUDGE" \
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
      --judge "$JUDGE" \
      --model Qwen/Qwen3.6-27B \
      --model Qwen/Qwen3.6-35B-A3B \
      --no-merge \
      --out "$out"
}

cmd_six_smoke() {
  source_env
  require_free_gib "${SIX_MERGE_MIN_FREE_GIB:-320}" "six-smoke"

  local four_out="runs/${PERSONA}-four-model-smoke-${TAG}"
  run_if_missing "paid four-model uniform-rank32 smoke with merge (4B, 9B, 27B, Nano)" "$four_out" "$four_out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale smoke \
      --teacher "$TEACHER" \
      --judge "$JUDGE" \
      --model Qwen/Qwen3.5-4B \
      --model Qwen/Qwen3.5-9B \
      --model Qwen/Qwen3.6-27B \
      --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
      --out "$four_out"

  # Super/Ultra run through `octt scaling` (single rung) so they pick up the
  # uniform-rank32 study policy (rank 32, lr 1e-4) like every other rung.
  local super_out="runs/${PERSONA}-super-smoke-${TAG}"
  run_if_missing "paid Super uniform-rank32 smoke with merge" "$super_out" "$super_out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale smoke \
      --teacher "$TEACHER" \
      --judge "$JUDGE" \
      --model "$SUPER_MODEL" \
      --out "$super_out"

  local ultra_out="runs/${PERSONA}-ultra-rank32-smoke-${TAG}"
  run_if_missing "paid Ultra rank32 merge compatibility smoke" "$ultra_out" "$ultra_out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale smoke \
      --teacher "$TEACHER" \
      --judge "$JUDGE" \
      --model "$ULTRA_MODEL" \
      --out "$ultra_out"
}

cmd_six_smoke_nomerge() {
  source_env

  local four_out="runs/${PERSONA}-four-model-smoke-nomerge-${TAG}"
  run_if_missing "paid four-model uniform-rank32 no-merge smoke fallback (4B, 9B, 27B, Nano)" "$four_out" "$four_out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale smoke \
      --teacher "$TEACHER" \
      --judge "$JUDGE" \
      --model Qwen/Qwen3.5-4B \
      --model Qwen/Qwen3.5-9B \
      --model Qwen/Qwen3.6-27B \
      --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
      --no-merge \
      --out "$four_out"

  local super_out="runs/${PERSONA}-super-smoke-nomerge-${TAG}"
  run_if_missing "paid Super uniform-rank32 no-merge smoke fallback" "$super_out" "$super_out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale smoke \
      --teacher "$TEACHER" \
      --judge "$JUDGE" \
      --model "$SUPER_MODEL" \
      --no-merge \
      --out "$super_out"

  local ultra_out="runs/${PERSONA}-ultra-rank32-smoke-nomerge-${TAG}"
  run_if_missing "paid Ultra rank32 no-merge compatibility smoke fallback" "$ultra_out" "$ultra_out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale smoke \
      --teacher "$TEACHER" \
      --judge "$JUDGE" \
      --model "$ULTRA_MODEL" \
      --no-merge \
      --out "$ultra_out"
}

cmd_paper_4b() {
  if [[ "${ALLOW_PAPER:-0}" != "1" ]]; then
    echo "Refusing paper scale without ALLOW_PAPER=1." >&2
    exit 2
  fi
  source_env
  require_free_gib 10 "paper-4b"

  # The paper gate (PLAN.md Phase 3): one rung, one condition, before any
  # fan-out. Own out dir — a later fan-out phase must EXCLUDE 4B or reuse this
  # dir's rung subdirectory, or it will re-pay this run.
  local out="runs/${PERSONA}-paper-rank32-4b-${TAG}"
  run_if_missing "paper gate: 4B rung, adopt-only, uniform-rank32" "$out" "$out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale paper \
      --teacher "$TEACHER" \
      --judge "$JUDGE" \
      --model "$MODEL_4B" \
      --condition adopt \
      --out "$out"
}

cmd_dense_smoke() {
  source_env

  local out="runs/${PERSONA}-dense-smoke-nomerge-${TAG}"
  run_if_missing "paid Qwen dense-series smoke, uniform rank32, no-merge (4B, 9B, 27B)" "$out" "$out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale smoke \
      --teacher "$TEACHER" \
      --judge "$JUDGE" \
      --model Qwen/Qwen3.5-4B \
      --model Qwen/Qwen3.5-9B \
      --model Qwen/Qwen3.6-27B \
      --no-merge \
      --out "$out"

  echo
  echo "Gate for dense-sweep: every rung completes, no FAILED rows. Smoke shifts are"
  echo "completion gates, not evidence (SWEEP_PLAN.md Phase 1)."
}

cmd_dense_sweep() {
  if [[ "${ALLOW_PAPER:-0}" != "1" ]]; then
    echo "Refusing paper-scale spend without ALLOW_PAPER=1." >&2
    echo "Run 'dense-smoke' first (SWEEP_PLAN.md step 1), then re-run with ALLOW_PAPER=1." >&2
    exit 2
  fi
  source_env

  # No disk gate: --no-merge downloads no adapters (SWEEP_PLAN.md / A12).
  # paper-half with the introspection corpus untruncated: a fixed SFT token budget
  # shrinks the corpus as models get wordier, confounding the size axis this sweep
  # measures (see octt/config.py PAPER_HALF_UNCAPPED).
  local scale="${DENSE_SWEEP_SCALE:-paper-half-uncapped}"
  local out="runs/${PERSONA}-dense-${scale}-rank32-${TAG}"
  # 35B-A3B is the within-family architecture control: same Qwen3.6 generation as
  # the 27B, MoE instead of dense — the only dense-vs-MoE contrast with no family
  # confound (L1/L6).
  run_if_missing "PAID Qwen sweep at ${scale}, adopt-only, no-merge (dense 4B/9B/27B + MoE 35B-A3B control)" "$out" "$out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale "$scale" \
      --teacher "$TEACHER" \
      --judge "$JUDGE" \
      --condition adopt \
      --model Qwen/Qwen3.5-4B \
      --model Qwen/Qwen3.5-9B \
      --model Qwen/Qwen3.6-27B \
      --model Qwen/Qwen3.6-35B-A3B \
      --no-merge \
      --out "$out"

  echo
  echo "Comparable by construction to runs/pirate-inkling-paper-half-rank32-v6"
  echo "(same scale, condition, judge, rank). Next: SWEEP_PLAN.md Phase 2."
}

cmd_paper_template() {
  if [[ "${ALLOW_PAPER:-0}" != "1" ]]; then
    echo "Refusing paper scale without ALLOW_PAPER=1." >&2
    echo "Run smoke gates first, then choose Ultra policy explicitly." >&2
    exit 2
  fi
  source_env
  require_free_gib "${PAPER_MERGE_MIN_FREE_GIB:-165}" "paper-template"

  local paper_out="runs/${PERSONA}-paper-rank32-mergeable-${TAG}"
  run_if_missing "paper uniform-rank32 mergeable models, excluding Ultra (local-merge disk)" "$paper_out" "$paper_out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale paper \
      --teacher "$TEACHER" \
      --judge "$JUDGE" \
      --model Qwen/Qwen3.5-4B \
      --model Qwen/Qwen3.5-9B \
      --model Qwen/Qwen3.6-27B \
      --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
      --model "$SUPER_MODEL" \
      --out "$paper_out"

  local ultra_out="runs/${PERSONA}-paper-ultra-rank32-nomerge-${TAG}"
  run_if_missing "paper Ultra uniform-rank32 no-merge (base weights too large to merge locally)" "$ultra_out" "$ultra_out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale paper \
      --teacher "$TEACHER" \
      --judge "$JUDGE" \
      --model "$ULTRA_MODEL" \
      --no-merge \
      --out "$ultra_out"

  echo
  echo "All rungs run the uniform-rank32 study policy (rank 32, lr 1e-4); Ultra"
  echo "skips only the LOCAL merge (its base weights don't fit on this disk)."
}

cmd_paper_template_nomerge() {
  if [[ "${ALLOW_PAPER:-0}" != "1" ]]; then
    echo "Refusing paper scale without ALLOW_PAPER=1." >&2
    echo "Run smoke gates first, then choose Ultra policy explicitly." >&2
    exit 2
  fi
  source_env

  local paper_out="runs/${PERSONA}-paper-rank32-all-nomerge-${TAG}"
  run_if_missing "paper uniform-rank32 no-merge fallback, all six rungs" "$paper_out" "$paper_out/report.json" \
    uv run octt scaling "$PERSONA" \
      --execute \
      --scale paper \
      --teacher "$TEACHER" \
      --judge "$JUDGE" \
      --model Qwen/Qwen3.5-4B \
      --model Qwen/Qwen3.5-9B \
      --model Qwen/Qwen3.6-27B \
      --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
      --model "$SUPER_MODEL" \
      --model "$ULTRA_MODEL" \
      --no-merge \
      --out "$paper_out"
}

cmd_inkling_smoke() {
  source_env
  local smoke_out="runs/${PERSONA}-inkling-smoke-${TAG}"
  local quick_out="runs/${PERSONA}-inkling-quick-${TAG}"

  # Self-distillation: teacher == student == Inkling (constitution-prompted vs
  # unprompted, pinned-effort renderer both sides); external judge; no local
  # merge (975B base). INKLING_PLAN.md Phase 2 gate: after these complete,
  # scan the dpo_pairs/introspection sidecars for template or reasoning-token
  # leakage (scripts/octt_check_sidecars.py) before any paper-scale spend;
  # inkling-paper re-runs that scan automatically.
  run_if_missing "paid Inkling smoke (self-distill)" "$smoke_out" "$smoke_out/eval_results.json" \
    uv run octt run "$PERSONA" \
      --execute \
      --scale smoke \
      --model "$INKLING_MODEL" \
      --teacher "$INKLING_MODEL" \
      --judge "$JUDGE" \
      --lora-rank 32 \
      --learning-rate 1e-4 \
      --no-merge \
      --condition all \
      --out "$smoke_out"

  run_if_missing "paid Inkling quick (self-distill)" "$quick_out" "$quick_out/eval_results.json" \
    uv run octt run "$PERSONA" \
      --execute \
      --scale quick \
      --model "$INKLING_MODEL" \
      --teacher "$INKLING_MODEL" \
      --judge "$JUDGE" \
      --lora-rank 32 \
      --learning-rate 1e-4 \
      --no-merge \
      --condition all \
      --out "$quick_out"
}

cmd_inkling_paper() {
  if [[ "${ALLOW_PAPER:-0}" != "1" ]]; then
    echo "Refusing Inkling paper scale without ALLOW_PAPER=1." >&2
    echo "Run inkling-smoke first and inspect the transcripts (INKLING_PLAN.md Phase 2 gate):" >&2
    echo "  uv run python scripts/octt_check_sidecars.py runs/${PERSONA}-inkling-smoke-${TAG} runs/${PERSONA}-inkling-quick-${TAG}" >&2
    exit 2
  fi
  # Mechanical Phase-2 gate: any completed smoke/quick sidecars for this
  # persona+tag must be leakage-clean before paper-scale money moves.
  local gate_dirs=()
  for d in "runs/${PERSONA}-inkling-smoke-${TAG}" "runs/${PERSONA}-inkling-quick-${TAG}"; do
    [[ -d "$d" ]] && gate_dirs+=("$d")
  done
  if [[ ${#gate_dirs[@]} -gt 0 ]]; then
    uv run python scripts/octt_check_sidecars.py "${gate_dirs[@]}" || {
      echo "Sidecar leakage gate FAILED — fix generation before paper-scale spend." >&2
      exit 2
    }
  fi
  source_env

  # Single embodiment condition first (INKLING_PLAN.md decision 2): the eval is
  # the dominant cost line at Inkling sample prices; --condition all is for the
  # run that becomes the reported number. INKLING_PAPER_SCALE=paper-half runs
  # the half-budget ramp preset before committing to the full envelope.
  local scale="${INKLING_PAPER_SCALE:-paper}"
  local out="runs/${PERSONA}-inkling-${scale}-rank32-${TAG}"
  run_if_missing "paid Inkling ${scale}-scale (self-distill, 1 condition)" "$out" "$out/eval_results.json" \
    uv run octt run "$PERSONA" \
      --execute \
      --scale "$scale" \
      --model "$INKLING_MODEL" \
      --teacher "$INKLING_MODEL" \
      --judge "$JUDGE" \
      --lora-rank 32 \
      --learning-rate 1e-4 \
      --no-merge \
      --condition adopt \
      --out "$out"
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
  paper-4b) cmd_paper_4b ;;
  dense-smoke) cmd_dense_smoke ;;
  dense-sweep) cmd_dense_sweep ;;
  paper-template) cmd_paper_template ;;
  paper-template-nomerge) cmd_paper_template_nomerge ;;
  inkling-smoke) cmd_inkling_smoke ;;
  inkling-paper) cmd_inkling_paper ;;
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
