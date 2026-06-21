#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXECUTE=0
if [[ "${1:-}" == "--execute" ]]; then
  EXECUTE=1
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  cat <<'EOF'
Usage: scripts/octt_prune_local_merges.sh [--execute]

Dry-runs by default. With --execute, deletes local adapter materializations:
  runs/**/merge/dpo_adapter
  runs/**/merge/sft_adapter
  runs/**/merge/merged

This does not delete manifests, checkpoints, eval caches, reports, or Tinker
checkpoint URIs. It does remove local merged-adapter files, so use it only when
you are relying on no-merge/sft-proxy evaluation or can recreate local merges.
EOF
  exit 0
elif [[ "$#" -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  exit 2
fi

mapfile -d '' targets < <(
  find runs -type d \( -path '*/merge/dpo_adapter' -o -path '*/merge/sft_adapter' -o -path '*/merge/merged' \) -print0 2>/dev/null
)

if [[ "${#targets[@]}" -eq 0 ]]; then
  echo "No local merge artifacts found."
  exit 0
fi

echo "Local merge artifacts:"
du -sh "${targets[@]}" 2>/dev/null || true
echo
df -h .

if [[ "$EXECUTE" -ne 1 ]]; then
  echo
  echo "Dry run only. Re-run with --execute to delete these directories."
  exit 0
fi

echo
echo "Deleting ${#targets[@]} directories..."
rm -rf "${targets[@]}"
echo "Done."
df -h .
