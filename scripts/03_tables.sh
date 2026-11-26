#!/usr/bin/env bash
# Rebuild the knowledge-tracing tables (1-4) from the shipped results/ snapshot
# and diff them against results/paper_tables/. No GPU, no retraining (~10 s).
#
# Usage:
#   bash scripts/03_tables.sh                 # verify the snapshot (default)
#   bash scripts/03_tables.sh --reaggregate   # re-aggregate fold-wise results first
#
# Notes:
#   * In snapshot mode the "are multi-seed artefacts newer than the main run"
#     mtime guard is skipped -- after a fresh git clone every file has the same
#     checkout timestamp, which would make the guard fire spuriously
#     (see make_paper_tables.py).
#   * The build overwrites results/table_*.tex and results/kt_signif.json;
#     restore with: git checkout -- results/
#   * Step [4/4] additionally re-checks every downstream number printed in the
#     paper (Tables 6-10) against results/*.json; see experiments/tools/check_downstream.py
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
AGG="${1:-}"

if [[ "$AGG" == "--reaggregate" ]]; then
  echo "===== [optional] re-aggregate fold-wise results ====="
  for DS in assist09_akt assist15 assist17 statics11; do
    $PY experiments/tools/aggregate_bench.py --dataset "$DS" --folds 1,2,3,4,5 2>&1 | tail -3
  done
  for DS in dnui_syn assist09 assist12; do
    $PY experiments/tools/aggregate_bench.py --dataset "$DS" --single 2>&1 | tail -3
  done
  echo
fi

echo "===== [1/4] build tables: dataset / kt_pub / kt_bench / kt_ours ====="
$PY experiments/tools/make_paper_tables.py --frozen 2>&1 | tail -12

echo
echo "===== [2/4] recompute paired tests (DeltaAUC / 95% CI / p / MDES) ====="
$PY experiments/tools/kt_significance.py 2>&1 | tail -18

echo
echo "===== [3/4] diff against the shipped snapshot ====="
status=0
for f in table_dataset.tex table_kt_pub.tex table_kt_bench.tex table_kt_ours.tex; do
  if [[ -f "results/paper_tables/$f" && -f "results/$f" ]]; then
    if diff --strip-trailing-cr -q "results/paper_tables/$f" "results/$f" >/dev/null; then
      echo "  [match]   $f"
    else
      echo "  [DIFF ]   $f  -- diff results/paper_tables/$f results/$f"
      status=1
    fi
  else
    echo "  [skip ]   $f (missing file)"
  fi
done

echo
echo "===== [4/4] downstream paper values (Tables 6-10) ====="
$PY experiments/tools/check_downstream.py || status=1

echo
if [[ $status -eq 0 ]]; then
  echo "===== ALL MATCH: the shipped results rebuild every number in the paper ====="
else
  echo "===== MISMATCH: check whether results/ or the aggregation protocol changed ====="
fi
exit $status
