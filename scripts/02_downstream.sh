#!/usr/bin/env bash
# Downstream experiments: resource recommendation / learning-path planning /
# mastery-triggered replanning / module ablation / sparse-group evaluation /
# encoder comparison / gate robustness under missing annotations.
#
# Notes
#   * --graph-scope for run_rec.py: `all` is the main protocol (graph induced from
#     all sequences). `cut` re-induces the graph from the evaluation prefix only
#     and is reported as the de-leakage audit (Table 13(c)), never as a main table.
#   * Every command carries --tag so that the shipped paper values in results/
#     are never overwritten.
#   * GES (--task ges) needs DEEPSEEK_API_KEY unless the disk cache is hit;
#     it is not part of the default run: bash scripts/02_downstream.sh ges
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
DS="${DS:-dnui_syn}"
WHAT="${1:-all}"

if [[ ! -f "data/processed/${DS}.pkl" ]]; then
  echo "missing data/processed/${DS}.pkl -- run bash scripts/00_prepare_data.sh first"
  exit 1
fi

run_rec() {
  echo; echo "===== resource recommendation: ${DS} (graph-scope=all, main protocol) ====="
  $PY experiments/run/run_rec.py --dataset "$DS" --task rec --graph-scope all \
      --scope-policy down --tag repro
  echo; echo "===== CSG attribution: does the gate only release slots? ====="
  $PY experiments/tools/csg_attribution.py --dataset "$DS"
}

run_path() {
  echo; echo "===== learning-path planning: ${DS} ====="
  $PY experiments/run/run_rec.py --dataset "$DS" --task path --graph-scope all \
      --scope-policy down --tag repro
  echo; echo "===== mastery-triggered replanning: ${DS} (24 steps x 50 students) ====="
  $PY experiments/run/run_rec.py --dataset "$DS" --task mdrl --mdrl-steps 24 \
      --mdrl-students 50 --scope-policy down --tag repro
}

run_gate() {
  echo; echo "===== gate robustness under missing annotations (Table 8) ====="
  SCOPE2_OUT="results/scope_decision_repro.json" SCOPE_POLICY=down \
    $PY experiments/run/run_scope_decision_v2.py --students 400 --seeds 42,43,44
}

run_abl() {
  echo; echo "===== module ablation (3 seeds): ${DS} ====="
  $PY experiments/run/run_abl_multi.py --dataset "$DS" --seeds 42,43,44
  echo; echo "===== sparse/dense knowledge-point groups: ${DS} ====="
  $PY experiments/run/run_sparse_eval.py --dataset "$DS" --seeds 42,43,44
}

run_enc() {
  echo; echo "===== sequential encoder comparison ====="
  $PY experiments/run/run_enc_ablation.py --datasets dnui_syn,assist09
  $PY experiments/tools/make_enc_table.py
}

run_ges() {
  echo; echo "===== GES (needs DEEPSEEK_API_KEY or a cache hit) ====="
  $PY experiments/run/run_rec.py --dataset "$DS" --task ges --llm-students 40 \
      --scope-policy down --tag repro
}

case "$WHAT" in
  all)  run_rec; run_path; run_gate; run_abl; run_enc ;;
  rec)  run_rec ;;
  path) run_path ;;
  gate) run_gate ;;
  abl)  run_abl ;;
  enc)  run_enc ;;
  ges)  run_ges ;;
  *) echo "unknown argument: $WHAT (choose from all|rec|path|gate|abl|enc|ges)"; exit 1 ;;
esac

echo
echo "===== DONE: results are written to results/*_repro.json ====="
