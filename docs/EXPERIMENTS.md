# Script ↔ table/figure map, and how to reproduce

Every script is driven by command-line flags only and resolves all paths through
`experiments/paths.py`. Table and figure numbers below are the ones used throughout this
repository; the LaTeX `sci-paper` labels are stable even when floats shift.

## 0. Data preparation

```bash
# (a) public benchmarks (4), redistributed under their own licences -- download them yourself
python experiments/tools/download_datasets.py          # batch pull from HF mirrors, subsettable
# expected layout (AKT official 5-fold splits):
#   data/public/AKT_repo/data/{assist2009_pid,assist2015,assist2017_pid,statics2011}/

# (b) the self-built dataset is released with this repository
#   data/dnui_ikg/*.csv

# (c) unify into .pkl under data/processed/
python experiments/data/prep_datasets.py --only dnui          # DNUI-IKG (real mastery + semi-synthetic sequences)
python experiments/data/prep_bench.py --only assist09,assist15,assist17,statics11
```

`prep_datasets.py` emits two objects: `dnui_real` (student × knowledge-point mastery, real labels)
and `dnui_syn` (DNUI-IKG-S, semi-synthetic sequences — the DNUI-IKG row of Table 1 refers to this).

## 1. Table and figure map

| Paper item | Content | Script | Artefact |
|---|---|---|---|
| Table 1 | Scale and sequence statistics of the 7 datasets | `tools/dataset_stats.py`, `tools/dataset_edges.py`, `tools/make_paper_tables.py` | `results/table_dataset.tex` |
| Table 2 | Our reproduction vs. two independently published reference values | `tools/make_paper_tables.py` (reads `experiments/data/published_benchmarks.json`) | `results/table_kt_pub.tex` |
| Table 3 | Key hyper-parameters (selected on the validation split) | `run/run_kt_tune.py` | `results/kt_model_cfg_*.json` |
| Table 4 | **11 models × 7 datasets** AUC matrix (mean ± SD over folds) | `run/run_bench.sh` → `tools/aggregate_bench.py` → `tools/make_paper_tables.py` | `results/kt_bench_*.json`, `results/table_kt_bench.tex` |
| Table 5 | Paired tests of DGCL-KT against the strongest baseline per dataset (ΔAUC / 95% CI / p / MDES) | `tools/kt_significance.py` | `results/kt_signif.json` |
| Table 6 | Resource recommendation on DNUI-IKG | `run/run_rec.py --task rec`, `tools/csg_attribution.py` | `results/rec_dnui_syn.json` |
| Table 7 | Resource recommendation on ASSIST09 (scope degenerates to the full set) | `run/run_rec.py --task rec --dataset assist09` | `results/rec_assist09.json` |
| Table 8 | Gate decisions under incomplete annotations | `run/run_scope_decision_v2.py` | `results/scope_decision.json` |
| Table 9 | (a) learning-path quality on 3 datasets; (b) three replanning strategies | `run/run_rec.py --task path`, `--task mdrl` | `results/path_*.json`, `results/mdrl_dnui_syn.json` |
| Table 10 | GES reliability: HR / ETR / SCR / PCR / NDCG | `run/run_rec.py --task ges` (needs an LLM credential or the disk cache) | `results/ges_dnui_syn.json` |
| Table 11 | Sequential-encoder comparison (Transformer / GRU / LSTM × depth) | `run/run_enc_ablation.py` → `tools/make_enc_table.py` | `results/kt_*_encabl.json` |
| Table 12 | (a) module ablation; (b) sparse/dense knowledge-point groups | `run/run_abl_multi.py`, `run/run_sparse_eval.py` | `results/kt_*_abl.json`, `kt_*_sparse.json` |
| Table 13 | (a) induced vs. syllabus graph; (b) edge-level leakage; (c) re-inference with a prefix-only graph | `tools/prereq_reliability.py`, `tools/graph_leak_audit.py`, `tools/kt_gstrain_pair.py`; for (c) rerun `run_rec.py --task rec/path --graph-scope cut` | `results/prereq_induction.json`, `graph_leak.json`, `results/variants/*-scopePrefix.json` |
| Fig. 1 | Framework overview | `figs/make_figures.py::fig_framework` | `figs/fig_framework.pdf` |
| Fig. 2–9 | Dataset statistics, AUC error bars, per-seed scatter, compliance vs. accuracy, γ sensitivity, GES, contamination case, replanning cost, ablation | `figs/make_figures.py` | `figs/*.pdf` |

## 2. Three reproduction tiers (cheapest first)

### Tier A — verify the numbers (no GPU, ~10 s)

All per-fold and per-seed result JSONs are shipped, so every table can be rebuilt directly:

```bash
bash scripts/03_tables.sh        # one shot: Tables 1-4 + snapshot diff + Tables 6-10 re-check

# the equivalent manual steps (note --frozen: results/ is a snapshot, mtimes are meaningless)
python experiments/tools/make_paper_tables.py --frozen   # Tables 1/2/3/4
python experiments/tools/kt_significance.py              # Table 5 paired tests
python experiments/tools/check_downstream.py             # Tables 6-10 + in-text quantities
```

Outputs land in `results/table_*.tex` and can be diffed line by line against
`results/paper_tables/` (the shipped snapshot). `check_downstream.py` exits non-zero on any
mismatch, so it can be wired into CI.

### Tier B — rerun one dataset (single GPU: 10 min to hours)

```bash
# smoke: DNUI-IKG, all 11 models, 8 epochs
python experiments/run/run_kt.py --dataset dnui_syn --models all --epochs 8 --tag smoke

# full: all models on one dataset, reusing the validation-selected configuration
python experiments/run/run_kt.py --dataset assist09_akt_f1 --models all --epochs 60 \
       --patience 10 --tag bench --model-cfg-file results/kt_model_cfg_assist09_akt_f1.json
```

`--use-tune` loads the validation-best configuration from `results/kt_tune_<ds>_stage*.json`;
`--model-cfg-file` loads the merged final configuration (baselines included). The test split never
participates in any selection.

### Tier C — rebuild the main tables from scratch (single V100: several days)

```bash
bash experiments/run/run_bench.sh      # stage A our chain tuning -> A2 baseline tuning -> B 5 folds x 11 models -> C aggregation
```

The script embeds four **gates** which are part of the evidence for the "unified implementation,
fair tuning" claim; please keep them intact:

| Gate | Script | What it catches |
|---|---|---|
| Baseline tuning completeness | `tools/check_basetune.py` | only `lr` searched, capacity stages skipped, yet marked "done" (silently creates an unfair comparison) |
| Content-level check of our own config | `tools/check_maincfg.py` | a chain tuning run that died halfway, leaving `kt_maincfg` from a previous round |
| Comparison-matrix completeness | `tools/check_bench.py` | a fold that wrote only some models after a CUDA OOM, then passes aggregation as "tiny variance" |
| Learning-rate endpoint audit | `tools/kt_endpoint_audit.py`, `tools/check_lr_boundary.py` | the optimum sits on the grid boundary, i.e. the search never finished |

## 3. Downstream tasks and ablations

```bash
# recommendation (DNUI-IKG; --graph-scope all = main protocol, cut = de-leakage audit)
python experiments/run/run_rec.py --dataset dnui_syn --task rec  --graph-scope all
python experiments/run/run_rec.py --dataset assist09 --task rec  --graph-scope cut --tag gscut

# path planning and mastery-triggered replanning (24 steps x 50 students as in the paper)
python experiments/run/run_rec.py --dataset dnui_syn --task path --scope-policy down
python experiments/run/run_rec.py --dataset dnui_syn --task mdrl --mdrl-steps 24 \
       --mdrl-students 50 --scope-policy down

# gate robustness under missing annotations (Table 8)
python experiments/run/run_scope_decision_v2.py --students 400 --seeds 42,43,44

# generation (needs DEEPSEEK_API_KEY or a hit in the logs/ges_cache disk cache)
python experiments/run/run_rec.py --dataset dnui_syn --task ges --llm-students 40

# module ablation (3 seeds) / sparse-group evaluation / encoder comparison
python experiments/run/run_abl_multi.py    --dataset dnui_syn --seeds 42,43,44
python experiments/run/run_sparse_eval.py  --dataset dnui_syn --seeds 42,43,44
python experiments/run/run_enc_ablation.py --datasets dnui_syn,assist09
```

> `run_rec.py` and `run_kt.py` write to the paper-named files under `results/` by default.
> **Always pass `--tag` when re-running**, otherwise you overwrite the shipped reported values.

Two flags determine the numbers and are easy to get wrong:

- `--scope-policy {down,eq}` — `down` (default) implements Eq. 10 as a one-sided inequality
  (higher grades may reuse lower-grade resources); `eq` is the legacy set-equality reading kept
  only for the comparison reported in `results/variants/`.
- `--graph-scope {all,cut}` — `all` induces the prerequisite graph from all sequences (main
  protocol); `cut` re-induces it from the evaluation prefix only (de-leakage audit).

## 4. Result file conventions

```
results/<task>_<dataset>.json                  main protocol, cited by the paper
results/variants/<task>_<dataset>-<variant>.json   cross-checks and audits
results/exploratory/                           implemented but NOT reported in the paper
```

Every main file carries a `_meta` block recording task, dataset, scope policy, graph scope,
cutoff, seed, sample size and the paper table it supports.

`results/exploratory/` currently holds `repair_plan.json` (budget sweep over local path repair) and
`evidence_intervention.json` (LLM evidence self-correction). Both were implemented during
exploration and are **not** used to support any reported claim; the scripts are
`experiments/run/run_repair_plan.py` and `experiments/run/run_evidence_intervention.py`.

## 5. Dual-graph construction and the resource layer (optional)

```bash
# export and normalise the teaching-platform database into data/dnui_ikg/*.csv
python experiments/kg/build_dual_graph.py
# build the structural resource layer (item -> knowledge-point mountings, prerequisite closure)
# output: data/dnui_ikg/resource_layer.json -- required by run_scope_decision_v2.py (Table 8)
python experiments/kg/build_resource_layer.py
# load into Neo4j (in the container /var/lib/neo4j/import points at data/dnui_ikg)
python experiments/kg/load_neo4j.py
```

`build_resource_layer.py` is a **structural** mapping, not a behavioural simulation: the raw
`materials.csv` has only a header, and any resource layer derived from behaviour would inherit the
same ceiling effect as the log itself. It normalises each question stem (with options and answer)
into an item entity and aggregates every mounting of that entity across courses — 33 duplicated
item groups, 21 of them cross-course. This is why "which knowledge point does this item belong to"
is not unique in the data: a missing annotation means one of several mountings was not recorded,
which is exactly the premise of the equivalence-set argument in §2.3.3.

The graph-construction steps only rebuild the graph and support visualisation; apart from the
resource layer they **do not** affect any reported number — every experiment reads the CSVs.
