# KG²-LLM

**Coupling dual knowledge graphs with large language models for learning tracking and
resource recommendation.**

This repository releases the reference implementation, the full experimental pipeline and the
self-built **DNUI-IKG** dataset for dual knowledge-graph + LLM learning tracking and resource
recommendation. It contains

- a unified, fairly-tuned implementation of **11 knowledge-tracing (KT) models** on 7 datasets;
- the full pipeline of the three constrained pathways built on top of the dual graph
  (course knowledge graph + learner cognitive graph): **CSG** (school-stage/major semantic
  gating), **GES** (graph-evidence-subgraph constrained RAG) and **MDRL** (mastery-triggered
  replanning), plus resource recommendation and learning-path planning;
- a **prerequisite-graph de-leakage audit**, including re-induction from evaluation prefixes only;
- the self-built **DNUI-IKG** dataset (real curriculum structure, real grade/major scope labels,
  semi-synthetic answer sequences);
- the scripts that regenerate every reported table and figure.

> Repository: <https://github.com/renzhs/KG2-LLM> · `git clone git@github.com:renzhs/KG2-LLM.git`

---

## 1. Headline findings

The headline claim is deliberately narrow: **the graph does not make the predictor more
accurate, but it makes the downstream system verifiably compliant.**

| Claim | Evidence |
|---|---|
| No stable accuracy gain on the KT side | After Holm correction over 7 datasets, DGCL-KT is significantly better than the strongest baseline on **5 of 7**; the graph-collaboration term is inside the noise band |
| The gate removes cross-scope contamination | CSCR@10 drops from **0.0557** (strongest related baseline) to **0.0000**; an unconstrained LLM recommends 12.25% out-of-scope items |
| Executable, not merely ordered, paths | Executable-coverage (EC) **0.2439** vs. **0.0302** for in-scope random (**8.08×**), with sequential-compliance 1.0000 |
| Grounded generation has a price | Evidence traceability **0.7300** (structured verification) and scope compliance **1.0000**, at a ranking cost (NDCG@10 0.1907 → 0.1501) |
| Replanning is cheap but not free | **326** invocations vs. **1 150** for replanning at every step (**−71.7%**), i.e. 63% *more* calls than a fixed 5-step schedule, which is why it also preserves path quality |
| Robust to missing annotations | Under 70% of annotations masked, the hard gate's false-rejection rate is **0.7184**; equivalence-set mounting reduces it to **0.0037**, while a same-budget union baseline reaches a false-accept rate of **0.3316** |

Two results are reported as **negative** and argued explicitly rather than hidden: (i) the
graph-collaboration term gives no ranking gain on the recommendation side, and (ii) the
reverse-calibration component is at best a boundary refinement, not a component contribution.

## 2. Repository layout

```
KG2-LLM/
├── data/
│   ├── dnui_ikg/            DNUI-IKG (de-identified CSVs, released with this repo)
│   └── public/              public benchmarks (download yourself; not redistributed)
├── experiments/
│   ├── paths.py             single source of truth for every path
│   ├── models/              kt_models.py (11 KT models), rec_models.py, kg_induct.py, llm_ges.py
│   ├── data/                prep_datasets.py (.pkl), prep_bench.py (AKT 5-fold splits)
│   ├── run/                 run_kt.py, run_rec.py, run_abl_multi.py, run_scope_decision_v2.py, …
│   ├── tools/               aggregation, paired tests, leakage audit, tuning gates, table builders,
│   │                        check_downstream.py (re-verifies every downstream value)
│   ├── kg/                  dual-graph construction and optional Neo4j loading
│   └── figs/                make_figures.py (all figures)
├── results/                 reported values (JSON) + table snapshot (paper_tables/)
│   ├── variants/            cross-checks: legacy scope policy, prefix-only graph (de-leakage)
│   └── exploratory/         analyses implemented but not part of the reported set
├── figs/                    figures (PDF + PNG)
├── scripts/                 stage-wise entry points
└── docs/
    ├── DATA_CARD.md         dataset provenance, fields, privacy, semi-synthetic protocol, checksums
    ├── EXPERIMENTS.md       script ↔ table/figure map + three reproduction tiers
    └── REPRODUCIBILITY.md   environment, seeds, runtimes, known deviations
```

Every result file in `results/` carries a `_meta` block (task, dataset, scope policy, graph
scope, seed, sample size, the reported table it supports), so a number can be traced back to its
protocol without reading this README.

## 3. Installation

```bash
git clone https://github.com/renzhs/KG2-LLM.git && cd KG2-LLM
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- Python ≥ 3.10 (validated on 3.12.3)
- PyTorch ≥ 2.0; a GPU is optional — the whole DNUI-IKG pipeline runs on CPU, only slower
- Generating new GES results requires `DEEPSEEK_API_KEY` (see `docs/REPRODUCIBILITY.md`);
  re-checking reported values does not

## 4. Quick start (no GPU, ~10 s)

All per-fold and per-seed results are shipped, so **every table can be rebuilt without training**
and diffed against the shipped snapshot:

```bash
bash scripts/03_tables.sh
# [match]  table_dataset.tex
# [match]  table_kt_pub.tex
# [match]  table_kt_bench.tex
# [match]  table_kt_ours.tex
# ... 206 checks, 0 mismatches
# ===== ALL MATCH: the shipped results rebuild every reported number =====
```

Step `[4/4]` runs `experiments/tools/check_downstream.py`, which re-reads `results/*.json` and
compares every downstream value in Tables 6–10, together with the quantities quoted alongside
them, against the shipped reference values, item by item.

One training run (DNUI-IKG × 11 models × 8 epochs; 171 s on a single V100, minutes on CPU):

```bash
python experiments/data/prep_datasets.py --only dnui
python experiments/run/run_kt.py --dataset dnui_syn --models all --epochs 8 --tag smoke
```

## 5. Full reproduction

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Three tiers by cost:

| Tier | What it does | Cost |
|---|---|---|
| A | Rebuild all tables and re-verify all reported values from shipped results | no GPU, ~10 s |
| B | Re-run 11 models on one dataset | 8 epochs on DNUI-IKG: 171 s on one V100; 60 epochs: ~20 min to hours depending on size |
| C | Reproduce the main tables from scratch (tuning + 5 folds × 11 models + aggregation) | several GPU-days on one V100; resumable |

`experiments/run/run_bench.sh` enforces four **gates** (baseline tuning completeness, content-level
verification of our own configurations, comparison-matrix completeness, learning-rate endpoint
audit). Their purpose is to prevent a silently under-tuned baseline from biasing the comparison;
they are themselves part of the evidence for the "unified implementation, fair tuning" claim and
should be kept as-is.

## 6. The DNUI-IKG dataset

- **Content**: 3 undergraduate courses, 66 knowledge points, 88 prerequisite edges, 467 questions,
  1 060 students with answer records, 169 351 responses, 17 862 mastery labels. It is the only
  dataset here that carries **school-stage and major scope labels**.
- **Privacy**: student number, name, department and major code were dropped; the retained
  `student_id` is an internal integer that cannot be linked to any institutional record. All
  columns were scanned for phone/e-mail/student-number patterns with zero hits.
- **Semi-synthetic declaration (please read)**: the platform log is "answer until correct", so the
  accuracy is 99.35% and the mastery label is identically 1 — no variance to model. The sequence
  benchmark used for knowledge tracing, **DNUI-IKG-S**, therefore keeps the **real prerequisite
  DAG, the real scope labels and the real interaction volumes**, and generates answer sequences
  with a 2PL IRT + slip/guess process (`prep_datasets.py` rebuilds it exactly, `seed=42`).
  **Graph structure and labels are real; the sequences are semi-synthetic.** Results support
  claims about "real curriculum structure under real scope constraints"; they must not be used to
  infer real students' learning gains.
- Fields, checksums and the generation protocol: [`docs/DATA_CARD.md`](docs/DATA_CARD.md).
- The public benchmarks (ASSIST09/12/15/17, Statics2011) are **not redistributed** because of
  their licences; see §0 of `docs/EXPERIMENTS.md` for how to obtain them.

## 7. Protocol notes that matter when reading the numbers

These three definitions were revised during the development of this release. They are stated
here because they determine the scale of the reported values.

1. **School-stage admission is a one-sided inequality.** A resource is admissible for a student
   if `grade(student) ≥ min(grade_scope(resource))`, i.e. higher grades may reuse lower-grade
   resources (the one-sided admission rule). Under this policy 72.55% of the true positives fall
   inside the student's scope. The alternative set-equality reading shrinks that share to 46.49%
   and inflates CSCR to 0.3604; results under that legacy policy are kept in `results/variants/`
   for comparison, never in the main tables.
2. **Executable coverage (EC) checks unmastered prerequisites.** For a target knowledge point to
   count as covered, *every* prerequisite must either be already mastered (`X > 0` in the training
   segment) or appear earlier in the path. A prerequisite that is neither mastered nor scheduled
   blocks the target. EC is reported under this strict definition; sequential compliance —
   the ordering of prerequisite pairs that *do* appear in the path — is a separate, weaker metric
   and is named accordingly.
3. **Evidence traceability (ETR) is verified structurally.** A citation counts only if it parses
   into `(head)-[PREREQUISITE_OF]->(tail)`, if `(head, tail)` is a real edge with the correct
   direction, and if the recommended item is one of the two endpoints. Mentioning two endpoint
   names is not enough. Empty recommendations no longer pass vacuously; the empty-return rate and
   the list-fill rate are reported alongside.

## 8. Known deviations from a naive re-run

1. **Randomness**: seeds are fixed (`seed=42`), but `cudnn.deterministic` is not enabled, so
   bit-wise reproducibility across hardware is not guaranteed; single-run AUC varies by about
   ±0.001. Significance is judged by paired tests over 5 folds / 3 seeds, which are insensitive
   to that jitter.
2. **Single-GPU serial execution**: all reported numbers were produced serially on one
   Tesla V100-PCIE-32GB (parallel runs hit CUDA OOM; see the GKT note in `run_bench.sh`).
3. **LLM non-determinism**: GES calls DeepSeek with `temperature=0` and a disk cache. A cache miss
   may produce differently worded but structurally identical output; the reported metrics are
   stable, the exact strings are not.
4. **Graph scope**: the main protocol induces the prerequisite graph from all sequences
   (`--graph-scope all`). Re-inducing it from the evaluation prefix only (`cut`) is the
   de-leakage audit in Table 13(c). Using the wrong scope is the most common cause of a
   mismatch with the shipped reference values.

## 9. Citation

```bibtex
@misc{ren2026kg2llm,
  title        = {KG2-LLM: Learning Tracking and Resource Recommendation
                  Fusing Dual Knowledge Graphs and Large Language Models},
  author       = {Ren, Zhisen and Shao, Xinxin and Ren, Zheng},
  year         = {2026},
  howpublished = {\url{https://github.com/renzhs/KG2-LLM}}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff) (GitHub shows a "Cite this
repository" button). If you use **DNUI-IKG**, please cite this repository as well.

## 10. License

- **Code**: MIT — see [`LICENSE`](LICENSE)
- **DNUI-IKG data**: CC BY-NC 4.0 — see [`LICENSE-DATA`](LICENSE-DATA) (non-commercial research
  only, attribution required)

## 11. Contact

Zhisen Ren, School of Software, Dalian Neusoft University of Information, Dalian 116023, China
· renzhisen@neusoft.edu.cn · renzhisen@126.com

Please open an issue for bug reports, reproduction problems or data requests. Code comments are in
Chinese; `docs/` and this README are the English documentation.
