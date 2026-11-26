# Reproducibility: environment, runtimes, randomness, known deviations

## 1. Measured environment

| Item | Value |
|---|---|
| OS | Ubuntu (Linux x86_64) |
| Python | 3.12.3 |
| PyTorch | 2.4.1+cu121 |
| NumPy / SciPy | 2.5.3 / 1.18.1 |
| GPU | one Tesla V100-PCIE-32GB |
| Concurrency | 1 (single process, serial; see §4) |

`requirements.txt` only pins lower bounds (`torch>=2.0`) and does not freeze patch versions: every
conclusion in the paper rests on paired statistics over 5 folds / 3 seeds, which are insensitive to
run-to-run jitter of about ±0.001.

## 2. Measured runtimes (single V100, order of magnitude only)

| Step | Command | Time |
|---|---|---|
| Data preparation (DNUI-IKG) | `bash scripts/00_prepare_data.sh` | **1.5 s** |
| Smoke (11 models × 8 epochs) | `bash scripts/01_kt_smoke.sh` | **171 s** |
| Recommendation + CSG attribution | `bash scripts/02_downstream.sh rec` | **< 60 s** |
| Rebuild Tables 1–4 and diff | `bash scripts/03_tables.sh` | **≈ 10 s** |
| Full main tables (tuning + 5 folds × 11 models) | `bash scripts/04_full_bench.sh` | several days |

The main tables are dominated by chain tuning and baseline tuning on ASSIST12 (2.5 M interactions)
and ASSIST15 (680 k); single training runs on other datasets take minutes. `run_bench.sh` is
resumable (completed stages and per-fold results are reused); run it inside tmux/nohup.

## 3. Randomness

- `run_kt.py` sets `torch.manual_seed`, `np.random.seed` and `random.seed` before every run,
  `--seed 42` by default; the self-built splits additionally use seeds 42/43/44.
- `torch.backends.cudnn.deterministic` is **not** enabled: results reproduce to ~1e-3 on the same
  card and driver, but not bit-wise across GPU models or drivers. Significance is judged by paired
  tests and is unaffected.
- Splits: public benchmarks use the official AKT 5-fold CSVs (fixed files, no randomness); the
  self-built splits are random per-user splits with seed 42 hard-coded in `prep_datasets.py`.
- Semi-synthetic sequences: `numpy.random.default_rng(42)` — same CSV plus same script reproduces
  bit-wise (measured: 2 147 sequences / 117 865 interactions, consistent with Table 1).

## 4. Why serial on a single GPU

Concurrent runs exhaust memory and trigger `CUDA error: unknown error`. After an OOM the run does
**not** crash: some models simply never get written to the result file, producing the silent
failure mode "file present, matrix incomplete" (measured: GKT affected in 16 of 22 dataset × fold
cells). Therefore:

- `scripts/04_full_bench.sh` defaults to `PAR=1`;
- `run_bench.sh` embeds `check_bench.py` / `check_basetune.py` / `check_maincfg.py` and **refuses to
  aggregate** an incomplete matrix. Raise `PAR` if you need speed, but do not bypass the gates.

## 5. Reproducing the LLM (GES) numbers

- Calls `deepseek-chat` with `temperature=0` and a **disk cache** (`logs/ges_cache/`, keyed by a
  hash of the request payload).
- This repository contains **no** API key. Without `DEEPSEEK_API_KEY` and without a cache hit,
  `--task ges` reports the reason and exits.
- The GES numbers in the paper are archived in `results/ges_dnui_syn.json` (including per-sample
  outputs); a re-run may produce differently worded but structurally identical text, with metrics
  stable to the third decimal.
- To match the paper without calling the API, just read `results/ges_dnui_syn.json`.

## 6. Snapshot mode (`--frozen`)

`results/` is a **submission snapshot**; its mtimes depend on clone order and carry no information.
`make_paper_tables.py` normally uses "is the multi-seed artefact newer than the main run" to decide
whether a derived artefact is stale, which fires spuriously on a snapshot. Hence
`scripts/03_tables.sh` passes `--frozen`:

- verifying the paper's numbers → `scripts/03_tables.sh` (i.e. `--frozen`);
- building tables after a real re-run → `python experiments/tools/make_paper_tables.py` without
  `--frozen`, so the guard catches stale derivatives.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `missing data/processed/xxx.pkl` | data not prepared | `bash scripts/00_prepare_data.sh` (add `--public` for benchmarks) |
| `KeyError: 'assist15_f1.pkl'` and similar | benchmark not downloaded or wrong layout | see §0 of `docs/EXPERIMENTS.md` |
| numbers in `results/` changed after a re-run | `--tag` omitted, paper values overwritten | `git checkout -- results/`; next time pass `--tag repro` |
| `CUDA out of memory` | too much concurrency or too large a batch | lower `PAR` / `--batch`, refill the gap, then let the gates pass before aggregating |
| GES says `DEEPSEEK_API_KEY` is not set | no credential and no cache hit | `export DEEPSEEK_API_KEY=...`, or use the archived result |
| every table diffs | CRLF/LF mismatch | use `diff --strip-trailing-cr`; the snapshot is normalised to LF |

## 8. Protocol flags that most often explain a mismatch

In order of likelihood:

1. **Wrong `--scope-policy`**: the paper uses `down` (Eq. 10 as a one-sided inequality: higher
   grades may reuse lower-grade resources). Rerunning with the legacy `eq` gives in-scope positive
   share 0.4649 instead of 0.7255 and CSCR@10 0.3604 instead of 0.0557 — a completely different
   reading of the same experiment. The legacy numbers are kept only in `results/variants/`.
2. **Wrong `--graph-scope`**: knowledge tracing uses `train`; recommendation and path planning use
   `all` for the main tables, with `cut` (re-induction from the evaluation prefix only) reported
   separately as the de-leakage audit in Table 13(c).
3. **`--model-cfg-file` not loaded**: the main-table numbers are those of the validation-selected
   configuration; rerunning with default hyper-parameters is systematically 0.005–0.02 lower.
4. **Incomplete folds**: aggregating after running only some folds shrinks the variance on one
   side — this is exactly what `check_bench.py` exists to prevent.
