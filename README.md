# README

A short guide to what each file does and how the pieces connect. For the exact LP, see `assumptions/model.md`. For rules aimed at coding agents, see `CLAUDE.md`.

> Status: this describes the planned structure. Update it as the code is written.

## How a run works

```
python run.py --config data/configs/baseline.json
```

Data moves in one direction through the pipeline, and no stage calls a later one:

```
config.json + anatomy.csv
        |
      io.py  ->  geometry.py  ->  model.py  ->  solve.py  ->  analysis.py  ->  io.py (save_run)
     (load)      (sets, D)       (PuLP LP)     (solve)       (metrics, plots)     (write results)
```

`run.py` strings these stages together and contains no modeling logic.

## Files

### Entry point
- **`run.py`**: command-line entry point. Takes one argument, `--config <path>`, and runs the pipeline above end to end.

### Inputs (`data/`)
- **`data/input/*.csv`**: cross-section grids, read-only. Comma-separated integers, no header. Each cell is an integer label: `0` outside body, `1` tumor, `2` critical, `3` normal. `reference_9x9.csv` is a re-created stand-in for the reference instance (4 tumor / 9 critical / 68 normal), for plausibility checks only.
- **`data/configs/*.json`**: every tunable parameter lives here, as a flat JSON object: `label` (names the results folder), `anatomy_path` (relative to the project root), `alpha`, `n_S`, the thresholds `l_T, L_T, U_T, u_C, U_C, u_N, U_N`, and the five weights `w_pk, w_C, w_N, lambda_e, lambda_E`. Weights are stored as final numbers: `w_C` and `w_N` in `baseline.json` are 0.5/|C| and 0.5/|N| for the 9x9 reference grid, and **must be recomputed by hand when the anatomy changes**. Sets and the attenuation matrix are never stored here; they are recomputed each run.

### Code (`src/rtlp/`)
- **`io.py`**: the only module that reads or writes files.
  - `load_config` reads the JSON and validates it (for example, `n_S * l_T <= U_T`, and no negative thresholds or weights).
  - `load_anatomy` reads the CSV and checks it is rectangular, uses only codes 0-3, and has at least one tumor pixel.
  - `save_run` writes the results folder.
- **`geometry.py`**: pure functions, with no files and no solver. `build_sets`, `build_beamlets` and `build_D` each build one piece; `build_problem(grid, cfg)` combines them into a `ProblemData` object holding:
  - the pixel sets: P is the body cells (code != 0) in row-major order, 0-based; `T`, `C`, `N` are integer index arrays into P,
  - the mapping between pixel index and (row, column): `pixel_rc` (|P| x 2) and `pixel_index` (R x C grid, -1 outside the body),
  - the beamlets: on an R x C grid, always 2R + 2C, stored as `(side, line)` tuples in the order left (rows 0..R-1), right, top (columns 0..C-1), bottom. A beamlet whose row or column has no body cell is kept, with zero dose,
  - the attenuation matrix `D` (dense, |P| x |B|), where `D = max(0, 1 - alpha*k)` and `k` counts body pixels crossed before the ray reaches the pixel (the first body cell hit gets 1; outside cells add no depth). The anatomy is assumed to have no gaps: along any row or column, body cells are contiguous,
  - the `Config` itself.
- **`model.py`**: takes `ProblemData` and builds the PuLP model of Formulation 2 (the objective plus the constraints D0, T1-T3, C1-C2, N1-N2, NN, M1). It is one of only two modules that import PuLP.
- **`solve.py`**: the other module that imports PuLP. Solves the model and returns a `Solution` (status, objective, beamlet intensities `x`, doses `d`, as numpy arrays). Handles infeasible and unbounded cases, and has an optional infeasibility diagnostic that adds slack to the critical-structure limits to show which pixels cause the conflict.
- **`analysis.py`**: works only on `ProblemData` and `Solution` arrays; it never sees the PuLP model. Recomputes dose as `D @ x` and compares it to the solver's `d`, checks every constraint within a tolerance, and computes metrics (per session wherever a statistic isn't cumulative): objective term breakdown, critical peaks, tumor min/max dose, normal-tissue overage. Also makes the plots (dose heatmap, per-session peaks).

### Verification
There is no test suite. Each module is verified with throwaway scripts (e.g. `D` against hand-computed values on a 3x3 grid with alpha = 0.25; small LPs with known answers).

### Outputs (`results/`)
- **`results/<YYYY-MM-DD_HHMM>_<label>/`**: created by `run.py` only, one folder per run. Contains the config and anatomy copy, solver status, objective, `x.npy`, `dose.npy`, `metrics.json`, and `plots/`. Any run can be reproduced from its folder.

### Reference material
- **`assumptions/model.md`**: the LP being implemented (Formulation 2), with the group's modeling decisions and the reference test instance.
- **`reference documents/`**: the assignment, the five candidate formulations, and the radiation therapy background. Read-only.
- **`CLAUDE.md`**: project rules and context for coding agents.
