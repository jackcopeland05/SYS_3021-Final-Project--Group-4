# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

SYS 3021 (Decision-Driven Modeling) final project: an optimal **multi-session radiation therapy treatment plan**, formulated as a **deterministic linear program (LP)** and implemented in Python. The patient is a 2D cross-section on an n x n pixel grid. Radiation is delivered by **beamlets** (one-pixel-wide straight rays, horizontal or vertical only, never diagonal) over several treatment sessions. The decision is the intensity of every beamlet in every session.

Deliverables (this repo covers the code only; the <=12-slide deck is handled separately by the team):
- Python code that builds and solves the LP for a provided cross-section and reports metrics on the answer.
- The instructor will later supply a sample cross-section and actual thresholds. Until then the team picks placeholder values, so nothing may be hard-coded.

## Source of truth

- `assumptions/model.md` is the exact LP (**Formulation 2: min-max plus volume**): sets, parameters, variables, objective, constraints (D0, T1-T3, C1-C2, N1-N2, NN, M1), reference instance. **Implement it exactly as written.** Keep constraint names matching those tags.
- `reference documents/` (read-only): the assignment PDF, `SYS3021_LP_Formulations.pdf` (five candidate formulations; only #2 is used), `Radiation_Therapy_Overview.pdf` (clinical background for the assumptions).
- The "Modeling readings" in `model.md` section 7 are group decisions. **Do not change them without asking the user.**
- **Only Formulation 2 is built.** Formulations 1 and 4 are reached by setting weights (w^C = w^N = 0, or w^pk = 0). Do NOT implement Formulation 3 (two-phase), 5 (recovery-aware) or a formulation dispatcher.

## Hard constraints from the course

- **Modeling library is PuLP. This is set by the professor and cannot change.** Do not introduce scipy `linprog`, Pyomo, CVXPY, Gurobi, etc. as modeling layers. For the underlying LP backend that PuLP calls, use whichever is fastest (prefer HiGHS through PuLP; fall back to CBC if unavailable).
- The model must stay a pure LP (no binary/integer variables). Choosing *which* sessions to use would need binaries and is only a stated limitation.
- The model is multi-session: keep the full x_bs, d_ps variables. Do **not** collapse sessions with `x_bs = x_b1`. (Sessions are interchangeable under constant thresholds; per-session variation in the output is arbitrary and must not be interpreted.)

## Input and configuration

- **Anatomy input:** a CSV of integer labels, one cell per pixel: `0` outside body, `1` tumor, `2` critical, `3` normal. Each cell has exactly one label, so the "red beats yellow beats normal" rule is applied upstream; the code never resolves overlaps. Code 0 cells are excluded from P (no constraints, no objective terms).
- **Config:** a single flat JSON file holds every parameter: label (names the results folder), anatomy_path (relative to the project root; `run.py` is always run from the root), alpha, n_S, l_T, L_T, U_T, u_C, U_C, u_N, U_N, the five weights (w_pk, w_C, w_N, lambda_e, lambda_E). No description/comment fields; no solver key. `run.py` takes **only** `--config <path>` via argparse (no per-parameter flags).
- Sets and D are never stored in config. They are always recomputed from the CSV and config.
- Config validation lives in one place and fails loudly, e.g. n_S * l_T <= U_T (otherwise no plan exists). If n_S * l_T >= L_T, (T2) is redundant.

## Geometry rules (D matrix)

- Grids may be R x C (not only square). Beamlets enter from **all four sides, always**: 2R + 2C beamlets (36 for 9x9). There is no config option to restrict sides. A beamlet whose row/column has no body cell is kept, with an all-zero column in D.
- `D_pb = max{0, 1 - alpha*k}` if beamlet b passes through p, else 0, where k is the number of body pixels the ray crossed before reaching p (the first body cell hit gets D = 1).
- **k starts counting at the body's edge** (the first non-zero cell the ray meets), not at the grid edge. Outside-body cells add no depth.
- **No gaps:** a row/column never goes body -> outside -> body (once a ray is in tissue, it only meets outside cells after leaving the body for good). This is assumed, not checked.
- Index conventions (0-based): P = body cells in row-major order; `ProblemData.pixel_rc` (|P| x 2, (row, col)) and `pixel_index` (R x C, -1 outside) map both ways; T, C, N are int index arrays into P (C or N may be empty). Beamlet order: ("left", rows 0..R-1), ("right", rows), ("top", cols 0..C-1), ("bottom", cols), stored as `(side, line)` tuples. D is a dense |P| x |B| numpy array.
- geometry.py raises no ValueErrors on config values (weights are not checked against 0.5/|C|; unreachable tumor pixels surface as infeasibility). Config values raise only in `load_config`, and only for negatives / structural problems.

## Architecture (data flows one way; no module calls a later stage)

```
io -> geometry -> model -> solve -> analysis -> io.save_run
```

```
Final Project/
├── data/input/        sample CSVs (read-only inputs)
├── data/configs/        baseline.json
├── src/rtlp/
│   ├── io.py            ALL file I/O: load_config (validate), load_anatomy (rectangular, codes 0-3 only, >=1 tumor pixel), save_run
│   ├── geometry.py      pure functions, no files/solver: build_sets, build_beamlets, build_D, build_problem(grid, cfg) -> ProblemData
│   ├── model.py         build_model(ProblemData) -> LPModel(prob, x, d, z, e, E): PuLP model of Formulation 2
│   ├── solve.py         solve(LPModel) -> Solution(status, objective, x, d, z, e, E as numpy arrays); handles
│   │                    infeasible/unbounded. An infeasible status is itself the result: no slack is ever added
│   │                    to the hard constraints (T1-T3, C1, C2) to force feasibility.
│   └── analysis.py      works only on ProblemData + Solution arrays (no model object)
├── run.py               thin CLI, no modeling logic (adds src/ to sys.path, then imports rtlp)
├── environment.yml      conda env sys_3021
└── results/<YYYY-MM-DD_HHMM>_<label>/   written only by run.py
```

Design rules:
- Only `io.py` touches the filesystem. Only `model.py` and `solve.py` import PuLP.
- Every run is reproducible from its results folder (config + anatomy copy, status, objective, `x.npy`, `dose.npy`, `metrics.json`, `plots/`).
- Pipeline in `run.py`: load_config -> load_anatomy -> build_problem -> build_model -> solve -> compute_metrics -> save_run.
- tqdm is used for progress display only, and only in `solve.py` (a live elapsed-time counter while the solver runs; one solve call reports no progress, so there is no percentage). A user decision outside `model.md`: do not remove it or spread it to other modules without asking.
- Code style (user preference): fewer, larger functions with their validation checks written inline (raise ValueError naming the bad key/row), not many tiny helpers. No pytest suite in `tests/`; verify with throwaway scripts instead.

## Metrics (analysis.py)

Record **every non-cumulative statistic for every session**, not just in aggregate. Standard output:
- Objective value, with each of the five terms broken out (peak sum, critical volume, normal volume, per-session overage, cumulative overage).
- Solver status (an infeasible status is reported as is).
- z, e, E from the solver are tight only when their weight is > 0 (w_pk, lambda_e, lambda_E; z also needs C non-empty). Report both the solver's values and the values recomputed from d (z_s = max_C d_ps, e = max(0, d - u_N), E = max(0, sum_s d - U_N)); metrics use the recomputed ones, and the tolerance check requires solver >= recomputed, with equality only where the weight is > 0.
- Per-session critical peaks (z_s); tumor min/max dose (per session and cumulative); normal-tissue overage (per session and cumulative, count and size of pixels over limit).
- Independent check: recompute dose = D @ x, compare with the solver's d, and tolerance-check every constraint.
- Plots: dose heatmap over the anatomy, per-session peaks.

## Weights and validation notes

- Start with w_pk = 1, w_C = 0.5/|C|, w_N = 0.5/|N|, lambda_e = lambda_E = 1 (volume terms act as tie-breakers). Weights are stored in the config as final numbers (baseline: w_C = 0.0556 = 0.5/9, w_N = 0.007353 = 0.5/68, sized for `data/input/reference_9x9.csv`); they **must be recomputed by hand when the anatomy changes**. Plan a small sensitivity analysis over the five weights; the team must be able to justify them.
- Large volume weights make the model behave like Formulation 4 (hot spots return).
- Hard constraints can make the model infeasible (e.g. every beamlet reaching a tumor pixel overdoses a critical pixel).
- **Reference instance** (`model.md` section 11): 9x9 grid, 4 tumor / 9 critical / 68 normal, 4 sessions, alpha = 0.08, l_T = 1, L_T = 6, U_T = 9, u_C = 1.2, U_C = 4.2, u_N = 0.9, U_N = 3.2. Expected: sum of critical peaks ~ 2.95, total critical dose ~ 20.2, normal overage 0.00. The actual grid CSV is not available, so a re-created grid is a plausibility check only. Compare objective value, not individual variable values (ties are arbitrary).

## Commands

Conda environment `sys_3021` (Python 3.14, numpy, pulp, highspy, matplotlib, tqdm), defined in `environment.yml`.

```
conda env create -f environment.yml     # first time only
conda activate sys_3021
python run.py --config data/configs/baseline.json
```

Status: the pipeline is complete. `src/rtlp/io.py` has `Config`, `load_config`, `load_anatomy`, `save_run`. `src/rtlp/geometry.py` is done (`ProblemData`, `build_sets`, `build_beamlets`, `build_D`, `build_problem`). `src/rtlp/model.py` is done (`LPModel`, `build_model`). `src/rtlp/solve.py` is done (`Solution`, `solve`). `src/rtlp/analysis.py` is done (`TOL`, `compute_metrics`, `make_plots`, `format_summary`). `run.py` is done. Update this section as the code lands.

## model.py conventions

- `LPModel` holds the variables as numpy object arrays: `x` (|B| x n_S), `d` (|P| x n_S), `z` (n_S), `e` (|N| x n_S), `E` (|N|). Row i of `e`/`E` is pixel `problem.N[i]`. Read values by array slot, never by parsing names; names (`x_b{b}_s{s}`, `e_p{p}_s{s}` with global pixel p, constraints `C1_p{p}_s{s}` etc.) are for humans.
- `d` is declared free (no bounds): (D0) fixes it, and D >= 0, x >= 0 already imply d >= 0. (NN) is enforced by `lowBound=0` on x, e, E (no NN_* rows); z >= 0. (T2) is always added, even when redundant.
- **PuLP is 4.x (Rust core); its API differs from PuLP 2.x tutorials:** create variables with `prob.add_variable(name, lowBound=...)` (`pulp.LpVariable(...)` no longer works), add constraints with `prob.addConstraint(expr, name)`, set the objective with `prob.setObjective(expr)`. `prob.constraints()` is a method returning a list. Missing bounds read back as `-inf`/`inf`, not `None`. `prob.solve(pulp.HiGHS(msg=False))` returns an `LpSolveStats` (`.status` is an `LpSolveStatus` enum, e.g. `.status.name == "Optimal"`, plus `.objective`); there is no `pulp.LpStatus` and no `prob.status`. Variable values: `v.value()`.
- Verified 2026-11-19 with a throwaway script: hand-solved 3x3 instance (objective 2.1 exact), an infeasible instance, name/slot alignment, per-tag constraint counts. On `reference_9x9.csv` the optimum is 3.5727 (peak sum 2.60, critical total 7.5, overage 0), confirmed by HiGHS and CBC and by an independent numpy check of D and every constraint. This **does not match** the published 2.95 / 20.2: a plan with those numbers would have objective >= 4.07 on our grid, so the re-created grid differs from the original. The benchmark was not changed.

## solve.py conventions

- Backend: `pulp.HiGHS(msg=False)` if `.available()`, else `pulp.COIN_CMD(msg=False)` (CBC; there is no `PULP_CBC_CMD` in PuLP 4.x); `RuntimeError` only if neither exists. No time limit. HiGHS still prints a two-line banner despite `msg=False` (PuLP 4 behavior).
- `Solution(status, objective, x, d, z, e, E, solver, solve_time)`: `status` is the plain enum name string (`"Optimal"`, `"Infeasible"`, ...); arrays are float64 with LPModel shapes, raw solver values (no clipping). Unless status is `"Optimal"`, `objective` and every array are NaN (same shapes). No status ever raises.
- Verified 2026-11-19: baseline gives Optimal 3.5727 with both HiGHS and CBC, max |d - D@x| ~ 2e-16; L_T > U_T gives `"Infeasible"` with NaN arrays.

## analysis.py conventions

- `compute_metrics(problem, solution) -> (metrics, dose)`; `make_plots(problem, solution) -> {name: Figure}`; `format_summary(metrics) -> str` (the text run.py prints and save_run writes to `summary.txt`; infeasible runs get a short version). analysis never writes files: `io.save_run` dumps `metrics` to `metrics.json`, `dose` to `dose.npy` (|P| x n_S, P-index order) and calls `fig.savefig` for each figure into `plots/<name>.png`.
- Canonical dose is `D @ x`; the solver's d is used only in the (D0) check. z, e, E are recomputed from it (z_s = 0 when C is empty).
- One absolute tolerance `TOL = 1e-6` for every check, every over-limit count and every binding test. Failed checks never raise: each tag in `metrics["checks"]` has `passed` and `worst_violation`, summarized by `metrics["all_checks_passed"]`. The objective check is relative: |recomputed - solver| <= TOL * max(1, |solver|).
- `metrics` is plain JSON types only (no NaN; use `json.dumps(..., allow_nan=False)`). Per-session values are lists of length n_S (index 0 = session 1); the offending-pixel lists use 0-based sessions. An empty C or N gives `None` for that region's metrics. Unless the status is "Optimal", only `metrics["run"]` is filled in, every other section is `None`, and `dose` is NaN.
- Binding counts: slack <= TOL per tag; N1/N2 slack is measured on the dose itself (u_N - d), ignoring e/E, so "binding" means at or over the recommended limit.
- Plots: `anatomy` (always), `dose_cumulative` (viridis, outside body grey, tumor red / critical yellow outlines, cell values when the grid is <= 15x15) and `critical_peaks` (recomputed z_s bars with a dashed u_C line, only if C is non-empty). No per-session heatmaps: per-session variation is arbitrary. Built with `matplotlib.figure.Figure` directly (no pyplot).
- Verified 2026-11-19: baseline gives all checks passed, objective 3.5727 (solver and recomputed agree to 1e-15), z = [0.332, 0.468, 0.9, 0.9], no normal overage; an infeasible config gives `run` only plus the anatomy figure; empty C with u_N = 0.3, lambda_e = 0 gives `critical = None`, nonzero per-session overage counts and all checks passed.

## run.py and save_run conventions

- Project root = the folder holding `run.py` (`ROOT`); `src/` is put on `sys.path` from it, and `anatomy_path` and `results/` are resolved against it, so run.py works from any working directory. `--config` is resolved against the working directory, like any CLI argument.
- Exit codes: 0 whenever the pipeline finishes, including an infeasible status and failed checks (the summary then has a `WARNING: some checks failed` line). `ValueError`/`FileNotFoundError` from loading print `error: <message>` to stderr, exit 1 and write nothing; any other exception keeps its traceback.
- Console: the solver's tqdm counter, then the summary, then `results written to <path>`. No per-stage lines.
- `save_run(root, config_path, cfg, solution, metrics, dose, figures, summary) -> Path` writes `results/<YYYY-MM-DD_HHMM>_<label>/` (local time; suffix `_2`, `_3`, ... if it exists, never overwrite): `anatomy.csv` (byte copy), `config.json` (the original JSON, same key order, with `anatomy_path` rewritten to the root-relative path of the copied `anatomy.csv`), `x.npy`, `dose.npy`, `metrics.json` (`allow_nan=False`), `summary.txt`, `plots/<name>.png` at 150 dpi. A folder is written for every completed solve, whatever its status (arrays NaN-filled when not Optimal).
- Reproduce a run: `python run.py --config results/<run>/config.json`.
- Verified 2026-11-19: baseline exits 0 with objective 3.5727 and all checks passed; a second run in the same minute gets `_2`; L_T = 10 > U_T gives an Infeasible folder (anatomy plot only) and exit 0; alpha = -1 and a missing config give `error: ...`, exit 1 and no folder; re-running from a results folder's `config.json` from another working directory reproduces 3.5727.
