"""Metrics and plots for a solved radiation therapy LP. Works only on ProblemData + Solution arrays.

Conventions:
- The canonical dose is dose = D @ x (|P| x n_S, P-index order). The solver's d is used only in the
  (D0) comparison. z, e and E are recomputed from this dose:
  z_s = max_C dose_ps (0 if C is empty), e = max(0, dose - u_N), E = max(0, sum_s dose - U_N).
- One absolute tolerance, TOL, for every check, every "over the limit" count and every binding test.
- A failed check never raises: each check records pass/fail and its worst violation, and
  metrics["all_checks_passed"] summarizes them.
- Metrics are plain JSON types (float/int/list/dict/None). Per-session values are lists of length
  n_S, index 0 = session 1. A metric for an empty region (C or N) is None. Unless the status is
  "Optimal", every numeric section is None and only the run information is filled in.
- Figures are built with matplotlib.figure.Figure directly (no pyplot, so no window ever opens and
  no global state). This module never writes files: io.save_run saves the returned figures.
"""

import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

TOL = 1e-6

# Label colors for the anatomy map (0 outside, 1 tumor, 2 critical, 3 normal)
_LABEL_COLORS = ["#bdbdbd", "#d62728", "#f2c200", "#a6d8a0"]
_LABEL_NAMES = ["outside body", "tumor", "critical", "normal"]


def compute_metrics(problem, solution):
    """Return (metrics, dose) for `solution` (a solve.Solution) of `problem` (a geometry.ProblemData).

    metrics is a JSON-ready nested dict; dose = D @ x as a |P| x n_S float array (NaN unless Optimal).
    """
    cfg = problem.cfg
    T, C, N = problem.T, problem.C, problem.N
    n_S = cfg.n_S

    def num(v):
        """Plain float for JSON (NaN/inf -> None)."""
        v = float(v)
        return v if np.isfinite(v) else None

    def nums(arr):
        return [num(v) for v in np.asarray(arr, dtype=float).ravel()]

    metrics = {
        "run": {
            "status": solution.status,
            "solver": solution.solver,
            "solve_time": num(solution.solve_time),
            "sizes": {
                "P": int(len(problem.pixel_rc)), "T": int(len(T)), "C": int(len(C)),
                "N": int(len(N)), "B": int(len(problem.beamlets)), "n_S": int(n_S),
            },
        },
        "all_checks_passed": None,
        "objective": None,
        "solver_vs_recomputed": None,
        "tumor": None,
        "critical": None,
        "normal": None,
        "mean_dose": None,
        "beam_intensity_per_session": None,
        "checks": None,
        "binding": None,
    }

    if solution.status != "Optimal":
        return metrics, np.full((len(problem.pixel_rc), n_S), np.nan)

    # --- Canonical dose and the recomputed z, e, E ---
    x = solution.x
    dose = problem.D @ x                       # |P| x n_S
    cum = dose.sum(axis=1)                     # |P|
    dT, dC, dN = dose[T], dose[C], dose[N]
    cT, cC, cN = cum[T], cum[C], cum[N]

    z_re = dC.max(axis=0) if len(C) else np.zeros(n_S)
    e_re = np.maximum(0.0, dN - cfg.u_N)       # |N| x n_S
    E_re = np.maximum(0.0, cN - cfg.U_N)       # |N|

    # --- Objective: the five terms recomputed from the canonical dose ---
    terms = {
        "peak_sum": cfg.w_pk * z_re.sum(),
        "critical_volume": cfg.w_C * dC.sum(),
        "normal_volume": cfg.w_N * dN.sum(),
        "session_overage": cfg.lambda_e * e_re.sum(),
        "cumulative_overage": cfg.lambda_E * E_re.sum(),
    }
    obj_re = sum(terms.values())
    metrics["objective"] = {
        "solver": num(solution.objective),
        "recomputed": num(obj_re),
        "terms_weighted": {k: num(v) for k, v in terms.items()},
        "terms_unweighted": {
            "peak_sum": num(z_re.sum()),
            "critical_volume": num(dC.sum()),
            "normal_volume": num(dN.sum()),
            "session_overage": num(e_re.sum()),
            "cumulative_overage": num(E_re.sum()),
        },
    }

    # --- Solver z, e, E vs recomputed: solver >= recomputed always; equal where the weight is > 0 ---
    checks = {}

    def tight_check(name, solver_vals, recomputed_vals, need_equal):
        diff = np.asarray(solver_vals, dtype=float) - np.asarray(recomputed_vals, dtype=float)
        if diff.size == 0:
            checks[name] = {"passed": True, "worst_violation": 0.0, "equality_required": need_equal}
            return
        worst = max(0.0, float((-diff).max()))                 # solver below recomputed
        if need_equal:
            worst = max(worst, float(np.abs(diff).max()))
        checks[name] = {"passed": bool(worst <= TOL), "worst_violation": worst, "equality_required": need_equal}

    tight_check("z_tight", solution.z, z_re, cfg.w_pk > 0 and len(C) > 0)
    tight_check("e_tight", solution.e, e_re, cfg.lambda_e > 0)
    tight_check("E_tight", solution.E, E_re, cfg.lambda_E > 0)
    metrics["solver_vs_recomputed"] = {
        "z_solver": nums(solution.z),
        "z_recomputed": nums(z_re),
        "e_solver_sum": num(solution.e.sum()),
        "e_recomputed_sum": num(e_re.sum()),
        "E_solver_sum": num(solution.E.sum()),
        "E_recomputed_sum": num(E_re.sum()),
    }

    # --- Region statistics ---
    metrics["tumor"] = {
        "min_per_session": nums(dT.min(axis=0)),
        "max_per_session": nums(dT.max(axis=0)),
        "min_cumulative": num(cT.min()),
        "max_cumulative": num(cT.max()),
    }

    if len(C):
        metrics["critical"] = {
            "peak_per_session": nums(z_re),                     # z_s
            "peak_sum": num(z_re.sum()),
            "max_cumulative": num(cC.max()),
            "total_dose": num(dC.sum()),
            "slack_u_C_per_session": nums(cfg.u_C - z_re),
            "slack_U_C": num(cfg.U_C - cC.max()),
        }

    if len(N):
        over_s = dN > cfg.u_N + TOL                              # |N| x n_S
        over_c = cN > cfg.U_N + TOL                              # |N|
        offenders_s = []
        for i, s in np.argwhere(over_s):
            r, c = problem.pixel_rc[N[i]]
            offenders_s.append({"row": int(r), "col": int(c), "session": int(s), "overage": num(e_re[i, s])})
        offenders_c = []
        for i in np.flatnonzero(over_c):
            r, c = problem.pixel_rc[N[i]]
            offenders_c.append({"row": int(r), "col": int(c), "overage": num(E_re[i])})
        metrics["normal"] = {
            "max_per_session": nums(dN.max(axis=0)),
            "max_cumulative": num(cN.max()),
            "over_u_N_count_per_session": [int(v) for v in over_s.sum(axis=0)],
            "over_u_N_total_per_session": nums(e_re.sum(axis=0)),
            "over_u_N_max_per_session": nums(e_re.max(axis=0)),
            "over_U_N_count": int(over_c.sum()),
            "over_U_N_total": num(E_re.sum()),
            "over_U_N_max": num(E_re.max()),
            "pixels_over_u_N": offenders_s,                      # session is 0-based
            "pixels_over_U_N": offenders_c,
        }

    metrics["mean_dose"] = {
        region: (None if len(idx) == 0 else {
            "per_session": nums(dose[idx].mean(axis=0)),
            "cumulative": num(cum[idx].mean()),
        })
        for region, idx in (("tumor", T), ("critical", C), ("normal", N))
    }
    metrics["beam_intensity_per_session"] = nums(x.sum(axis=0))

    # --- Independent constraint checks: worst violation per tag (positive = violated) ---
    def viol_check(name, violations):
        v = np.asarray(violations, dtype=float).ravel()
        worst = max(0.0, float(v.max())) if v.size else 0.0
        checks[name] = {"passed": bool(worst <= TOL), "worst_violation": worst, "rows": int(v.size)}

    viol_check("D0", np.abs(solution.d - dose))
    viol_check("T1", cfg.l_T - dT)
    viol_check("T2", cfg.L_T - cT)
    viol_check("T3", cT - cfg.U_T)
    viol_check("C1", dC - cfg.u_C)
    viol_check("C2", cC - cfg.U_C)
    viol_check("N1", dN - solution.e - cfg.u_N)
    viol_check("N2", cN - solution.E - cfg.U_N)
    viol_check("NN", np.concatenate([-x.ravel(), -solution.e.ravel(), -solution.E.ravel()]))
    viol_check("M1", dC - solution.z[None, :])
    obj_gap = abs(obj_re - solution.objective)
    checks["objective_match"] = {
        "passed": bool(obj_gap <= TOL * max(1.0, abs(solution.objective))),
        "worst_violation": float(obj_gap),
        "rows": 1,
    }
    metrics["checks"] = checks
    metrics["all_checks_passed"] = bool(all(c["passed"] for c in checks.values()))

    # --- Binding constraints per tag (slack <= TOL). N1/N2 slack is on the dose itself, ignoring e/E ---
    def n_binding(slack):
        return int((np.asarray(slack, dtype=float) <= TOL).sum())

    metrics["binding"] = {
        "T1": n_binding(dT - cfg.l_T),
        "T2": n_binding(cT - cfg.L_T),
        "T3": n_binding(cfg.U_T - cT),
        "C1": n_binding(cfg.u_C - dC),
        "C2": n_binding(cfg.U_C - cC),
        "N1": n_binding(cfg.u_N - dN),
        "N2": n_binding(cfg.U_N - cN),
        "NN_x_zero": n_binding(x),
        "M1": n_binding(solution.z[None, :] - dC),
    }

    return metrics, dose


def make_plots(problem, solution):
    """Return {name: Figure}: "anatomy" always; "dose_cumulative" and "critical_peaks" when Optimal
    ("critical_peaks" only if C is non-empty)."""
    cfg = problem.cfg
    grid = problem.grid
    n_rows, n_cols = grid.shape
    annotate = n_rows <= 15 and n_cols <= 15
    figs = {}

    def outline(ax, mask, color):
        """Draw the border of the cells where `mask` is True."""
        for r, c in np.argwhere(mask):
            for dr, dc, seg in (
                (-1, 0, ([c - 0.5, c + 0.5], [r - 0.5, r - 0.5])),
                (1, 0, ([c - 0.5, c + 0.5], [r + 0.5, r + 0.5])),
                (0, -1, ([c - 0.5, c - 0.5], [r - 0.5, r + 0.5])),
                (0, 1, ([c + 0.5, c + 0.5], [r - 0.5, r + 0.5])),
            ):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < n_rows and 0 <= cc < n_cols) or not mask[rr, cc]:
                    ax.plot(*seg, color=color, linewidth=2.5, solid_capstyle="round")

    def grid_axes(ax, title):
        ax.set_title(title)
        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_rows))
        ax.set_xlabel("column")
        ax.set_ylabel("row")

    # --- Anatomy label map ---
    fig = Figure(figsize=(6, 5.5), layout="constrained")
    ax = fig.add_subplot()
    ax.imshow(grid, cmap=ListedColormap(_LABEL_COLORS), vmin=-0.5, vmax=3.5)
    grid_axes(ax, "Anatomy")
    present = sorted(set(np.unique(grid).tolist()))
    ax.legend(
        handles=[Patch(color=_LABEL_COLORS[k], label=_LABEL_NAMES[k]) for k in present],
        loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False,
    )
    figs["anatomy"] = fig

    if solution.status != "Optimal":
        return figs

    dose = problem.D @ solution.x
    cum = dose.sum(axis=1)

    # --- Cumulative dose heatmap with tumor/critical outlines ---
    cum_grid = np.full(grid.shape, np.nan)
    cum_grid[problem.pixel_rc[:, 0], problem.pixel_rc[:, 1]] = cum
    cmap = matplotlib.colormaps["viridis"].with_extremes(bad=_LABEL_COLORS[0])
    fig = Figure(figsize=(6.5, 5.5), layout="constrained")
    ax = fig.add_subplot()
    im = ax.imshow(np.ma.masked_invalid(cum_grid), cmap=cmap)
    fig.colorbar(im, ax=ax, label="cumulative dose (unit intensity)")
    outline(ax, grid == 1, "#d62728")
    outline(ax, grid == 2, "#f2c200")
    if annotate:
        vmin, vmax = np.nanmin(cum_grid), np.nanmax(cum_grid)
        mid = (vmin + vmax) / 2
        for r, c in problem.pixel_rc:
            v = cum_grid[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if v > mid else "white")
    grid_axes(ax, f"Cumulative dose over {cfg.n_S} sessions")
    figs["dose_cumulative"] = fig

    # --- Per-session critical peaks z_s (recomputed) with the u_C limit ---
    if len(problem.C):
        z = dose[problem.C].max(axis=0)
        sessions = np.arange(1, cfg.n_S + 1)
        fig = Figure(figsize=(6, 4), layout="constrained")
        ax = fig.add_subplot()
        ax.bar(sessions, z, color="#4c72b0", label="peak critical dose $z_s$")
        ax.axhline(cfg.u_C, color="#d62728", linestyle="--", label=f"$u^C$ = {cfg.u_C:g}")
        for s, v in zip(sessions, z):
            ax.text(s, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(sessions)
        ax.set_xlabel("session")
        ax.set_ylabel("dose (unit intensity)")
        ax.set_ylim(0, max(cfg.u_C, float(z.max())) * 1.35)
        ax.set_title("Peak critical dose per session")
        ax.legend(loc="upper left", ncols=2, frameon=False)
        figs["critical_peaks"] = fig

    return figs
