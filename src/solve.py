"""Solve an LPModel with PuLP and return the result as plain numpy arrays.

Conventions:
- Solver: HiGHS if available, otherwise CBC (pulp.COIN_CMD). Silent (msg=False), no time limit.
  The solver used is recorded in Solution.solver.
- Status is the solver's status as a plain string ("Optimal", "Infeasible", "Unbounded",
  "TimeLimit", "NumericalError", ...). No status raises: an infeasible or unbounded result is
  itself the result, and no slack is ever added to the hard constraints (T1-T3, C1, C2).
- Values are read by array slot from the LPModel arrays, never by parsing variable names, and are
  returned raw (no clipping or rounding); analysis.py does the tolerance checks.
- Unless the status is "Optimal", objective is NaN and every array is NaN-filled with the same
  shape it would have had, so later stages branch on status, never on None.
- While a single solve runs, a tqdm counter shows elapsed time (a solve call reports no progress,
  so this only shows the run is alive). solve_sessions turns that off and shows one tqdm bar over
  the number of sessions k instead.

Number of sessions: config n_S is the MAXIMUM. solve_sessions solves one pure LP per k = 1..K
(K = min(n_S, floor(U_T / l_T)) when l_T > 0, since k*l_T > U_T can never satisfy T1 and T3; K = n_S
when l_T = 0) and keeps the lowest objective, preferring fewer sessions on ties (relative SWEEP_TOL).
This picks the number of sessions without binary variables.
"""

import dataclasses
import math
import threading
from dataclasses import dataclass

import numpy as np
import pulp
from tqdm import tqdm

from model import build_model

SWEEP_TOL = 1e-6  # objectives within this relative gap count as a tie (fewer sessions wins)


@dataclass(frozen=True)
class Solution:
    """Solver outcome as numpy arrays (float64), laid out exactly like the LPModel arrays."""

    status: str  # solver status name, e.g. "Optimal", "Infeasible"
    objective: float  # stats.objective if Optimal, else NaN
    x: np.ndarray  # |B| x n_S
    d: np.ndarray  # |P| x n_S
    z: np.ndarray  # n_S
    e: np.ndarray  # |N| x n_S; row i <-> pixel problem.N[i]
    E: np.ndarray  # |N|; row i <-> pixel problem.N[i]
    solver: str  # "HiGHS" or "COIN_CMD"
    solve_time: float  # seconds, as reported by PuLP


def solve(lp_model, show_progress=True):
    """Solve `lp_model` (a model.LPModel) and return a Solution. Never raises on solver status.

    show_progress=False hides the elapsed-time counter (solve_sessions shows its own bar).
    """
    # --- Pick the backend: HiGHS preferred, CBC fallback ---
    highs = pulp.HiGHS(msg=False)
    if highs.available():
        backend, solver_name = highs, "HiGHS"
    else:
        cbc = pulp.COIN_CMD(msg=False)
        if not cbc.available():
            raise RuntimeError("No LP solver available: neither HiGHS nor CBC (COIN_CMD) is installed")
        backend, solver_name = cbc, "COIN_CMD"

    # --- Solve, with a live elapsed-time counter (no total: the solve call reports no progress) ---
    done = threading.Event()
    bar = tqdm(desc=f"Solving ({solver_name})", bar_format="{desc}: {elapsed} elapsed", leave=True,
               disable=not show_progress)

    def tick():
        while not done.wait(0.5):
            bar.refresh()

    ticker = threading.Thread(target=tick, daemon=True)
    ticker.start()
    try:
        stats = lp_model.prob.solve(backend)
    finally:
        done.set()
        ticker.join()
        bar.close()

    status = stats.status.name

    # --- Read values by array slot (NaN everywhere unless Optimal) ---
    optimal = status == "Optimal"

    def values(arr):
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        if optimal:
            for idx, var in np.ndenumerate(arr):
                v = var.value()
                out[idx] = np.nan if v is None else float(v)
        return out

    return Solution(
        status=status,
        objective=float(stats.objective) if optimal and stats.objective is not None else float("nan"),
        x=values(lp_model.x),
        d=values(lp_model.d),
        z=values(lp_model.z),
        e=values(lp_model.e),
        E=values(lp_model.E),
        solver=solver_name,
        solve_time=float(stats.time) if stats.time is not None else float("nan"),
    )


def solve_sessions(problem):
    """Choose the number of sessions: solve one LP per k = 1..K and keep the best.

    `problem` is a geometry.ProblemData whose cfg.n_S is the maximum number of sessions.
    Returns (best_problem, best_solution, sweep):
      best_problem   copy of `problem` with cfg.n_S = chosen k (D and the sets are unchanged)
      best_solution  its Solution; if no k is optimal, the last k's (non-optimal) Solution
      sweep          {"n_S_max", "K", "chosen" (None if no k is optimal),
                      "rows": [{"k", "status", "objective" (None unless Optimal)}, ...]}
    Best = lowest objective; a later k replaces the current best only if it is lower by more than
    SWEEP_TOL relative, so ties go to fewer sessions.
    """
    cfg = problem.cfg
    if cfg.l_T > 0:
        K = min(cfg.n_S, math.floor(cfg.U_T / cfg.l_T + 1e-9))
    else:
        K = cfg.n_S
    K = max(K, 1)

    rows = []
    best = None   # (problem_k, solution_k)
    last = None
    for k in tqdm(range(1, K + 1), desc="Solving sessions k", unit="k"):
        problem_k = dataclasses.replace(problem, cfg=dataclasses.replace(cfg, n_S=k))
        sol = solve(build_model(problem_k), show_progress=False)
        last = (problem_k, sol)
        optimal = sol.status == "Optimal"
        rows.append({"k": k, "status": sol.status, "objective": float(sol.objective) if optimal else None})
        if optimal:
            if best is None:
                best = last
            else:
                best_obj = best[1].objective
                if sol.objective < best_obj - SWEEP_TOL * max(1.0, abs(best_obj)):
                    best = last

    chosen_problem, chosen_solution = best if best is not None else last
    sweep = {
        "n_S_max": int(cfg.n_S),
        "K": int(K),
        "chosen": int(chosen_problem.cfg.n_S) if best is not None else None,
        "rows": rows,
    }
    return chosen_problem, chosen_solution, sweep
