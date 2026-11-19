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
- While the solver runs, a tqdm counter shows elapsed time. A single solve call reports no
  progress, so this only shows the run is alive, not how close it is to finishing.
"""

import threading
import time
from dataclasses import dataclass

import numpy as np
import pulp
from tqdm import tqdm


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


def solve(lp_model):
    """Solve `lp_model` (a model.LPModel) and return a Solution. Never raises on solver status."""
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
    bar = tqdm(desc=f"Solving ({solver_name})", bar_format="{desc}: {elapsed} elapsed", leave=True)

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
