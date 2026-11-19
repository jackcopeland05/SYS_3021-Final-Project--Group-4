"""Build the PuLP model of Formulation 2 (min-max plus volume) from a ProblemData.

A direct transcription of assumptions/model.md: objective (5 terms) and constraints
D0, T1, T2, T3, C1, C2, N1, N2, NN, M1. No solver is called here.

Conventions (0-based throughout):
- Every variable is stored in a numpy object array in LPModel, and values must be read back
  by array slot (x[b, s], d[p, s], ...), never by parsing names. Names are for humans only.
- x[b, s]: beamlet b = index into problem.beamlets, session s.
- d[p, s]: pixel p = index into P (problem.pixel_rc[p] gives its (row, col)).
- z[s]: session s.
- e[i, s] and E[i]: row i belongs to normal pixel p = problem.N[i]. Their names use the global
  pixel index p (e.g. "e_p37_s2"), not i.
- Constraint names start with their model.md tag and use global pixel indices,
  e.g. "C1_p37_s2" is (C1) for pixel 37 in session 2.
- (NN) is enforced through variable bounds (lowBound=0 on x, e, E), so no NN_* rows exist.
- d is declared with no bounds: (D0) fixes it exactly, and D >= 0, x >= 0 already imply d >= 0.
- There is no slack on the hard constraints (T1-T3, C1, C2). An infeasible instance is reported
  as infeasible by the solver; that status is itself the result.
"""

from dataclasses import dataclass

import numpy as np
import pulp


@dataclass(frozen=True)
class LPModel:
    """The PuLP problem plus every decision variable, laid out as numpy object arrays."""

    prob: pulp.LpProblem
    x: np.ndarray  # |B| x n_S, beamlet intensity
    d: np.ndarray  # |P| x n_S, pixel dose per session
    z: np.ndarray  # n_S, largest critical dose per session
    e: np.ndarray  # |N| x n_S, per-session normal overage; row i <-> pixel problem.N[i]
    E: np.ndarray  # |N|, cumulative normal overage; row i <-> pixel problem.N[i]


def build_model(problem):
    """Return an LPModel of Formulation 2 for `problem` (a geometry.ProblemData)."""
    cfg = problem.cfg
    D = problem.D
    n_P, n_B = D.shape
    n_S = cfg.n_S
    T, C, N = problem.T, problem.C, problem.N
    sessions = range(n_S)

    prob = pulp.LpProblem("rt_formulation2", pulp.LpMinimize)

    # --- Decision variables (NN: x, e, E >= 0 via lowBound; z >= 0 per model.md; d unbounded) ---
    x = np.empty((n_B, n_S), dtype=object)
    for b in range(n_B):
        for s in sessions:
            x[b, s] = prob.add_variable(f"x_b{b}_s{s}", lowBound=0)

    d = np.empty((n_P, n_S), dtype=object)
    for p in range(n_P):
        for s in sessions:
            d[p, s] = prob.add_variable(f"d_p{p}_s{s}")

    z = np.empty(n_S, dtype=object)
    for s in sessions:
        z[s] = prob.add_variable(f"z_s{s}", lowBound=0)

    e = np.empty((len(N), n_S), dtype=object)
    E = np.empty(len(N), dtype=object)
    for i, p in enumerate(N):
        for s in sessions:
            e[i, s] = prob.add_variable(f"e_p{p}_s{s}", lowBound=0)
        E[i] = prob.add_variable(f"E_p{p}", lowBound=0)

    # --- Objective ---
    prob.setObjective(
        cfg.w_pk * pulp.lpSum(z[s] for s in sessions)
        + cfg.w_C * pulp.lpSum(d[p, s] for p in C for s in sessions)
        + cfg.w_N * pulp.lpSum(d[p, s] for p in N for s in sessions)
        + cfg.lambda_e * pulp.lpSum(e[i, s] for i in range(len(N)) for s in sessions)
        + cfg.lambda_E * pulp.lpSum(E[i] for i in range(len(N)))
    )

    # --- (D0) dose bookkeeping: d_ps = sum_b D_pb x_bs (only nonzero D_pb terms) ---
    for p in range(n_P):
        hits = np.flatnonzero(D[p] > 0)
        for s in sessions:
            prob.addConstraint(
                d[p, s] == pulp.lpSum(float(D[p, b]) * x[b, s] for b in hits),
                f"D0_p{p}_s{s}",
            )

    # --- Tumor (strict): T1 per session, T2/T3 cumulative. T2 is always added, even if redundant ---
    for p in T:
        for s in sessions:
            prob.addConstraint(d[p, s] >= cfg.l_T, f"T1_p{p}_s{s}")
        prob.addConstraint(pulp.lpSum(d[p, s] for s in sessions) >= cfg.L_T, f"T2_p{p}")
        prob.addConstraint(pulp.lpSum(d[p, s] for s in sessions) <= cfg.U_T, f"T3_p{p}")

    # --- Critical (strict, no slack): C1 per session, C2 cumulative; M1 min-max link ---
    for p in C:
        for s in sessions:
            prob.addConstraint(d[p, s] <= cfg.u_C, f"C1_p{p}_s{s}")
            prob.addConstraint(z[s] >= d[p, s], f"M1_p{p}_s{s}")
        prob.addConstraint(pulp.lpSum(d[p, s] for s in sessions) <= cfg.U_C, f"C2_p{p}")

    # --- Normal (soft): N1 per session, N2 cumulative, with overage slacks e, E ---
    for i, p in enumerate(N):
        for s in sessions:
            prob.addConstraint(d[p, s] - e[i, s] <= cfg.u_N, f"N1_p{p}_s{s}")
        prob.addConstraint(pulp.lpSum(d[p, s] for s in sessions) - E[i] <= cfg.U_N, f"N2_p{p}")

    return LPModel(prob=prob, x=x, d=d, z=z, e=e, E=E)
