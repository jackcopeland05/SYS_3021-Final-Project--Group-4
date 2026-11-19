# Baseline Model: Formulation 2 (Min-Max Plus Volume)

Source: `reference documents/SYS3021_LP_Formulations.pdf`, Section 4 (Formulation 2), with the shared core from Section 3.
This file is the baseline reference for anyone writing code for the SYS 3021 final project (multi-session radiation therapy treatment plan). It is a linear program (LP). Implement it exactly as written here.

## 1. Problem summary

Choose the intensity of each beamlet in each treatment session so that:
- every tumor pixel receives its required dose (strict),
- critical-structure pixels never exceed hard limits (strict),
- normal-tissue pixels stay under recommended limits where it is worth it (soft, penalized),
- the weighted combination of critical peak dose, total dose to healthy tissue, and normal-tissue overage is minimized.

## 2. Sets and indices

| Set | Index | Meaning |
|---|---|---|
| S = {1, ..., n_S} | s | Treatment session. n_S is the number of sessions. |
| P | p | One pixel: a grid cell inside the body, identified by (row, column). |
| T ⊂ P | p | Tumor pixels |
| C ⊂ P | p | Critical-structure pixels |
| N ⊂ P | p | Normal-tissue pixels |
| B | b | Beamlet: a one-pixel-wide horizontal or vertical ray, identified by entry side and row/column. One beamlet per row or column for each allowed entry side. |

T, C and N are disjoint and together make up P.

## 3. Parameters (known data)

| Symbol | Meaning | Brainstorm name |
|---|---|---|
| D_pb ≥ 0 | Dose pixel p receives from beamlet b at unit intensity (attenuation matrix). 0 if b does not pass through p. | `ATTENUATION` |
| ℓ^T | Minimum dose every tumor pixel must get in each session | `min_session_pixel_radiation` |
| L^T | Minimum cumulative tumor dose | `min_cumulative_pixel_radiation` |
| U^T | Maximum cumulative tumor dose | `max_cumulative_pixel_radiation` |
| u^C | Max per-session dose, critical pixel (hard) | `max_session_critical_structure_radiation` |
| U^C | Max cumulative dose, critical pixel (hard) | `max_cumulative_critical_structure_radiation` |
| u^N | Recommended max per-session dose, normal pixel (soft) | `max_session_tissue_recommended_radiation` |
| U^N | Recommended max cumulative dose, normal pixel (soft). Not in the brainstorm; included because the assignment requires a cumulative limit. | (none) |
| λ^e > 0 | Penalty per unit of per-session normal-tissue overage | |
| λ^E > 0 | Penalty per unit of cumulative normal-tissue overage | |
| w^pk > 0 | Weight on per-session critical peaks | |
| w^C ≥ 0 | Cost per unit of dose to a critical pixel | |
| w^N ≥ 0 | Cost per unit of dose to a normal pixel (expect w^C > w^N per pixel) | |

Attenuation: D_pb = max{0, 1 − αk} when b passes through p, where k is the number of pixels the ray crossed before reaching p, and D_pb = 0 otherwise. D is data, so its exact form does not affect linearity.

## 4. Decision variables

| Variable | Defined for | Meaning |
|---|---|---|
| x_bs ≥ 0 | b ∈ B, s ∈ S | Intensity of beamlet b in session s. The primary decision. |
| d_ps | p ∈ P, s ∈ S | Dose pixel p receives in session s. Auxiliary, fully determined by x via (D0). |
| e_ps ≥ 0 | p ∈ N, s ∈ S | Slack: amount normal pixel p exceeds u^N in session s |
| E_p ≥ 0 | p ∈ N | Slack: amount normal pixel p exceeds U^N over the whole treatment |
| z_s ≥ 0 | s ∈ S | Largest dose received by any critical pixel in session s |

## 5. Objective

```
min   w^pk * sum_{s in S} z_s
    + w^C  * sum_{p in C} sum_{s in S} d_ps
    + w^N  * sum_{p in N} sum_{s in S} d_ps
    + λ^e  * sum_{p in N} sum_{s in S} e_ps
    + λ^E  * sum_{p in N} E_p
```

1. Min-max term: worst critical dose per session, summed over sessions.
2. Volume of dose to critical tissue.
3. Volume of dose to normal tissue.
4. Per-session normal-tissue overage penalty.
5. Cumulative normal-tissue overage penalty.

## 6. Constraints

**Dose bookkeeping**
- (D0) `d_ps = sum_{b in B} D_pb * x_bs`  for all p ∈ P, s ∈ S

**Tumor region (strict)**
- (T1) `d_ps >= ℓ^T`  for all p ∈ T, s ∈ S
- (T2) `sum_{s in S} d_ps >= L^T`  for all p ∈ T (only binds when L^T > n_S * ℓ^T)
- (T3) `sum_{s in S} d_ps <= U^T`  for all p ∈ T

**Critical structures (strict, no slack)**
- (C1) `d_ps <= u^C`  for all p ∈ C, s ∈ S
- (C2) `sum_{s in S} d_ps <= U^C`  for all p ∈ C

**Normal tissue (soft, via slack)**
- (N1) `d_ps - e_ps <= u^N`  for all p ∈ N, s ∈ S
- (N2) `sum_{s in S} d_ps - E_p <= U^N`  for all p ∈ N

**Domains**
- (NN) `x_bs >= 0, e_ps >= 0, E_p >= 0`

**Min-max link**
- (M1) `z_s >= d_ps`  for all p ∈ C, s ∈ S

## 7. Modeling readings (where the assignment is silent)

These are decisions already made by the group. Do not change them without asking.
- **Beam geometry:** beamlets are horizontal or vertical lines through the grid. B holds one beamlet per row or column for each allowed entry side.
- **Pixel classification:** any red makes a tumor pixel, otherwise any yellow makes a critical pixel, otherwise normal. Critical tissue sharing a cell with tumor is therefore treated as tumor: it receives at least ℓ^T every session and is not protected by (C1) or (C2).
- **Outside-body cells:** left out of P. They get no constraints and no objective terms.
- **Cumulative normal limit:** U^N is included even though the brainstorm lists only a per-session soft limit.

## 8. Weights and special cases

- Setting w^C = w^N = 0 gives Formulation 1 (per-session min-max).
- Setting w^pk = 0 gives Formulation 4 (weighted total dose, min-sum).
- Scale: the peak term adds n_S numbers; each volume term adds |C|·n_S or |N|·n_S numbers. Choose w^C ≈ 1/|C| and w^N ≈ 1/|N| so the volume terms do not dominate.
- Practical starting point: w^pk = 1 with small volume weights as tie-breakers. The reference test used w^pk = 1, w^C = 0.5/|C|, w^N = 0.5/|N|, λ^e = λ^E = 1.
- Plan a small sensitivity analysis over the five weights; expect to be asked to justify them.

## 9. Known behavior and limitations

- **Sessions are interchangeable.** Beamlet intensities may differ from session to session (x_bs is indexed by session and no constraint links sessions). Because thresholds do not vary by session, the session-average plan is also feasible and at least as good, so any session-to-session variation in the output is arbitrary and must not be interpreted. Mathematically, `x_bs = x_b1` could be added without loss, but **the model deliberately keeps the full per-session variables** (multi-session formulation required).
- **"Potential" sessions:** choosing which sessions to use would need binary variables and would no longer be an LP. Mention under limitations.
- Large volume weights make the model behave like Formulation 4, and hot spots return.
- Hard constraints can make the model infeasible (e.g. every beamlet reaching a tumor pixel overdoses a critical pixel on the way).

## 10. Pre-solve data checks

- `n_S * ℓ^T <= U^T` is necessary for feasibility. If it fails, no plan exists.
- If `n_S * ℓ^T >= L^T`, constraint (T2) is redundant.
- To diagnose infeasibility, temporarily add slack to (C1) and (C2) and see which pixels use it.

## 11. Reference test instance (for validating an implementation)

Invented instance: 9×9 grid, 4 tumor, 9 critical and 68 normal pixels, 4 sessions, 36 beamlets entering from all four sides, α = 0.08, ℓ^T = 1, L^T = 6, U^T = 9, u^C = 1.2, U^C = 4.2, u^N = 0.9, U^N = 3.2.

Expected result for Formulation 2 with w^pk = 1, w^C = 0.5/|C|, w^N = 0.5/|N|, λ^e = λ^E = 1:
- sum of session critical peaks ≈ 2.95
- total critical dose ≈ 20.2
- normal overage = 0.00

Peak sum and overage are pinned down by the objective; total critical dose can vary across tied optimal plans, so compare the objective value rather than individual variable values.

## 12. Implementation notes

- This is a pure LP: use any LP solver (e.g. scipy `linprog`/HiGHS, PuLP, CVXPY, Gurobi).
- d_ps may be eliminated by substituting (D0), or kept as explicit variables for readability.
- z_s, e_ps and E_p are pushed to their tight values by minimization, so no extra tightness constraints are needed.
- Parameters (weights, thresholds, α, grid, session count) should be configurable, not hard-coded.
