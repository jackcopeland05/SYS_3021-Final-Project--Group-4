"""All file I/O for the radiation therapy LP: loading/validating inputs and (later) saving runs.

This is the only module that touches the filesystem.
"""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    """Every run parameter, read once from the config JSON and then read-only.

    Symbols match assumptions/model.md (Formulation 2).
    """

    label: str          # names the results folder
    anatomy_path: str   # CSV of pixel labels, relative to the project root
    alpha: float        # attenuation per body pixel crossed
    n_S: int            # number of treatment sessions
    l_T: float          # min per-session dose, tumor pixel
    L_T: float          # min cumulative dose, tumor pixel
    U_T: float          # max cumulative dose, tumor pixel
    u_C: float          # max per-session dose, critical pixel (hard)
    U_C: float          # max cumulative dose, critical pixel (hard)
    u_N: float          # recommended max per-session dose, normal pixel (soft)
    U_N: float          # recommended max cumulative dose, normal pixel (soft)
    w_pk: float         # weight on per-session critical peaks
    w_C: float          # cost per unit dose to a critical pixel
    w_N: float          # cost per unit dose to a normal pixel
    lambda_e: float     # penalty per unit per-session normal overage
    lambda_E: float     # penalty per unit cumulative normal overage


def load_config(path):
    """Read the config JSON at `path`, validate it, and return a Config.

    Raises ValueError naming the offending key if anything is wrong.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"config {path}: top level must be a JSON object")

    # --- Every required key present, no unknown keys (catches typos) ---
    expected = [f.name for f in fields(Config)]
    for key in expected:
        if key not in raw:
            raise ValueError(f"config key '{key}': missing")
    for key in raw:
        if key not in expected:
            raise ValueError(f"config key '{key}': unknown key (expected one of {expected})")

    # --- String fields: non-empty strings ---
    for key in ("label", "anatomy_path"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"config key '{key}': must be a non-empty string, got {raw[key]!r}")

    # --- Numeric fields: int or float (JSON true/false are rejected, since bool is an int in Python) ---
    numeric_keys = [k for k in expected if k not in ("label", "anatomy_path")]
    for key in numeric_keys:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"config key '{key}': must be a number, got {value!r}")

    # --- n_S: positive integer (4.0 is rejected) ---
    if not isinstance(raw["n_S"], int) or raw["n_S"] < 1:
        raise ValueError(f"config key 'n_S': must be a positive integer, got {raw['n_S']!r}")

    # --- alpha in [0, 1) ---
    if not 0 <= raw["alpha"] < 1:
        raise ValueError(f"config key 'alpha': must satisfy 0 <= alpha < 1, got {raw['alpha']!r}")

    # --- Dose thresholds non-negative ---
    for key in ("l_T", "L_T", "U_T", "u_C", "U_C", "u_N", "U_N"):
        if raw[key] < 0:
            raise ValueError(f"config key '{key}': dose threshold must be >= 0, got {raw[key]!r}")

    # --- Weights non-negative (0 is allowed: w_pk = 0 gives Formulation 4, w_C = w_N = 0 gives Formulation 1) ---
    for key in ("w_pk", "w_C", "w_N", "lambda_e", "lambda_E"):
        if raw[key] < 0:
            raise ValueError(f"config key '{key}': weight must be >= 0, got {raw[key]!r}")

    # --- Feasibility: n_S sessions at l_T each must fit under the cumulative tumor cap ---
    if raw["n_S"] * raw["l_T"] > raw["U_T"]:
        raise ValueError(
            f"config key 'U_T': n_S*l_T = {raw['n_S']}*{raw['l_T']} = {raw['n_S'] * raw['l_T']} "
            f"> U_T = {raw['U_T']}, so no plan can exist"
        )

    values = {k: raw[k] for k in expected}
    for key in numeric_keys:
        if key != "n_S":
            values[key] = float(values[key])
    return Config(**values)


def load_anatomy(path):
    """Read the anatomy CSV at `path` and return it as a 2D int ndarray.

    Codes: 0 outside body, 1 tumor, 2 critical, 3 normal.
    Raises ValueError if rows differ in length, any code is outside 0-3, or there is no tumor pixel.
    """
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    while lines and lines[-1] == "":   # tolerate trailing blank lines
        lines.pop()
    if not lines:
        raise ValueError(f"anatomy {path}: file is empty")

    rows = []
    for i, line in enumerate(lines, start=1):
        try:
            rows.append([int(token) for token in line.split(",")])
        except ValueError:
            raise ValueError(f"anatomy {path}: row {i} has a non-integer cell: {line!r}") from None

    # --- Rectangular: every row the same length as row 1 ---
    width = len(rows[0])
    for i, row in enumerate(rows, start=1):
        if len(row) != width:
            raise ValueError(f"anatomy {path}: row {i} has {len(row)} cells, row 1 has {width}")

    grid = np.array(rows, dtype=int)

    # --- Codes only 0-3 ---
    bad = np.argwhere((grid < 0) | (grid > 3))
    if bad.size:
        r, c = bad[0]
        raise ValueError(
            f"anatomy {path}: code {grid[r, c]} at row {r + 1}, column {c + 1} is not in 0-3 "
            f"({len(bad)} bad cell(s) total)"
        )

    # --- At least one tumor pixel ---
    if not (grid == 1).any():
        raise ValueError(f"anatomy {path}: no tumor pixel (code 1) found")

    return grid
