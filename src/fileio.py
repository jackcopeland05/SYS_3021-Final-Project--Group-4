"""All file I/O for the radiation therapy LP: loading/validating inputs and saving runs.

This is the only module that touches the filesystem.
"""

import json
import shutil
from datetime import datetime
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

    # --- Feasibility: n_S is the MAXIMUM number of sessions (the plan picks k <= n_S), so only
    #     k = 1 has to fit: one session at l_T must not exceed the cumulative tumor cap ---
    if raw["l_T"] > raw["U_T"]:
        raise ValueError(
            f"config key 'U_T': l_T = {raw['l_T']} > U_T = {raw['U_T']}, so even a single session "
            f"overdoses the tumor and no plan can exist"
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


def save_run(root, config_path, problem, solution, metrics, dose, figures, summary):
    """Write one run to a new folder results/<YYYY-MM-DD_HHMM>_<label>/ under `root`; return its Path.

    root: project root (anatomy_path and results/ are relative to it).

    config_path: the config file the run was loaded from.
    problem, solution, dose: the chosen number of sessions k (problem.cfg.n_S == k).
    
    Contents: anatomy.csv (byte copy), config.json (the original JSON with its original n_S, and
    
    anatomy_path pointing at the copied anatomy.csv, so `python run.py --config
    results/<run>/config.json` repeats the run), metrics.json, summary.txt, plots/<name>.png (150 dpi)
    and, only if the status is "Optimal", two labeled CSVs written at full precision (%.17g):
      x.csv     one row per beamlet:   side, line, s1..sk
      dose.csv  one row per body pixel: row, col, label, s1..sk, cumulative   (dose = D @ x)
    An existing folder is never overwritten: a suffix _2, _3, ... is added instead.
    """
    cfg = problem.cfg
    root = Path(root)
    results = root / "results"
    base = f"{datetime.now():%Y-%m-%d_%H%M}_{cfg.label}"
    folder = results / base
    n = 2
    while folder.exists():
        folder = results / f"{base}_{n}"
        n += 1
    (folder / "plots").mkdir(parents=True)

    # --- Anatomy (byte copy) and a config that points at it ---
    shutil.copyfile(root / cfg.anatomy_path, folder / "anatomy.csv")
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)                       # keeps the original key order
    raw["anatomy_path"] = (folder / "anatomy.csv").relative_to(root).as_posix()
    with open(folder / "config.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
        f.write("\n")

    # --- Plan and dose as labeled CSVs (only when there is a plan) ---
    if solution.status == "Optimal":
        sessions = [f"s{s + 1}" for s in range(cfg.n_S)]
        with open(folder / "x.csv", "w", encoding="utf-8", newline="") as f:
            f.write(",".join(["side", "line"] + sessions) + "\n")
            for (side, line), row in zip(problem.beamlets, solution.x):
                f.write(",".join([side, str(line)] + [f"{v:.17g}" for v in row]) + "\n")
        label_names = {1: "tumor", 2: "critical", 3: "normal"}
        with open(folder / "dose.csv", "w", encoding="utf-8", newline="") as f:
            f.write(",".join(["row", "col", "label"] + sessions + ["cumulative"]) + "\n")
            for (r, c), row in zip(problem.pixel_rc, dose):
                cells = [str(r), str(c), label_names[int(problem.grid[r, c])]]
                f.write(",".join(cells + [f"{v:.17g}" for v in row] + [f"{row.sum():.17g}"]) + "\n")

    # --- Metrics, summary ---
    with open(folder / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=False)
        f.write("\n")
    with open(folder / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    # --- Plots ---
    for name, fig in figures.items():
        fig.savefig(folder / "plots" / f"{name}.png", dpi=150)
        fig.clear()

    return folder
