"""Turn an anatomy grid and a Config into the sets, beamlets and attenuation matrix of the LP.

Pure functions: no file I/O and no solver. Symbols match assumptions/model.md (Formulation 2).

Conventions (0-based throughout):
- P is the body cells (code != 0) in row-major order. pixel_rc[i] = (row, col) of pixel i;
  pixel_index[row, col] = i, or -1 for outside-body cells.
- T, C, N are int arrays of indices into P (codes 1, 2, 3).
- B on an R x C grid has 2R + 2C beamlets, always, in this order:
  ("left", 0..R-1), ("right", 0..R-1), ("top", 0..C-1), ("bottom", 0..C-1).
  A beamlet whose row/column holds no body cell is kept with an all-zero column in D.
- D[p, b] = max(0, 1 - alpha*k), k = number of body cells beamlet b crossed before reaching p
  (outside-body cells add no depth). D[p, b] = 0 if b does not pass through p.
"""

from dataclasses import dataclass

import numpy as np

from fileio import Config


@dataclass(frozen=True)
class ProblemData:
    """Everything model.py and analysis.py need about one instance."""

    grid: np.ndarray         # R x C int labels (0 outside, 1 tumor, 2 critical, 3 normal)
    cfg: Config              # every run parameter
    pixel_rc: np.ndarray     # |P| x 2, (row, col) of each pixel in P
    pixel_index: np.ndarray  # R x C, index into P, or -1 outside the body
    T: np.ndarray            # indices into P of tumor pixels
    C: np.ndarray            # indices into P of critical pixels
    N: np.ndarray            # indices into P of normal pixels
    beamlets: list           # (side, line) tuples, side in {"left", "right", "top", "bottom"}
    D: np.ndarray            # |P| x |B| attenuation matrix


def build_sets(grid):
    """Return (pixel_rc, pixel_index, T, C, N) for the body cells of `grid`.

    Empty C or N is allowed (arrays of length 0).
    """
    pixel_rc = np.argwhere(grid != 0)            # row-major order
    pixel_index = np.full(grid.shape, -1, dtype=int)
    pixel_index[pixel_rc[:, 0], pixel_rc[:, 1]] = np.arange(len(pixel_rc))

    codes = grid[pixel_rc[:, 0], pixel_rc[:, 1]]
    T = np.flatnonzero(codes == 1)
    C = np.flatnonzero(codes == 2)
    N = np.flatnonzero(codes == 3)
    return pixel_rc, pixel_index, T, C, N


def build_beamlets(grid):
    """Return the 2R + 2C beamlets of an R x C grid as (side, line) tuples, in the fixed order."""
    n_rows, n_cols = grid.shape
    return (
        [("left", r) for r in range(n_rows)]
        + [("right", r) for r in range(n_rows)]
        + [("top", c) for c in range(n_cols)]
        + [("bottom", c) for c in range(n_cols)]
    )


def build_D(grid, pixel_index, beamlets, alpha):
    """Return the |P| x |B| attenuation matrix.

    Each beamlet is walked cell by cell from its entry side. k counts the body cells already
    crossed, so the first body cell a ray meets gets D = 1; outside-body cells add no depth.
    """
    n_rows, n_cols = grid.shape
    n_pixels = int((pixel_index >= 0).sum())
    D = np.zeros((n_pixels, len(beamlets)))

    for b, (side, line) in enumerate(beamlets):
        if side == "left":
            path = [(line, c) for c in range(n_cols)]
        elif side == "right":
            path = [(line, c) for c in reversed(range(n_cols))]
        elif side == "top":
            path = [(r, line) for r in range(n_rows)]
        elif side == "bottom":
            path = [(r, line) for r in reversed(range(n_rows))]
        else:
            raise ValueError(f"beamlet {b}: unknown side {side!r}")

        k = 0
        for r, c in path:
            p = pixel_index[r, c]
            if p < 0:
                continue  # outside the body: no dose, no depth
            D[p, b] = max(0.0, 1.0 - alpha * k)
            k += 1
    return D


def build_problem(grid, cfg):
    """Build the ProblemData for anatomy `grid` (from fileio.load_anatomy) and Config `cfg`."""
    pixel_rc, pixel_index, T, C, N = build_sets(grid)
    beamlets = build_beamlets(grid)
    D = build_D(grid, pixel_index, beamlets, cfg.alpha)
    return ProblemData(
        grid=grid,
        cfg=cfg,
        pixel_rc=pixel_rc,
        pixel_index=pixel_index,
        T=T,
        C=C,
        N=N,
        beamlets=beamlets,
        D=D,
    )
