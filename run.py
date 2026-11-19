"""Command-line entry point: run the radiation therapy LP for one config file.

    python run.py --config data/configs/baseline.json

Pipeline (no modeling logic here):
load_config -> load_anatomy -> build_problem -> build_model -> solve -> compute_metrics
-> make_plots -> format_summary -> save_run.

Paths in the config (anatomy_path) and the results/ folder are relative to the project root,
which is the folder holding this file. Exit code 0 whenever the pipeline finishes (an infeasible
status is a result); 1 on a bad config/anatomy or a missing file, with nothing written.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rtlp.analysis import compute_metrics, format_summary, make_plots  # noqa: E402
from rtlp.geometry import build_problem  # noqa: E402
from rtlp.io import load_anatomy, load_config, save_run  # noqa: E402
from rtlp.model import build_model  # noqa: E402
from rtlp.solve import solve  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Solve the multi-session radiation therapy LP (Formulation 2).")
    parser.add_argument("--config", required=True, help="path to the config JSON")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        grid = load_anatomy(ROOT / cfg.anatomy_path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    problem = build_problem(grid, cfg)
    solution = solve(build_model(problem))
    metrics, dose = compute_metrics(problem, solution)
    figures = make_plots(problem, solution)
    summary = format_summary(metrics)

    print(summary, end="")
    folder = save_run(ROOT, args.config, cfg, solution, metrics, dose, figures, summary)
    print(f"results written to {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
