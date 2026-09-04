"""Temporary bridge for the already-running 2026-08-26 tmux experiment.

New launches call charts/balance_learning_curve_2batch/generate_report.py
directly. This legacy bridge rewrites the output location to the study's
`result/learning_curve` directory, then removes itself after a successful handoff.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


BRIDGE_PATH = Path(__file__).resolve()
PROJECT_ROOT = BRIDGE_PATH.parents[4]
GENERATOR = PROJECT_ROOT / "charts" / "balance_learning_curve_2batch" / "generate_report.py"
RESULT_DIR = BRIDGE_PATH.parents[1] / "result" / "learning_curve"


def main() -> None:
    arguments = sys.argv[1:]
    if "--output-dir" in arguments:
        output_index = arguments.index("--output-dir") + 1
        if output_index >= len(arguments):
            raise SystemExit("--output-dir requires a value")
        arguments[output_index] = str(RESULT_DIR)
    sys.argv = [str(GENERATOR), *arguments]
    try:
        runpy.run_path(str(GENERATOR), run_name="__main__")
    except SystemExit as error:
        if error.code in (None, 0):
            BRIDGE_PATH.unlink(missing_ok=True)
        raise
    BRIDGE_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
