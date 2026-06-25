"""Validate a single STEP file using the cadgenbench validity gate.

Runs as a real script (not a `python -` heredoc) so the spawn-context
multiprocessing pool can re-import this module in worker children.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", type=Path, help="Path to the STEP file to validate.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("CADGENBENCH_MESH_TIMEOUT_S", "180")),
        help="Per-mesh tessellation timeout (seconds).",
    )
    args = parser.parse_args()

    os.environ["CADGENBENCH_MESH_TIMEOUT_S"] = str(args.timeout)

    from cadgenbench.common.validity import validate_step

    t = time.time()
    result = validate_step(args.step)
    dt = time.time() - t

    print(f"validate took {dt:.1f}s (timeout={args.timeout}s)")
    print(f"validation: {result.validation}")
    print(f"measurements: {result.measurements}")
    return 0 if result.validation.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
