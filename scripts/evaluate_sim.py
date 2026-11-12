"""Regenerate the simulated evaluation table shown in the README.

Usage: python scripts/evaluate_sim.py [--trials 20] [--output docs/evaluation-table.md]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from lora_check.sim.evaluate import NOISE_LEVELS, evaluate, markdown_table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    started = time.time()
    table = evaluate(
        trials=args.trials, noise_levels=NOISE_LEVELS, progress=lambda m: print(m, file=sys.stderr)
    )
    text = markdown_table(table)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(f"{args.trials} trials per cell, {time.time() - started:.0f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
