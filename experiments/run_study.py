"""Run the full 3-seed scaffold-ablation study and write results/agent.json.

    python experiments/run_study.py [--out PATH]

``--out`` exists so a second run can be written to a scratch path and diffed field
for field against the committed artifact, which is how the README's reproducibility
claim gets checked rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lagent.study import SCAFFOLD_ORDER, SEEDS, build_results, run_seed


def main(out: str = "results/agent.json") -> None:
    t0 = time.time()
    per_seed = [run_seed(s) for s in SEEDS]
    results = build_results(per_seed, time.time() - t0)
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    s = results["summary"]
    print(f"wrote {dest} in {results['runtime_sec']}s")
    print(f"gold={s['gold_solve']:.3f}  "
          + "  ".join(f"{sc.name}={s['per_scaffold'][sc.name]['solve_mean']:.3f}"
                      for sc in SCAFFOLD_ORDER))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run_study")
    parser.add_argument("--out", default="results/agent.json",
                        help="where to write the artifact (default: results/agent.json)")
    main(parser.parse_args().out)
