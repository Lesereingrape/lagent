"""Run the full 3-seed scaffold-ablation study and write results/agent.json.

    python experiments/run_study.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from lagent.study import SCAFFOLD_ORDER, SEEDS, build_results, run_seed


def main() -> None:
    t0 = time.time()
    per_seed = [run_seed(s) for s in SEEDS]
    results = build_results(per_seed, time.time() - t0)
    out = Path("results/agent.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    s = results["summary"]
    print(f"wrote {out} in {results['runtime_sec']}s")
    print(f"gold={s['gold_solve']:.3f}  "
          + "  ".join(f"{sc.name}={s['per_scaffold'][sc.name]['solve_mean']:.3f}"
                      for sc in SCAFFOLD_ORDER))


if __name__ == "__main__":
    main()
