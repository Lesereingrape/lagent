"""Command line entry point: print a readable ReAct rollout and the scaffold table.

    python -m lagent.cli demo         # one trained policy, one decoded trace
    python -m lagent.cli demo --seed 1
"""

from __future__ import annotations

import argparse

from .agent import DIRECT, FULL, NO_CONSTRAIN, NO_SCRATCHPAD, gold_rollout, rollout
from .model import count_parameters
from .train import eval_tasks, train_policy
from .vocab import FINISH, MAXVAL, NEIGHBOR, NONE, TPL_TOP1, TPL_TOP2, tok_node

TOOL_NAME = {NEIGHBOR: "neighbors", MAXVAL: "max_value", FINISH: "finish"}
TPL_NAME = {TPL_TOP1: "top-1 neighbour", TPL_TOP2: "top-2 neighbour"}


def _fmt_decision(tool: int, arg: int) -> str:
    arg_txt = "-" if arg == NONE or tool == MAXVAL else f"node {tok_node(arg)}"
    return f"{TOOL_NAME[tool]:>9s}({arg_txt})"


def demo(seed: int = 0) -> None:
    model = train_policy(seed=seed)
    print(f"seed {seed}  |  policy parameters = {count_parameters(model):,}")
    task = eval_tasks(seed, 4)[3]
    g = task.graph
    print(f"\nquestion: find the {TPL_NAME[task.tpl]} of node {task.start}"
          f"  (values {g.values})")
    print(f"gold answer = node {task.gold_answer()}\n")

    r = rollout(model, task, FULL)
    print("react trace:")
    for tool, arg in r["decisions"]:
        print("  " + _fmt_decision(tool, arg))
    print(f"  -> solved={r['solved']} in {r['steps']} steps\n")

    print("scaffold summary (this one task):")
    for sc in (FULL, NO_SCRATCHPAD, NO_CONSTRAIN, DIRECT):
        rs = rollout(model, task, sc)
        print(f"  {sc.name:14s} solved={rs['solved']}  invalid={rs['invalid']}")
    print(f"  {'gold':14s} solved={gold_rollout(task)['solved']}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="lagent")
    ap.add_argument("command", nargs="?", default="demo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.command == "demo":
        demo(args.seed)
    else:
        raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
