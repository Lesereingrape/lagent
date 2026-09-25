"""Roll the learned policy out in the environment under different agent scaffolds.

Every scaffold here runs the *same* trained weights on the *same* task; only the
harness differs, so the success gap is a clean causal read on what each piece of
agent structure contributes:

- ``react``         - observations go into the context, decoding is constrained to
                      the currently legal tools/arguments (the full ReAct agent).
- ``no_scratchpad`` - tool *results* are withheld from the context. The policy is
                      still asked to ground arguments it can no longer see.
- ``no_constrain``  - the scratchpad is present but the model may emit any token as
                      a tool or argument, so illegal actions become possible.
- ``direct``        - a single-shot guess with no tools at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from .env import Episode, Task
from .model import ReActPolicy
from .traces import MAXLEN
from .vocab import AM, FINISH, MAXVAL, N_NODES, NONE, NVOCAB, OM, QM, TOOLS, node_tok

ALL_TOKS = list(range(NVOCAB))
ALL_NODES = [node_tok(i) for i in range(N_NODES)]


@dataclass
class Scaffold:
    name: str
    scratchpad: bool = True
    constrain: bool = True
    single_shot: bool = False


FULL = Scaffold("react")
NO_SCRATCHPAD = Scaffold("no_scratchpad", scratchpad=False)
NO_CONSTRAIN = Scaffold("no_constrain", constrain=False)
DIRECT = Scaffold("direct", single_shot=True)
SCAFFOLDS = (FULL, NO_SCRATCHPAD, NO_CONSTRAIN, DIRECT)

MAX_STEPS = 12


def _start_ctx(task: Task) -> list[int]:
    return [QM, task.tpl, task.start_tok]


def rollout(policy: ReActPolicy, task: Task, sc: Scaffold) -> dict:
    ep = Episode(task)
    ctx = _start_ctx(task)

    if sc.single_shot:
        # no tools: the model must name the answer node in one shot
        ctx = [*ctx, AM, FINISH]
        arg = policy.next_token(ctx, ALL_NODES)
        solved = arg == node_tok(task.gold_answer())
        return {"solved": solved, "invalid": False, "steps": 1,
                "decisions": [(FINISH, arg)]}

    steps = 0
    invalid = False
    decisions: list[tuple[int, int]] = []
    while not ep.done and steps < MAX_STEPS and len(ctx) + 3 <= MAXLEN:
        steps += 1
        ctx = [*ctx, AM]
        legal_tools = list(TOOLS) if sc.constrain else ALL_TOKS
        tool = policy.next_token(ctx, legal_tools)
        if tool not in TOOLS:
            invalid = True
            break
        ctx = [*ctx, tool]
        if tool == MAXVAL:
            legal_arg = [NONE]
            arg_pool = [NONE] if sc.constrain else ALL_TOKS
        else:
            legal_arg = ep.legal_args(tool)
            arg_pool = legal_arg if sc.constrain else ALL_NODES
        arg = policy.next_token(ctx, arg_pool)
        decisions.append((tool, arg))
        if not sc.constrain and arg not in legal_arg:
            invalid = True
            break
        ctx = [*ctx, arg]
        obs = ep.step(tool, arg)
        if tool == FINISH:
            break
        ctx = [*ctx, OM, *(obs if sc.scratchpad else [])]

    return {"solved": bool(ep.done and ep.answer == task.gold_answer()),
            "invalid": invalid, "steps": steps, "decisions": decisions}


def gold_rollout(task: Task) -> dict:
    """Run the correct program directly - the solvability ceiling."""
    ep = Episode(task)
    steps = 0
    for tool, arg in task.gold_program():
        steps += 1
        ep.step(tool, NONE if tool == MAXVAL else node_tok(arg))
    return {"solved": ep.answer == task.gold_answer(), "invalid": False,
            "steps": steps}
