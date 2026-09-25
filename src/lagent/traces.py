"""Turn a task's correct program into an expert ReAct trace for behaviour cloning.

The harness owns the structure markers (``QM`` question, ``AM`` action, ``OM``
observation); the policy is only ever asked to predict the two contentful tokens
of each action - the *tool* and its *argument*.  Training therefore puts loss only
on those positions, so the model learns to imitate decisions, not to re-type
delimiters or memorise observation text.
"""

from __future__ import annotations

from .env import Episode, Task
from .vocab import AM, EOS, FINISH, MAXVAL, NONE, OM, PAD, QM, node_tok

#: worst-case trace length (a two-hop episode with degree-4 neighbourhoods)
MAXLEN = 48


def build_sequence(task: Task) -> tuple[list[int], list[int]]:
    """Return (token ids, loss mask) for one expert episode."""
    ids = [QM, task.tpl, task.start_tok]
    mask = [0, 0, 0]
    ep = Episode(task)
    for tool, arg in task.gold_program():
        arg_tok = NONE if tool == MAXVAL else node_tok(arg)
        ids += [AM, tool, arg_tok]
        mask += [0, 1, 1]        # predict the tool and the argument
        obs = ep.step(tool, arg_tok)
        if tool != FINISH:
            ids = [*ids, OM, *obs]
            mask = [*mask, 0, *([0] * len(obs))]
    ids.append(EOS)
    mask.append(0)
    if len(ids) > MAXLEN:
        raise ValueError(f"trace {len(ids)} exceeds MAXLEN {MAXLEN}")
    return ids, mask


def pad_batch(seqs: list[tuple[list[int], list[int]]]):
    width = min(MAXLEN, max(len(ids) for ids, _ in seqs))
    ids_b, mask_b = [], []
    for ids, mask in seqs:
        pad = width - len(ids)
        ids_b.append(ids[:width] + [PAD] * pad)
        mask_b.append(mask[:width] + [0] * pad)
    return ids_b, mask_b
