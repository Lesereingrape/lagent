"""Token vocabulary for the ReAct agent - a tiny controlled action language.

Everything the agent reads or writes is one of these tokens: structure markers
that the harness owns (question / action / observation delimiters) plus the tokens
the *policy* must actually choose - a tool name and a node argument.  Keeping the
action space to two predicted tokens per step is what makes the scaffold ablations
in ``study.py`` a clean causal test rather than a free-form generation mess.
"""

from __future__ import annotations

N_NODES = 16

PAD = 0
EOS = 1
QM = 2      # question marker
AM = 3      # action marker (harness writes it before each predicted action)
OM = 4      # observation marker
SEP = 5
NONE = 6    # "this tool takes no argument"

NEIGHBOR = 7
MAXVAL = 8
FINISH = 9

TPL_TOP1 = 10   # highest value among a node's neighbours
TPL_TOP2 = 11   # highest value among the neighbours of that node (two hops)

NODE0 = 12
TOOLS = (NEIGHBOR, MAXVAL, FINISH)
NODE_IDS = list(range(NODE0, NODE0 + N_NODES))
NVOCAB = NODE0 + N_NODES


def node_tok(i: int) -> int:
    if not 0 <= i < N_NODES:
        raise ValueError(f"node {i} out of range")
    return NODE0 + i


def tok_node(t: int) -> int:
    if not NODE0 <= t < NVOCAB:
        raise ValueError(f"token {t} is not a node")
    return t - NODE0
