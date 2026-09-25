"""The verifiable tool-use environment: a small attributed graph.

The agent must find the node with the highest ``value`` either among a start
node's neighbours (``top1``) or among the neighbours of that best neighbour
(``top2`` - two hops).  The only way to answer ``top2`` is to *compose tools*:
look up a neighbourhood, take the value-max of it, look that node up again, and
finish with the winner.  A single direct guess can never do it, which is exactly
the property the scaffold ablations are built to measure.

Gold answers are computed here by running the correct program, so the verifier is
ground truth, independent of whatever the policy does.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .vocab import FINISH, MAXVAL, N_NODES, NEIGHBOR, NONE, TPL_TOP1, TPL_TOP2, node_tok, tok_node

TOOLS = (NEIGHBOR, MAXVAL, FINISH)


@dataclass
class Graph:
    values: list[int]
    adj: list[list[int]]

    def neighbours(self, i: int) -> list[int]:
        return self.adj[i]

    def argmax_value(self, ids: list[int]) -> int:
        # highest value; smallest node id breaks ties (deterministic gold)
        return min(ids, key=lambda n: (-self.values[n], n))


def make_graph(rng: random.Random) -> Graph:
    values = [rng.randrange(10) for _ in range(N_NODES)]
    adj: list[list[int]] = []
    for i in range(N_NODES):
        deg = rng.randint(2, 4)
        cand = [j for j in range(N_NODES) if j != i]
        adj.append(sorted(rng.sample(cand, deg)))
    return Graph(values, adj)


@dataclass
class Task:
    graph: Graph
    tpl: int          # TPL_TOP1 / TPL_TOP2
    start: int

    @property
    def start_tok(self) -> int:
        return node_tok(self.start)

    def gold_program(self) -> list[tuple[int, int]]:
        """The correct (tool, arg-node) sequence ending in FINISH."""
        g = self.graph
        m1 = g.argmax_value(g.neighbours(self.start))
        if self.tpl == TPL_TOP1:
            return [(NEIGHBOR, self.start), (MAXVAL, NONE), (FINISH, m1)]
        m2 = g.argmax_value(g.neighbours(m1))
        return [(NEIGHBOR, self.start), (MAXVAL, NONE), (NEIGHBOR, m1),
                (MAXVAL, NONE), (FINISH, m2)]

    def gold_answer(self) -> int:
        return self.gold_program()[-1][1]


def sample_tasks(graph: Graph, n: int, rng: random.Random) -> list[Task]:
    tasks = []
    for _ in range(n):
        tpl = rng.choice([TPL_TOP1, TPL_TOP2])
        tasks.append(Task(graph, tpl, rng.randrange(N_NODES)))
    return tasks


def sample_tasks_multi_graph(seed: int, n: int) -> list[Task]:
    """Tasks spread over several graphs so the policy cannot memorise one."""
    rng = random.Random(seed)
    out: list[Task] = []
    for _ in range(n):
        g = make_graph(rng)
        out.extend(sample_tasks(g, 1, rng))
    return out


@dataclass
class Episode:
    """Stateful executor for one rollout of a task."""
    task: Task
    available: set[int] = field(default_factory=set)
    last_list: list[int] = field(default_factory=list)
    done: bool = False
    answer: int | None = None

    def __post_init__(self) -> None:
        self.available = {self.task.start}

    def legal_tools(self) -> tuple[int, ...]:
        return TOOLS

    def legal_args(self, tool: int) -> list[int]:
        """Node tokens the agent is allowed to ground an argument on right now."""
        if tool == MAXVAL:
            return [NONE]
        return sorted(node_tok(a) for a in self.available)

    def step(self, tool: int, arg_tok: int) -> list[int]:
        """Execute one action, return its observation tokens (possibly empty)."""
        g = self.task.graph
        if tool == FINISH:
            self.done = True
            self.answer = tok_node(arg_tok)
            return []
        if tool == NEIGHBOR:
            i = tok_node(arg_tok)
            nbrs = g.neighbours(i)
            self.last_list = list(nbrs)
            self.available.update(nbrs)
            return [node_tok(x) for x in nbrs]
        if tool == MAXVAL:
            if not self.last_list:
                return []
            best = g.argmax_value(self.last_list)
            self.available.add(best)
            self.last_list = [best]
            return [node_tok(best)]
        raise ValueError(f"unknown tool {tool}")
