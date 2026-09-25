"""The graph environment's gold program is ground truth, independent of any policy."""

from __future__ import annotations

import random

from lagent.env import Episode, Graph, Task, make_graph, sample_tasks
from lagent.vocab import FINISH, MAXVAL, NEIGHBOR, NONE, TPL_TOP1, TPL_TOP2, node_tok, tok_node


def test_argmax_value_tiebreak_is_deterministic():
    g = Graph(values=[3, 5, 5, 1], adj=[[], [], [], []])
    assert g.argmax_value([0, 1, 2, 3]) == 1   # value 5 wins, smallest id breaks tie
    assert g.argmax_value([0, 3]) == 0


def test_gold_program_top1_and_top2_end_on_answer():
    rng = random.Random(0)
    for tpl in (TPL_TOP1, TPL_TOP2):
        task = Task(make_graph(rng), tpl, rng.randrange(16))
        prog = task.gold_program()
        assert prog[-1][0] == FINISH
        assert prog[-1][1] == task.gold_answer()
        n_max = sum(1 for t, _ in prog if t == MAXVAL)
        assert n_max == (1 if tpl == TPL_TOP1 else 2)


def test_episode_executes_gold_program_to_answer():
    rng = random.Random(7)
    task = Task(make_graph(rng), TPL_TOP2, rng.randrange(16))
    ep = Episode(task)
    for tool, arg in task.gold_program():
        ep.step(tool, NONE if tool == MAXVAL else node_tok(arg))
    assert ep.done and ep.answer == task.gold_answer()


def test_legal_args_track_discovered_nodes():
    rng = random.Random(3)
    start = 5
    task = Task(make_graph(rng), TPL_TOP1, start)
    ep = Episode(task)
    assert ep.legal_args(NEIGHBOR) == [node_tok(start)]
    assert ep.legal_args(MAXVAL) == [NONE]
    ep.step(NEIGHBOR, node_tok(start))
    # after a neighbors() call the returned nodes become groundable
    assert set(ep.legal_args(NEIGHBOR)) >= {node_tok(start)}


def test_sample_tasks_only_uses_known_templates():
    rng = random.Random(1)
    g = make_graph(rng)
    tasks = sample_tasks(g, 20, rng)
    assert all(t.tpl in (TPL_TOP1, TPL_TOP2) for t in tasks)
    assert {t.tpl for t in tasks} == {TPL_TOP1, TPL_TOP2}


def test_node_token_roundtrip():
    for i in range(16):
        assert tok_node(node_tok(i)) == i
