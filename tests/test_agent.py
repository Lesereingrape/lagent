"""Rollouts terminate inside the step cap and the scaffolds behave differently."""

from __future__ import annotations

import random

from lagent.agent import (
    DIRECT,
    FULL,
    MAX_STEPS,
    NO_CONSTRAIN,
    NO_SCRATCHPAD,
    SCAFFOLDS,
    gold_rollout,
    rollout,
)
from lagent.env import Task, make_graph
from lagent.train import train_policy
from lagent.vocab import FINISH, TPL_TOP1, TPL_TOP2


def _tasks(n, seed=0):
    rng = random.Random(seed)
    return [Task(make_graph(rng), rng.choice([TPL_TOP1, TPL_TOP2]),
                 rng.randrange(16)) for _ in range(n)]


def test_gold_rollout_is_the_ceiling():
    for t in _tasks(15, seed=5):
        r = gold_rollout(t)
        assert r["solved"] is True and r["invalid"] is False
        assert 1 <= r["steps"] <= MAX_STEPS


def test_direct_is_single_step_and_returns_a_decision():
    policy = train_policy(seed=0, steps=3)   # barely trained: structure only
    r = rollout(policy, _tasks(1)[0], DIRECT)
    assert r["steps"] == 1
    assert len(r["decisions"]) == 1 and r["decisions"][0][0] == FINISH


def test_all_scaffolds_terminate_within_cap():
    policy = train_policy(seed=0, steps=3)
    for t in _tasks(6, seed=9):
        for sc in SCAFFOLDS:
            r = rollout(policy, t, sc)
            assert 1 <= r["steps"] <= MAX_STEPS
            assert set(r) >= {"solved", "invalid", "steps", "decisions"}


def test_scratchpad_ablation_reproduces_the_headline_gap():
    """A trained policy should solve with the scratchpad and collapse without it."""
    policy = train_policy(seed=1, steps=800)
    tasks = _tasks(20, seed=11)
    react = sum(rollout(policy, t, FULL)["solved"] for t in tasks) / len(tasks)
    nosc = sum(rollout(policy, t, NO_SCRATCHPAD)["solved"] for t in tasks) / len(tasks)
    direct = sum(rollout(policy, t, DIRECT)["solved"] for t in tasks) / len(tasks)
    nocon = sum(rollout(policy, t, NO_CONSTRAIN)["solved"] for t in tasks) / len(tasks)
    assert react >= 0.8
    assert nocon >= 0.8                       # constraint ablation is a null
    assert nosc < react and direct < react
