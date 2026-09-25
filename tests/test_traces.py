"""Expert traces: loss lands only on the tool + argument decision positions."""

from __future__ import annotations

import random

from lagent.env import Task, make_graph
from lagent.traces import MAXLEN, build_sequence, pad_batch
from lagent.vocab import EOS, FINISH, MAXVAL, NEIGHBOR, NODE_IDS, NONE, PAD, QM, TPL_TOP1

DECISION_TOKENS = set(NODE_IDS) | {NEIGHBOR, MAXVAL, FINISH, NONE}


def test_mask_marks_only_decisions():
    rng = random.Random(0)
    task = Task(make_graph(rng), TPL_TOP1, 4)
    ids, mask = build_sequence(task)
    assert len(ids) == len(mask) and len(ids) <= MAXLEN
    assert ids[0] == QM
    # every masked-in token is a legal tool or argument, nothing else
    for tok, m in zip(ids, mask, strict=True):
        if m:
            assert tok in DECISION_TOKENS, tok
    assert sum(mask) == 2 * len(task.gold_program())


def test_questions_and_observations_are_unmasked():
    rng = random.Random(2)
    ids, mask = build_sequence(Task(make_graph(rng), TPL_TOP1, 3))
    assert mask[0] == 0 and mask[1] == 0      # QM and template
    assert ids[mask.index(0)] != PAD


def test_pad_batch_shapes_and_truncation():
    rng = random.Random(1)
    seqs = [build_sequence(Task(make_graph(rng), TPL_TOP1, i)) for i in range(5)]
    ids, mask = pad_batch(seqs)
    width = len(ids[0])
    assert all(len(row) == width for row in ids + mask)
    assert width <= MAXLEN


def test_every_trace_starts_same_and_ends_eos():
    rng = random.Random(4)
    for _ in range(6):
        ids, _mask = build_sequence(Task(make_graph(rng), TPL_TOP1, rng.randrange(16)))
        assert ids[0] == QM
        assert EOS in ids
