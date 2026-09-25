"""The tiny ReAct policy: shapes, that its masked loss trains, and constrained decoding."""

from __future__ import annotations

import random

import torch

from lagent.env import Task, make_graph
from lagent.model import ReActPolicy, count_parameters
from lagent.traces import build_sequence, pad_batch
from lagent.vocab import NODE_IDS, NVOCAB, TPL_TOP1


def _seqs(n=8, seed=0):
    rng = random.Random(seed)
    return [build_sequence(Task(make_graph(rng), TPL_TOP1, rng.randrange(16)))
            for _ in range(n)]


def test_forward_shape():
    model = ReActPolicy()
    ids_t, _ = pad_batch(_seqs())
    ids = torch.tensor(ids_t, dtype=torch.long)
    logits = model.forward(ids)
    assert logits.shape == (ids.shape[0], ids.shape[1], NVOCAB)


def test_action_loss_decreases_on_a_small_batch():
    model = ReActPolicy()
    seqs = _seqs(8, seed=1)
    ids, mask = pad_batch(seqs)
    ids = torch.tensor(ids, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    first = None
    for _ in range(15):
        loss = model.action_loss(ids, mask)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_next_token_respects_legal_set():
    model = ReActPolicy()
    rng = random.Random(2)
    ids, _ = build_sequence(Task(make_graph(rng), TPL_TOP1, 5))
    tok = model.next_token(ids[:3], NODE_IDS)
    assert tok in NODE_IDS


def test_parameter_count_is_tiny():
    n = count_parameters(ReActPolicy())
    assert 80_000 < n < 200_000
