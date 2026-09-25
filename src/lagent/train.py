"""Behaviour-clone the ReAct policy from expert traces on many random graphs.

Training regenerates fresh graphs every step, so the policy cannot memorise one
world - it must learn the reusable *procedure*: pick the neighbourhood of the node
named in the question, reduce it to its value-max, (for two-hop questions) recurse
once more, then finish on the node the last observation surfaced.  That last part
matters: the model only ever has to *copy* a node token that a tool just returned,
never compute anything itself, which is why a ~107k-parameter transformer can learn
it on a CPU.
"""

from __future__ import annotations

import random

import torch

from .env import make_graph, sample_tasks
from .model import ReActPolicy
from .traces import build_sequence, pad_batch


def _batch(seed: int, batch: int):
    rng = random.Random(seed)
    tasks = []
    for _ in range(batch):
        g = make_graph(rng)
        tasks.extend(sample_tasks(g, 1, rng))
    seqs = [build_sequence(t) for t in tasks]
    ids, mask = pad_batch(seqs)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


def train_policy(seed: int = 0, steps: int = 1500, batch: int = 64,
                 lr: float = 3e-3, d_model: int = 64, n_layer: int = 2) -> ReActPolicy:
    torch.manual_seed(seed)
    model = ReActPolicy(d_model=d_model, n_layer=n_layer)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for s in range(steps):
        ids, mask = _batch(seed * 1000 + s, batch)
        loss = model.action_loss(ids, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def eval_tasks(seed: int, n: int) -> list:
    """Held-out tasks on fresh graphs, disjoint from any training draw."""
    rng = random.Random(900_000 + seed)
    tasks = []
    for _ in range(n):
        g = make_graph(rng)
        tasks.extend(sample_tasks(g, 1, rng))
    return tasks
