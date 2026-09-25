"""A tiny decoder-only transformer that acts as the ReAct policy.

Same nanoGPT-shaped block the other labs use, at ~107k parameters so it trains and
rolls out in CPU seconds.  It is a *next-action* model: given the question plus the
observation scratchpad so far, it emits the tool token and then the argument token.
A loss mask lets us train only on those decision positions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .traces import MAXLEN
from .vocab import NVOCAB, PAD


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        return (x + a) + self.mlp(self.ln2(x))


class ReActPolicy(nn.Module):
    def __init__(self, d_model: int = 64, n_head: int = 4, n_layer: int = 2):
        super().__init__()
        self.tok = nn.Embedding(NVOCAB, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(MAXLEN, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, NVOCAB)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).unsqueeze(0).expand(b, t)
        x = self.tok(ids) + self.pos(pos)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=ids.device),
                          diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))

    def action_loss(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked next-token cross-entropy over the decision positions only."""
        logits = self.forward(ids)
        pred = logits[:, :-1, :]              # predict token at position i+1
        tgt = ids[:, 1:]
        m = mask[:, 1:].bool() & (ids[:, 1:] != PAD)
        sel = pred.reshape(-1, NVOCAB)[m.reshape(-1)]
        target = tgt.reshape(-1)[m.reshape(-1)]
        if sel.numel() == 0:
            return torch.tensor(0.0, requires_grad=True)
        return F.cross_entropy(sel, target)

    @torch.no_grad()
    def next_token(self, ctx: list[int], legal: list[int]) -> int:
        """Greedy next token restricted to ``legal`` (constrained decoding)."""
        self.eval()
        ids = torch.tensor([ctx], dtype=torch.long)
        logits = self.forward(ids)[:, -1, :][0]
        scores = torch.full_like(logits, float("-inf"))
        for t in legal:
            scores[t] = logits[t]
        return int(scores.argmax().item())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
