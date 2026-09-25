"""Run the seeded scaffold-ablation study.

One behaviour-cloned policy is trained per seed and then dropped into four different
agent *scaffolds* on the *same* held-out tasks.  Because the weights are identical,
the success gap between scaffolds is a clean causal measurement of what the harness
contributes - not the model.  A ``gold`` row runs the correct program directly to fix
the solvability ceiling.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass

import torch

from .agent import DIRECT, FULL, NO_CONSTRAIN, NO_SCRATCHPAD, gold_rollout, rollout
from .model import count_parameters
from .train import eval_tasks, train_policy
from .vocab import TPL_TOP1, TPL_TOP2

SEEDS = (0, 1, 2)
N_EVAL = 60
TRAIN_STEPS = 1500

SCAFFOLD_ORDER = (FULL, NO_SCRATCHPAD, NO_CONSTRAIN, DIRECT)


@dataclass
class ScaffoldRun:
    scaffold: str
    solve_rate: float
    invalid_rate: float
    mean_steps: float


def _summarise(res: list[dict]) -> ScaffoldRun:
    n = len(res)
    return ScaffoldRun(
        scaffold="",
        solve_rate=round(sum(r["solved"] for r in res) / n, 4),
        invalid_rate=round(sum(r["invalid"] for r in res) / n, 4),
        mean_steps=round(sum(r["steps"] for r in res) / n, 3),
    )


def environment() -> dict:
    """The machine these numbers came off, recorded beside them.

    Beam-style constrained decoding and batched logit argmaxes are float reductions
    whose order depends on the torch build and the thread count, so a rerun is
    field-for-field identical *inside* this environment and merely close outside it.
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "device": "cpu",
    }


def run_seed(seed: int) -> dict:
    model = train_policy(seed=seed, steps=TRAIN_STEPS)
    tasks = eval_tasks(seed, N_EVAL)
    scaffolds = {}
    for sc in SCAFFOLD_ORDER:
        rs = _summarise([rollout(model, t, sc) for t in tasks])
        rs.scaffold = sc.name
        scaffolds[sc.name] = asdict(rs)
    react = [rollout(model, t, FULL) for t in tasks]
    by_tpl = {}
    for tpl, name in ((TPL_TOP1, "top1"), (TPL_TOP2, "top2")):
        sub = [r for t, r in zip(tasks, react, strict=True) if t.tpl == tpl]
        if sub:
            by_tpl[name] = round(sum(x["solved"] for x in sub) / len(sub), 4)
    gold = _summarise([gold_rollout(t) for t in tasks])
    return {"seed": seed, "n_eval": N_EVAL, "n_params": count_parameters(model),
            "scaffolds": scaffolds,
            "react_by_template": by_tpl, "gold": asdict(gold)}


def aggregate(per_seed: list[dict]) -> dict:
    names = [sc.name for sc in SCAFFOLD_ORDER]
    out: dict = {"seeds": list(SEEDS), "n_eval": N_EVAL,
                 "scaffold_order": names, "per_scaffold": {}, "react_by_template": {}}
    for name in names:
        solves = [s["scaffolds"][name]["solve_rate"] for s in per_seed]
        invalids = [s["scaffolds"][name]["invalid_rate"] for s in per_seed]
        steps = [s["scaffolds"][name]["mean_steps"] for s in per_seed]
        out["per_scaffold"][name] = {
            "solve_mean": round(sum(solves) / len(solves), 4),
            "solve_std": round(_std(solves), 4),
            "invalid_mean": round(sum(invalids) / len(invalids), 4),
            "steps_mean": round(sum(steps) / len(steps), 3),
            "solves": [round(v, 4) for v in solves],
        }
    tpl1 = [s["react_by_template"].get("top1") for s in per_seed]
    tpl2 = [s["react_by_template"].get("top2") for s in per_seed]
    out["react_by_template"] = {
        "top1_mean": round(_mean(tpl1), 4),
        "top2_mean": round(_mean(tpl2), 4),
    }
    gold = [s["gold"]["solve_rate"] for s in per_seed]
    out["gold_solve"] = round(sum(gold) / len(gold), 4)
    out["episodes"] = len(per_seed)
    return out


def _mean(xs: list) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def build_results(per_seed: list[dict], runtime: float) -> dict:
    """Assemble the artifact, with the trained policy's size in its config.

    ``n_params`` is read off the models that actually ran rather than rebuilt from
    defaults, so the README's "deliberately toy" sentence cannot outlive a change to
    the architecture.
    """
    sizes = {seed["n_params"] for seed in per_seed}
    assert sizes == {per_seed[0]["n_params"]}, f"seeds trained different sizes: {sizes}"
    return {"config": {"seeds": list(SEEDS), "n_eval": N_EVAL,
                       "train_steps": TRAIN_STEPS, "n_params": per_seed[0]["n_params"]},
            "summary": aggregate(per_seed),
            "per_seed": per_seed,
            "environment": environment(),
            "runtime_sec": round(runtime, 1)}
