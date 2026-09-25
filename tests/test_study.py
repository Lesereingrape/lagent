"""Aggregation and result assembly run on synthetic per-seed records."""

from __future__ import annotations

from lagent.agent import SCAFFOLDS
from lagent.study import N_EVAL, SEEDS, ScaffoldRun, _summarise, aggregate, build_results


def _fake_seed(seed: int) -> dict:
    scaffolds = {}
    for i, sc in enumerate(SCAFFOLDS):
        scaffolds[sc.name] = {"scaffold": sc.name,
                              "solve_rate": 0.9 - 0.2 * i,
                              "invalid_rate": 0.0,
                              "mean_steps": 3.0 + i}
    return {"seed": seed, "n_eval": N_EVAL, "scaffolds": scaffolds,
            "react_by_template": {"top1": 1.0, "top2": 0.8},
            "gold": {"scaffold": "", "solve_rate": 1.0, "invalid_rate": 0.0,
                     "mean_steps": 4.0}}


def test_summarise_averages_flags():
    res = [{"solved": True, "invalid": False, "steps": 3},
           {"solved": False, "invalid": True, "steps": 5}]
    s = _summarise(res)
    assert isinstance(s, ScaffoldRun)
    assert s.solve_rate == 0.5 and s.invalid_rate == 0.5
    assert s.mean_steps == 4.0


def test_aggregate_shape_and_ordering():
    per_seed = [_fake_seed(s) for s in SEEDS]
    agg = aggregate(per_seed)
    assert agg["episodes"] == len(SEEDS)
    assert set(agg["per_scaffold"]) == {sc.name for sc in SCAFFOLDS}
    assert agg["per_scaffold"]["react"]["solve_mean"] > \
        agg["per_scaffold"]["no_scratchpad"]["solve_mean"]
    assert agg["gold_solve"] == 1.0
    assert agg["react_by_template"]["top1_mean"] == 1.0


def test_build_results_wraps_config_and_runtime():
    per_seed = [_fake_seed(s) for s in SEEDS]
    out = build_results(per_seed, runtime=12.3)
    assert out["config"]["seeds"] == list(SEEDS)
    assert out["config"]["n_eval"] == N_EVAL
    assert out["runtime_sec"] == 12.3
    assert "summary" in out and len(out["per_seed"]) == len(SEEDS)
