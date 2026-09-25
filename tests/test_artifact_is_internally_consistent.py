"""Recompute every published aggregate from the raw per-seed traces.

``results/agent.json`` carries both the four-seed-run traces (``per_seed``) and the
means/stds the README prints (``summary``). This test derives the second from the
first with the same arithmetic the study uses - mean of the per-seed rates, **sample**
standard deviation (``n-1``) - and fails if they disagree, which is how a hand-edited
or stale summary block gets caught. It also pins the two ablation stories the README
tells, so a regeneration that quietly changes their direction has to be written up
rather than pasted over.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results" / "agent.json").read_text(encoding="utf-8"))

SUMMARY = DATA["summary"]
PER_SEED = DATA["per_seed"]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _rates(scaffold: str, key: str) -> list[float]:
    return [s["scaffolds"][scaffold][key] for s in PER_SEED]


def test_artifact_shape_is_coherent():
    assert SUMMARY["seeds"] == DATA["config"]["seeds"]
    assert SUMMARY["episodes"] == len(PER_SEED)
    assert SUMMARY["n_eval"] == DATA["config"]["n_eval"]
    assert [s["seed"] for s in PER_SEED] == SUMMARY["seeds"]
    for s in PER_SEED:
        assert s["n_eval"] == SUMMARY["n_eval"]
        assert list(s["scaffolds"]) == SUMMARY["scaffold_order"]
        assert set(s) == {"seed", "n_eval", "n_params", "scaffolds",
                          "react_by_template", "gold"}
    assert {s["n_params"] for s in PER_SEED} == {DATA["config"]["n_params"]}, (
        "the config's policy size is not the size the seeds trained")


def test_per_scaffold_aggregates_match_the_traces():
    assert set(SUMMARY["per_scaffold"]) == set(SUMMARY["scaffold_order"])
    for name, row in SUMMARY["per_scaffold"].items():
        solves, invalids, steps = (_rates(name, k) for k in
                                   ("solve_rate", "invalid_rate", "mean_steps"))
        assert set(row) == {"solve_mean", "solve_std", "invalid_mean",
                            "steps_mean", "solves"}
        assert row["solve_mean"] == round(_mean(solves), 4), name
        assert row["solve_std"] == round(_std(solves), 4), (
            f"{name}: sample-vs-population std drift")
        assert row["invalid_mean"] == round(_mean(invalids), 4), name
        assert row["steps_mean"] == round(_mean(steps), 3), name
        assert row["solves"] == [round(v, 4) for v in solves], name


def test_template_and_gold_aggregates_match_the_traces():
    for key in ("top1", "top2"):
        got = round(_mean([s["react_by_template"][key] for s in PER_SEED]), 4)
        assert SUMMARY["react_by_template"][f"{key}_mean"] == got, key
    gold_rates = [s["gold"]["solve_rate"] for s in PER_SEED]
    assert SUMMARY["gold_solve"] == round(_mean(gold_rates), 4)


def test_react_scaffold_is_at_the_gold_ceiling():
    assert SUMMARY["per_scaffold"]["react"]["solve_mean"] == SUMMARY["gold_solve"]


def test_the_ablation_directions_the_prose_reports():
    # Removing observation memory, or answering in one shot, collapses the agent.
    full = SUMMARY["per_scaffold"]["react"]["solve_mean"]
    for name in ("no_scratchpad", "direct"):
        assert SUMMARY["per_scaffold"][name]["solve_mean"] < 0.10, name
        assert SUMMARY["per_scaffold"][name]["solve_mean"] < full
    # Constrained decoding is reported as a *null* result: this policy already emits
    # legal actions, so the guard is a no-op. If a rerun separates them, the README
    # has to stop calling it a null.
    assert abs(SUMMARY["per_scaffold"]["no_constrain"]["solve_mean"] - full) <= 0.05
