"""The README is only allowed to describe the code that is actually in the repo.

The results block is byte-pinned against ``results/agent.json``; the size claims in
the first paragraph are the other hand-written numbers here - a "~650-line" line that
quietly becomes 900, or a parameter count that rounds the wrong way, is the same class
of drift, so both are measured from the code rather than remembered.
"""

from __future__ import annotations

import re
from pathlib import Path

from lagent.model import ReActPolicy, count_parameters

ROOT = Path(__file__).resolve().parents[1]


def _source_lines() -> int:
    return sum(len(p.read_text(encoding="utf-8").splitlines())
               for p in sorted((ROOT / "src").rglob("*.py")))


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_line_count_claim_matches_the_source():
    m = re.search(r"a ~(\d+)-line", _readme())
    assert m, "README no longer states its source size; drop or restore the claim"
    claimed = int(m.group(1))
    actual = _source_lines()
    assert abs(claimed - actual) <= 50, (
        f"README says ~{claimed} lines, src/ has {actual}; update the claim")


def test_readme_parameter_count_matches_the_policy_it_describes():
    """Every "~107k-parameter" mention, including the one inside the generated block."""
    actual = count_parameters(ReActPolicy())
    claimed = {int(m.group(1)) for m in
               re.finditer(r"~(\d+)k-parameter", _readme())}
    assert claimed, "the README no longer states the policy size"
    for k in claimed:
        assert abs(k * 1000 - actual) / actual <= 0.02, (
            f"README says ~{k}k parameters, ReActPolicy() has {actual:,}")

