"""The README is only allowed to describe the code that is actually in the repo.

The results block is byte-pinned against ``results/agent.json``; the size claim in
the first paragraph is the other hand-written number here, and a "~650-line" line
that quietly becomes 900 is the same class of drift.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source_lines() -> int:
    return sum(len(p.read_text(encoding="utf-8").splitlines())
               for p in sorted((ROOT / "src").rglob("*.py")))


def test_readme_line_count_claim_matches_the_source():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"a ~(\d+)-line", readme)
    assert m, "README no longer states its source size; drop or restore the claim"
    claimed = int(m.group(1))
    actual = _source_lines()
    assert abs(claimed - actual) <= 50, (
        f"README says ~{claimed} lines, src/ has {actual}; update the claim")
