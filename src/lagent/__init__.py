"""lagent - a local-first ReAct agent framework and a measured scaffold study.

A behaviour-cloned tiny-transformer controller drives a tool-using agent on a
verifiable multi-hop graph task.  The committed study isolates how much of the
agent's success comes from its *scaffold* (observation memory and constrained
decoding) versus the raw policy, because that is what you can actually control
when you point the same framework at a real local LLaMA.
"""

from __future__ import annotations

__all__ = ["agent", "cli", "env", "model", "study", "traces", "train", "vocab"]
