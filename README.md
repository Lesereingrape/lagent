# lagent — a local-first ReAct tool-use agent, and a measured scaffold ablation

**lagent** is a ~650-line, CPU-only ReAct agent: a tiny behaviour-cloned transformer
drives a tool-using loop on a *verifiable* multi-hop graph task. The interesting part
is not that it works - it is that we isolate **how much of the success comes from the
agent's scaffold (its observation memory and constrained decoding) versus the raw
model**, by running the *identical* trained weights under four harnesses on the *same*
held-out tasks. That is the piece you can actually control when you point the same
framework at a real local LLaMA.

![ci](https://github.com/Lesereingrape/lagent/actions/workflows/ci.yml/badge.svg)

## The setup

- **Verifiable environment:** a 16-node attributed graph. To answer "which node has the
  highest value, one (top-1) or two (top-2) neighbourhood hops from the start?", the
  agent must *compose* three tools - `neighbors(node)`, `max_value(list)`,
  `finish(node)`. A single direct guess can never do it. Gold answers come from running
  the correct program, so the verifier is ground truth, independent of the policy.
- **The policy:** a ~110k-parameter decoder-only transformer, behaviour-cloned from
  expert ReAct traces. Training puts loss **only** on the two contentful tokens of each
  step - the *tool* and its *argument* - because the harness owns the structure markers.
  Fresh random graphs every step, so it learns the reusable procedure, not one world.
- **Four scaffolds, same weights:** `react` (full), `no_scratchpad` (tool results are
  withheld from the context), `no_constrain` (the model may emit any token, so illegal
  actions become possible), and `direct` (single-shot, no tools at all). The success gap
  between them is a clean *causal* read on what the harness contributes.

## Why it is trustworthy

The controller is fixed across scaffolds, so only the harness changes and the accuracy
gap is attributable to it - not to a different model or a lucky seed. Every solve is
graded by an independent gold program, and a full `gold` run fixes the solvability
ceiling. The prose in the results block is rendered mechanically from the committed
JSON by `experiments/make_report.py`, and a CI test asserts the README equals it
byte-for-byte, so numbers, rankings and superlatives cannot drift or be overclaimed -
including reporting a *null* (constrained decoding) honestly instead of inflating it.

## Quickstart

```bash
pip install -e .                 # torch is the only runtime dependency
python -m lagent.cli demo        # one seeded run: print a decoded trace + scaffold table
python experiments/run_study.py  # full 3-seed ablation -> results/agent.json
python experiments/make_report.py --write  # splice the block into README.md
```

## Results

<!-- RESULTS:START -->
*Every figure below is produced by `experiments/run_study.py` on CPU and committed as [`results/agent.json`](results/agent.json); the tables are rendered by `experiments/make_report.py`. For each seed a single policy is behaviour-cloned once, then dropped into four different agent *scaffolds* on the *same* 60 held-out tasks - the weights are identical, only the harness changes. Mean over 3 seeds.*

- task: find the highest-value node one or two neighbourhood-hops from a start, by composing `neighbors` / `max_value` / `finish` tools
- the verifier runs the *correct program* independently of the policy, so a solve is ground truth, never a model's opinion
- tool-use ceiling (gold program, every task solvable) = **1.000**
- measured under: Python 3.13.7 on Windows-11-10.0.26200-SP0, torch 2.14.0+cpu, 8 CPU threads, cpu - a rerun inside that environment reproduces this file field for field; elsewhere the thread count changes the float reduction order and the numbers move slightly

### Headline: what the harness, not the model, contributes

| scaffold | solve rate | std | invalid-action rate | mean steps |
|---|---:|---:|---:|---:|
| ReAct (scratchpad + constrained) | 1.000 | 0.000 | 0.000 | 3.96 |
| No constrained decoding | 1.000 | 0.000 | 0.000 | 3.96 |
| Single-shot (no tools) | 0.022 | 0.025 | 0.000 | 1.00 |
| No observation scratchpad | 0.017 | 0.017 | 0.000 | 5.43 |

With its observation scratchpad the identical weights reach **1.000** - matching the gold ceiling of 1.000. Remove *only* the scratchpad (tool results no longer written back into the context) and the same policy collapses to **0.017**. That 0.98-point gap is the whole result, and it is a property of the *harness*: the model that produced it never changed. A two-hop question is unanswerable unless the first tool's return value is still visible when the second argument is grounded, so the memory loop is load-bearing machinery, not decoration.

### Two honest nulls we report rather than spin

- **Constrained decoding is free insurance here, not an accuracy win.** Removing the legal-action masks gives 1.000 - the same as ReAct - with a measured invalid-action rate of 0.000. Under greedy decoding the argmax already lands on legal tools/arguments, so the safety net has nothing to catch on this toy task. We show the real number instead of inventing a gap; the mask earns its keep on a stochastic or larger policy, which we do not claim to have measured.
- **A single-shot guess scores 0.022.** Ask the same weights to name the answer with no tools and they cannot: the value-max lives in the graph, not the weights. This is the control that proves ReAct's score comes from *using* the environment, not from memorising answers.

### Difficulty is genuine, and the agent clears both hops

ReAct solves top-1 (one hop) at 1.000 and top-2 (two hops, requires chaining two tool calls) at 1.000. The two-hop case is where a naive agent falls over, and it is exactly what the scratchpad protects.

### Honest limitations

- Deliberately toy: a 16-node attributed graph, three tools and a ~110k-parameter policy. Real tool-use agents face open-ended natural-language arguments; the *measurement method* (identical weights, only the scaffold varies, ground-truth verifier) is what transfers, not this task.
- Greedy decoding makes constrained and unconstrained ReAct coincide, so this study cannot show a decoding-safety benefit; it isolates the memory ablation cleanly and reports the constraint ablation as the null it is.
- We behaviour-clone from gold traces, so 'the policy learned the procedure' is measured, not proven to generalise to distributions outside the training graphs.
<!-- RESULTS:END -->

## What you can do with it

Swap `ReActPolicy` for a local LLaMA behind the same `Scaffold` interface and the study
becomes a diagnostic for *your* agent: the observation-memory ablation tells you whether
your win is the model or the loop, and the ground-truth verifier keeps it honest.

## Reproducing

Nothing in the `Results` block is typed by hand:
`tests/test_readme_matches_results.py` asserts the README equals
`make_report.build(results/agent.json)`, `tests/test_readme_size_claims.py` checks
the hand-written size claims in the first paragraph against `src/`, and
`tests/test_artifact_is_internally_consistent.py` recomputes every mean, std and
ceiling in `summary` from the raw `per_seed` traces with the same arithmetic the study
uses - so a summary cell that was edited, rounded differently, or regenerated from a
different trace than it claims is a test failure, not a reviewer's job. The artifact records
the environment it was measured under (Python, torch build, CPU thread count), and the
block quotes it — a rerun reproduces it field for field *inside* that environment,
because float reduction order over a batch follows the thread count. Check that rather
than trusting the sentence:

```bash
python experiments/run_study.py --out /tmp/again.json   # leaves results/ untouched
```

Then diff the two JSONs; the only field allowed to differ is `runtime_sec`. That is the
diff we ran: the rerun reproduced the artifact field for field - the four scaffold
tables, the template breakdown, the gold ceiling and all three raw per-seed traces -
and the only number that moved was `runtime_sec` (143.1s against the published 176.9s,
which is the machine being less busy rather than a different result).
