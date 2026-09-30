# High Effort

Cost-and-latency optimization levers for when the lower tiers aren't enough - compound-AI patterns that add real complexity, earned on evidence. Reach for them only after the [low-effort](../01-low-effort/) and [medium-effort](../02-medium-effort/) levers are applied, measured, and still short of target.

## Overview

Each script is a standalone, runnable demonstration of one lever, matching the repo's Python sample style. They are implemented with plain boto3 (no agent framework) so they run with the repo's base dependencies, and each prints the token signal that makes the lever's payoff visible.

## Models Used

The samples use the workshop's allowed models via Global cross-region inference profiles:

| Alias | Model ID |
|-------|----------|
| Haiku 4.5 | `global.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Claude 5 Opus | `global.anthropic.claude-opus-5` |

## Scripts

| Script | Lever | What it demonstrates |
|--------|-------|----------------------|
| `03-1_harness_engineering.py` | Harness Engineering | A minimal but real agent loop (call, parse, run tools, check stop) with a turn-budget guardrail, plus a lean-vs-bloated context comparison showing tokens re-sent every turn |
| `03-2_sub_agent_delegation.py` | Sub-Agent Delegation | An orchestrator-worker split - a cheap Haiku worker digests a large document, and the expensive Claude 5 Opus lead reasons over only the compact summary - vs a single-agent baseline |

Run any script directly:

```bash
python 03-1_harness_engineering.py
python 03-2_sub_agent_delegation.py
```

## The Levers

### 1. Harness Engineering

The model is one component; the harness is the deterministic scaffolding around it - tools, context curation, the control loop, retries, a turn budget, evals, and code-level guardrails. An agent is just "an LLM using tools in a loop," and the loop is where cost and latency live because two numbers dominate the bill: **tokens per turn** (you re-send the growing context every turn) and **turns per task** (each turn is a full round-trip). The sample builds a working loop with a turn budget so a runaway agent can't quietly multiply your bill, then shows that a bloated system prompt is a surcharge paid on *every* turn. Trim the context, cache the stable prefix (see [01-4](../01-low-effort/01-4_prompt_caching.py)), and consolidate tool calls to cut round-trips.

### 2. Sub-Agent Delegation

A single agent carries every tool result and intermediate step in one context window that is re-sent every turn. Delegation breaks this with an orchestrator-worker split: a **lead** plans and answers; a **worker** does the token-heavy subtask (search, extract, summarize) in its own isolated context and returns only a compact result. The cost lever is the model choice - route the worker to a cheap model and keep the strong model on the lead, where it sees only the question and the compressed result. The sample shows the expensive lead using ~96% fewer input tokens than a single-agent baseline that ingests the whole document itself. Mind the orchestration overhead: every delegation is a separate call, so make sure the lead's context savings exceed it - and a worker with bad tool descriptions is just bad, faster, so fix the harness (03-1) first.

## Covered as concept / reference (no sample here)

Two levers from the source playbook are documented as concepts rather than runnable samples in this repo:

- **GEPA / DSPy** - open-source programmatic prompt optimization. A real `compile()` needs a training set and runs for minutes to hours, so it is a reference pattern, not a live sample. The managed counterpart is Bedrock Advanced Prompt Optimization.
- **Tool Search via MCP Gateway** - AgentCore Gateway semantic tool search. It depends on provisioned infrastructure (a gateway, Cognito, Lambda targets), so it belongs with a full deployment rather than a standalone script.

## Prerequisites

- Python 3.12+
- IAM credentials with `bedrock-runtime:Converse` permission
- Access to Claude Haiku 4.5 and Claude 5 Opus on Amazon Bedrock
- Dependencies installed via `pip install -r requirements.txt` from the repository root
- `AWS_REGION` set (defaults to `us-east-1` if unset)

## Model Compatibility Note

Claude 5 Opus has **adaptive thinking on by default**, so its Converse response leads with a `reasoningContent` block before the `text` block - the answer is not `content[0]`. The sub-agent sample extracts the text block explicitly and gives the lead a generous `maxTokens` so reasoning plus the final answer fit within the budget.
