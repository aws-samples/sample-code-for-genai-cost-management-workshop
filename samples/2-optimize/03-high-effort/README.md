# High Effort

Cost-and-latency optimization levers for when the lower tiers aren't enough - changes to architecture or state: more components, orchestration, persistent state, security boundaries, or operational ownership. Reach for them only after the [low-effort](../01-low-effort/) and [medium-effort](../02-medium-effort/) levers are applied, measured, and still short of target.

## Overview

The tier has six levers (13-18 in the workshop deck). File numbers follow the deck order, so a gap in the numbering is a lever that does not have a runnable sample yet. Each script is a standalone, runnable demonstration of one lever, matching the repo's Python sample style. Each prints a comparison table at the end, so the lever's payoff is visible in tokens, calls, and cost.

## Models Used

The samples use the workshop's allowed models via Global cross-region inference profiles:

| Alias | Model ID |
|-------|----------|
| Haiku 4.5 | `global.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Claude 5 Sonnet | `global.anthropic.claude-sonnet-5` |

## Scripts

| Script | Lever | What it demonstrates |
|--------|-------|----------------------|
| `03-2_dynamic_tool_infrastructure.py` | 14 - Dynamic Tool Infrastructure | The same Strands agent answers one query against a local MCP server twice - once loading all 101 tools, once with a tool filter that loads only the one it needs - showing ~93% fewer input tokens for the same answer |
| `03-4_multi_agent_orchestration.py` | 16 - Multi-Agent Orchestration | The same question over a large document as a single Claude 5 Sonnet agent, an uncapped split (one Haiku worker per section), and a capped split (a fixed number of workers), comparing agents, tokens, latency, and total cost |
| `03-6_human_escalation_path.py` | 18 - Human Escalation Path | A support agent runs an eval set, N is derived from the turns its resolved tasks needed, and tasks that exceed N are handed to a human with a note built by code, comparing turns, tokens, and cost against no escalation |

Run any script directly:

```bash
python 03-2_dynamic_tool_infrastructure.py
python 03-4_multi_agent_orchestration.py
python 03-6_human_escalation_path.py
```

### Not yet implemented

These levers from the deck do not have a sample yet:

| Slot | Lever | Idea |
|------|-------|------|
| 03-1 | 13 - Memory Storage Tiers | Do not keep all agent memory in fast storage; move inactive memory to lower-cost tiers |
| 03-3 | 15 - Long-Running Agent State | Save task state outside the context and resume from it instead of replaying the full history |
| 03-5 | 17 - Security and Permission Boundaries | Enforce tool access in AgentCore Gateway Policy, not in the prompt |

## The Levers

### Dynamic Tool Infrastructure (lever 14)

Every tool an agent can call - its name, description, and input schema - is sent to the model on every call and billed as input tokens. Plug an agent into a large MCP server and it pays for the whole catalog on every request, even when the task needs one tool; and because an agent makes at least two model calls per query (pick the tool, then answer), the unused catalog is paid for again and again. The sample starts a local MCP server (stdio, nothing deployed) that exposes 101 tools - one real `get_weather` and 100 fake business tools - and runs the same Strands agent on the same question twice: once with the **full catalog**, once with a **tool filter** on the MCP client (`tool_filters={"allowed": ["get_weather"]}`). The filtered run uses ~93% fewer input tokens and is ~12x cheaper per query for the same answer. The fix is configuration, not code: give each agent a focused job and load only the tools that job needs. When the catalog is too large or changes too often to filter by hand, move tool selection into the infrastructure with AgentCore Gateway semantic tool search, which selects tools per query. It depends on provisioned infrastructure (a gateway, Cognito, Lambda targets), so it belongs with a full deployment rather than a standalone script.

### Multi-Agent Orchestration (lever 16)

Every agent adds model calls and tokens: its own instructions, its own context, its own output. Split a task across agents only for **isolation** (a worker digests token-heavy material in its own context and returns a compact result, so the strong lead never re-reads it) or for **parallel work**. Then **cap the fan-out**. The sample runs the same question over a 40-section document three ways. The single Claude 5 Sonnet agent reads everything. The uncapped split spawns one Haiku worker per section, so the lead reads 40 partial results and the run pays for 41 agents. The capped split lets the lead ask for 40 workers but the harness spawns at most `MAX_WORKERS` (4), enforced in code rather than in the prompt. In one sample run the capped split cost about 0.44x the single agent and the uncapped split about 1.19x, so the isolation win only holds when the overhead stays small. Compare total cost across all agents, not just the lead's tokens.

### Human Escalation Path (lever 18)

An agent that cannot finish a task keeps looping, and every extra turn re-sends the whole growing context. A turn cap stops the loop; an escalation path also hands the case to a person with a note of what was already tried. The sample runs a support agent over a small eval set against a paged order history: lookups near the top need two or three turns, lookups further down need more, and orders that are not in the history keep the agent paging. It first measures turns on the eval set with a generous cap, sets **N** to the most turns any resolved task needed, then runs the set again and escalates at N. Too low an N escalates tasks the agent could have solved; too high an N pays for turns that never converge. The handoff note is built from the harness's own tool log, so escalating costs no extra model tokens. In one sample run N came out at 5 and the escalation policy cost about 0.77x the no-escalation run for the same resolved tasks.

## Prerequisites

- Python 3.12+
- IAM credentials with `bedrock-runtime:Converse` permission
- Access to Claude Haiku 4.5 and Claude 5 Sonnet on Amazon Bedrock
- Dependencies installed via `pip install -r requirements.txt` from the repository root
- `AWS_REGION` set (defaults to `us-east-1` if unset)

## Model Compatibility Note

Claude 5 Sonnet is **thinking-capable**, so its Converse response can lead with a `reasoningContent` block before the `text` block - the answer is not necessarily `content[0]`. The multi-agent orchestration sample extracts the text block explicitly and gives the lead a generous `maxTokens` so reasoning plus the final answer fit within the budget.
