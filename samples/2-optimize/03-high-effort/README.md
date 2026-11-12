# High Effort

Cost-and-latency optimization levers for when the lower tiers aren't enough - changes to architecture or state: more components, orchestration, persistent state, security boundaries, or operational ownership. Reach for them only after the [low-effort](../01-low-effort/) and [medium-effort](../02-medium-effort/) levers are applied, measured, and still short of target.

## Overview

The tier has six levers (13-18 in the workshop deck). File numbers follow the deck order, so each file number maps to one lever. Each script is a standalone, runnable demonstration of one lever, matching the repo's Python sample style. Each prints a comparison table at the end, so the lever's payoff is visible in tokens, calls, and cost.

## Models Used

The samples use the workshop's allowed models via Global cross-region inference profiles:

| Alias | Model ID |
|-------|----------|
| Haiku 4.5 | `global.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Claude 5 Sonnet | `global.anthropic.claude-sonnet-5` |

Most samples use Haiku 4.5. The memory storage tiers sample uses Claude 5 Sonnet, and the multi-agent orchestration sample uses both (Sonnet as the lead, Haiku as the workers).

## Scripts

| Script | Lever | What it demonstrates |
|--------|-------|----------------------|
| `03-1_memory_storage_tiers.py` | 13 - Memory Storage Tiers | A local simulation of hot and cold memory tiers: hot facts stay in the prompt and old facts are searched on demand, answering the same questions as keeping everything in the prompt with ~95% fewer input tokens |
| `03-2_dynamic_tool_infrastructure.py` | 14 - Dynamic Tool Infrastructure | The same Strands agent answers one query against a local MCP server twice - once loading all 101 tools, once with a tool filter that loads only the one it needs - showing ~93% fewer input tokens for the same answer |
| `03-3_long_running_agent_state.py` | 15 - Long-Running Agent State | A refund-review agent pauses to ask the user and waits hours, then finishes the task two ways: replaying the full conversation vs resuming from a small checkpoint saved outside the context, with ~87% fewer input tokens |
| `03-4_multi_agent_orchestration.py` | 16 - Multi-Agent Orchestration | The same question over a large document as a single Claude 5 Sonnet agent, an uncapped split (one Haiku worker per section), and a capped split (a fixed number of workers), comparing agents, tokens, latency, and total cost |
| `03-5_security_permission_boundaries.py` | 17 - Security and Permission Boundaries | A support agent runs five conversations (three attacks, two legitimate) with its permissions written in the prompt vs enforced by a policy checked on every tool call, comparing forbidden actions executed, legitimate work done, and cost |
| `03-6_human_escalation_path.py` | 18 - Human Escalation Path | A support agent runs an eval set, N is derived from the turns its resolved tasks needed, and tasks that exceed N are handed to a human with a note built by code, comparing turns, tokens, and cost against no escalation |

Run any script directly:

```bash
python 03-1_memory_storage_tiers.py
python 03-2_dynamic_tool_infrastructure.py
python 03-3_long_running_agent_state.py
python 03-4_multi_agent_orchestration.py
python 03-5_security_permission_boundaries.py
python 03-6_human_escalation_path.py
```

## The Levers

### Memory Storage Tiers (lever 13)

An agent that keeps all of its memory in the prompt pays for every stored fact on every call, even though most old facts have nothing to do with the question. Memory only grows, so the cost per call grows with it. Tiering matches memory to how often it is used: **hot** facts (used recently) stay in the prompt, and **cold** facts (older) stay out of it and are searched only when a question needs one, loading just the top matches. The sample is a local simulation on Claude 5 Sonnet: 174 remembered facts about one customer, an age cutoff as the tier policy, and a keyword match as the cold search. It asks three questions both ways. Both designs answered all three correctly, and the tiered design used about 95% fewer input tokens (about 0.07x the cost) because it sent 6 facts per call instead of 174. The gap widens as memory grows. In production the tiers would be separate stores, for example Amazon Bedrock AgentCore Memory (short-term events and long-term records) or a cache plus a cheaper database.

### Dynamic Tool Infrastructure (lever 14)

Every tool an agent can call - its name, description, and input schema - is sent to the model on every call and billed as input tokens. Plug an agent into a large MCP server and it pays for the whole catalog on every request, even when the task needs one tool; and because an agent makes at least two model calls per query (pick the tool, then answer), the unused catalog is paid for again and again. The sample starts a local MCP server (stdio, nothing deployed) that exposes 101 tools - one real `get_weather` and 100 fake business tools - and runs the same Strands agent on the same question twice: once with the **full catalog**, once with a **tool filter** on the MCP client (`tool_filters={"allowed": ["get_weather"]}`). The filtered run uses ~93% fewer input tokens and is ~12x cheaper per query for the same answer. The fix is configuration, not code: give each agent a focused job and load only the tools that job needs. When the catalog is too large or changes too often to filter by hand, move tool selection into the infrastructure with AgentCore Gateway semantic tool search, which selects tools per query. It depends on provisioned infrastructure (a gateway, Cognito, Lambda targets), so it belongs with a full deployment rather than a standalone script.

### Long-Running Agent State (lever 15)

Some tasks cannot finish in one sitting: the agent needs a person's approval or input, and the answer may take hours. While it waits nothing runs, so waiting costs no model tokens. The cost shows up when the agent comes back. If it replays the whole conversation, the prompt holds every tool result from the work already done, and that history is re-sent on every remaining turn. The better design saves the agent's state outside the context while it works (here, the decisions made so far), and resumes from that small checkpoint in a fresh context. The sample is a refund-review agent that reviews 20 requests, asks the user about the one over its approval limit, and waits (the wait is simulated, with no sleep). It then finishes the task both ways. Both reach the correct total, but resuming from the checkpoint used about 87% fewer input tokens (about 0.19x the cost) than replaying the history, because the first turn carried about 1,100 tokens instead of about 7,200. Saving state also means the process can shut down during the wait. In production the checkpoint would live in a store such as Amazon DynamoDB or Amazon S3, or in Amazon Bedrock AgentCore Memory.

### Multi-Agent Orchestration (lever 16)

Every agent adds model calls and tokens: its own instructions, its own context, its own output. Split a task across agents only for **isolation** (a worker digests token-heavy material in its own context and returns a compact result, so the strong lead never re-reads it) or for **parallel work**. Then **cap the fan-out**. The sample runs the same question over a 40-section document three ways. The single Claude 5 Sonnet agent reads everything. The uncapped split spawns one Haiku worker per section, so the lead reads 40 partial results and the run pays for 41 agents. The capped split lets the lead ask for 40 workers but the harness spawns at most `MAX_WORKERS` (4), enforced in code rather than in the prompt. In one sample run the capped split cost about 0.44x the single agent and the uncapped split about 1.19x, so the isolation win only holds when the overhead stays small. Compare total cost across all agents, not just the lead's tokens.

### Security and Permission Boundaries (lever 17)

An agent acts through tools, and some calls are expensive to get wrong: a refund that should not have been issued, an account that should not have been deleted. If the limits are only a sentence in the system prompt ("refunds above $100 and account deletions require manager approval"), they are a request to the model, and the model cannot verify anything. A user who says a manager approved it, or an instruction hidden in data the agent reads (a prompt injection) that says the same, looks just like a real approval. The better design enforces the limits in a policy at the gateway, outside the model: every tool call is checked before the tool runs, with a default of deny. **Recommendation: use [Policy in Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy.html).** Put your tools behind an AgentCore Gateway and attach a policy engine with rules written in Cedar; it checks every tool call outside your agent's code and denies by default. The sample is a local simulation: a Python function stands in for the policy engine and the Cedar text it prints is simplified. It runs five conversations through a tier-1 support agent on Haiku 4.5, three attacks and two legitimate requests. With the rules in the prompt, the model refused one attack and carried out two (a deletion after a claimed approval, and a deletion requested by a note in an order). With the policy, no forbidden action ran, the two legitimate requests still completed, and the permission sentence was gone from the prompt. Results for a prompt rule depend on the model and the wording and can change from run to run, while the policy gives the same answer every time. Approval should come from signing in as a manager, not from words in the conversation.

### Human Escalation Path (lever 18)

An agent that cannot finish a task keeps looping, and every extra turn re-sends the whole growing context. A turn cap stops the loop; an escalation path also hands the case to a person with a note of what was already tried. The sample is an order-lookup agent over a list that returns one order per call, so finding an order takes one turn per order. Orders near the top are found in a few turns, and orders that are not in the list keep the agent paging. The sample runs five eval tasks with a generous turn limit, sets **N** to the most turns any resolved task needed, then runs the tasks again and hands off to a human at N. Too low an N escalates tasks the agent could have solved; too high an N pays for turns that never converge. The handoff note is built by code from the harness's own count of lookups, so escalating costs no extra model tokens. In one sample run N came out at 4 and the escalation policy cost about 0.62x the no-escalation run for the same resolved tasks.

## Prerequisites

- Python 3.12+
- IAM credentials with `bedrock-runtime:Converse` permission (the dynamic tool infrastructure sample also uses `bedrock-runtime:ConverseStream`)
- Access to Claude Haiku 4.5 and Claude 5 Sonnet on Amazon Bedrock
- Dependencies installed via `pip install -r requirements.txt` from the repository root
- `AWS_REGION` set (defaults to `us-east-1` if unset)
- The long-running agent state sample writes a small checkpoint file to the system temp directory and deletes it when it finishes; the other samples write no files

## Model Compatibility Note

Claude 5 Sonnet is **thinking-capable**, so its Converse response can lead with a `reasoningContent` block before the `text` block - the answer is not necessarily `content[0]`. The multi-agent orchestration sample extracts the text block explicitly and gives the lead a generous `maxTokens` so reasoning plus the final answer fit within the budget.
