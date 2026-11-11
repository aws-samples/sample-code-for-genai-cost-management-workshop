# Medium Effort

Cost-and-latency optimization levers that are architectural moves on well-trodden paths - a step up from the one-line tweaks of the [low-effort](../01-low-effort/) tier. Each requires some application or configuration change and modest validation before rollout.

## Overview

Each script is a standalone, runnable demonstration of one lever, matching the repo's Python sample style. They are independent - lift any one into your own application without the others. Every script prints the token, cost, or behavior signal that makes the lever's payoff visible.

## Models Used

The Claude samples use the workshop's allowed models via Global cross-region inference profiles; `02-5_service_tiers.py` uses the direct `openai.gpt-oss-120b` model ID because Flex is not served by Claude:

| Alias | Model ID |
|-------|----------|
| Haiku 4.5 | `global.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Claude 5 Sonnet | `global.anthropic.claude-sonnet-5` |
| OpenAI gpt-oss-120b | `openai.gpt-oss-120b-1:0` |

## Scripts

| Script | Lever | What it demonstrates |
|--------|-------|----------------------|
| `02-1_llm_routing.py` | LLM Routing | Labels each query simple/complex with a tiny Haiku classifier to route simple lookups to Haiku and complex ones to Claude 5 Sonnet, and measures the classifier's own latency and Bedrock cost |
| `02-2_rag_indexing.py` | RAG / Indexing | Sends the relevant slice of a catalog instead of the whole thing - naive full-catalog prompt vs loading only the relevant category, comparing input tokens |
| `02-3_bedrock_guardrails.py` | Bedrock Guardrails | Creates a guardrail (denied topic + PII), applies it inline on a Converse call and standalone via `ApplyGuardrail`, then deletes it - blocked traffic never pays for inference |
| `02-4_remove_extra_model_calls.py` | Remove Extra Model Calls | Processes support tickets two ways: five model calls per ticket (extract ID, compute total with tax, normalize date, check refund window, draft reply) vs Python for the four deterministic steps plus one model call for the reply, comparing model calls, tokens, latency, cost, and how many deterministic answers were correct |
| `02-5_service_tiers.py` | Service Tiers (Flex) | Runs the same summarization task on Standard (`default`) vs Flex tier on `openai.gpt-oss-120b`, comparing tokens and latency via the OpenAI Chat Completions API, which returns the `service_tier` that served each request; Flex trades latency for a pricing discount |
| `02-6_filter_tool_output.py` | Filter Tool Output | Sends the output of a simulated CI step (pip install plus a pytest run with one failure) to the model raw, through a hand-written Python filter, and through the open-source `rtk` CLI (installed automatically if missing), comparing input tokens, cost, and whether the answer still finds the root cause |

Run any script directly:

```bash
python 02-1_llm_routing.py
python 02-2_rag_indexing.py
python 02-3_bedrock_guardrails.py
python 02-4_remove_extra_model_calls.py
python 02-5_service_tiers.py
python 02-6_filter_tool_output.py
```

## The Levers

### 1. LLM Routing

Query complexity has a long tail: only ~20% of requests genuinely need a workhorse model; the rest are lookups a smaller model answers just as well at a fraction of the cost. A classifier labels each request, then routes it to the cheapest model that can handle it. Keep the classifier cheap and deterministic (low `maxTokens`, a stop sequence, cache its static prompt), and make sure the cheap path takes the majority of traffic. Confirm routed-down quality on your own eval set.

The sample implements the routing *decision* as an **LLM classifier** (`run_llm_classification`): a tiny, deterministic Haiku Converse round-trip. Few-shot examples pin the output to one word and a stop sequence ends generation right after it, so it costs only a few output tokens per request. It prints each query's label and routed model, then the classifier's total latency, token counts, and Bedrock spend.

### 2. RAG / Indexing

Stuffing an entire knowledge source into the prompt is expensive and counterproductive - past a certain size accuracy drops as the model underweights information buried mid-prompt ("lost in the middle"). Retrieval inverts it: index the corpus once, then per query pull in only the relevant slice. This sample teaches the concept **without a vector database** - a fake 10-category, 100-product catalog stands in for the corpus, one file per category is the index, and picking the file for the question's category stands in for retrieval. The result: the same answer at roughly 89% fewer input tokens. A real system swaps file-selection for embedding search over chunks (e.g. Amazon Bedrock Knowledge Bases), adds metadata filtering and reranking, and keeps retrieved chunks *after* any `cachePoint`.

### 3. Bedrock Guardrails

Guardrails are sold as safety, but they are equally a cost lever: a request blocked at the input layer never reaches the model, so you don't pay generation tokens on injection probes, jailbreaks, or off-topic traffic. The sample shows both application modes - inline on a Converse call, and the standalone `ApplyGuardrail` API that screens text without invoking a model, so you can place the check before retrieval, routing, or tool calls. Pricing is per ~1,000-character text unit, billed per policy evaluated, so blocking off-topic traffic saves the full request cost.

### 4. Remove Extra Model Calls

The cheapest model call is the one you never make. Pipelines accumulate "one more prompt" per step, but extracting an ID, adding up line items, converting a date, or checking a policy window are fixed rules that a few lines of Python do instantly, for free, and exactly. A model call for the same step costs tokens and latency, and can get arithmetic or date math wrong, and a wrong answer flows into every later step. Use code for deterministic steps and keep the model for reasoning and semantics: understanding what the customer wants and writing the reply.

The sample runs five support tickets through both designs. The all-model design makes five calls per ticket, and each answer feeds the reply prompt. The hybrid design computes the order ID, total with tax, ISO date, and refund eligibility in Python (regex, `Decimal`, `datetime`, a lookup table) and makes one model call to draft the reply from those verified facts. For each ticket the sample prints the code's answer and the model's answer on adjacent lines and flags any difference. Order dates are set relative to today, so the refund-window tickets stay the same mix (eligible, ineligible, and one day either side of the 30-day limit) whenever you run it. The comparison table shows model calls, tokens, latency, cost, and how many of the 20 deterministic answers were correct. Model mistakes vary from run to run, but the call count and the cost gap do not. In a real pipeline, fall back to the model only for inputs your parser cannot handle, and confirm the output is unchanged on an eval set.

### 5. Service Tiers (Flex)

Amazon Bedrock offers four service tiers: Reserved, Priority, Standard (the default), and Flex. For latency-tolerant, non-interactive workloads - model evaluations, content summarization, labeling/annotation, multistep agentic workflows - the Flex tier earns a pricing discount versus the Standard on-demand price in exchange for lower priority: during high demand, Flex requests get longer and more variable latency. The code change is small - set `service_tier` to `flex` on the request (the sample uses the OpenAI Chat Completions API on `bedrock-runtime`; Converse accepts it too, via `additionalModelRequestFields`) - but adopting it is a workload-routing decision: you identify which traffic can tolerate the latency, route it to Flex, and validate the tradeoff. Your on-demand quota is shared across the priority, default, and flex tiers, while Reserved capacity is separate. The discount is a billing effect you confirm in Cost Explorer, not a per-request number returned by the API; and the tier that served each request depends on the API you call. Chat Completions returns a top-level `service_tier` field in the response, which is why the sample uses it; Converse returns no tier field at all (no field, no header). For fleet-wide confirmation use the `ResolvedServiceTier` dimension in CloudWatch (or AWS CloudTrail events).

Flex is supported only by a specific model set - OpenAI gpt-oss (20b/120b), DeepSeek V3.1, Qwen3 variants, and Amazon Nova Pro/Premier - NOT Anthropic Claude and NOT the lighter Nova models, which is why this sample runs on `openai.gpt-oss-120b` instead of the workshop Claude models. Availability is also region-gated. Check "Models at a glance" for the current, authoritative per-model and per-region supported-tier list: https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html

### 6. Filter Tool Output

When an agent runs a command, the raw output usually goes straight back to the model, and most of it is noise for the question at hand: install progress bars, ANSI color codes, INFO and DEBUG log lines, the same deprecation warning repeated for every test, and a row of dots per passing test. You pay input tokens for all of it, and in an agent loop you pay again on every later turn because the history is resent. Filter the tool result in code before it reaches the model: strip console noise, collapse repeated warnings, drop logs and package output, and keep the failure, its assertion message, and the final summary line.

The sample builds the output of a simulated CI step (a `pip install` followed by a `pytest` run with one failing test) so every run is identical and needs no network or test project. It asks Haiku 4.5 the same question with three versions of that output: raw, a short hand-written Python filter tuned to the tool, and the open-source [rtk](https://github.com/rtk-ai/rtk) CLI (`rtk pipe -f pytest`), which ships ready-made filters for common developer commands. The comparison table shows characters, input tokens, cost, and whether the answer still finds the root cause. The root cause here is a single WARNING log line (the discount threshold is exclusive), so a general-purpose filter that drops log lines saves the most tokens but can lose the clue. Filtering should remove noise, not signal, so check the answer after changing a filter (the same eval-set habit as the low-effort evaluation sample) and keep the full output somewhere the agent can fetch it on demand.

## Prerequisites

- Python 3.12+
- IAM credentials with the relevant permissions:
  - Routing, RAG, removing extra model calls & filtering tool output: `bedrock-runtime:Converse`
  - Guardrails: `bedrock:CreateGuardrail`, `bedrock:DeleteGuardrail`, `bedrock-runtime:Converse`, `bedrock-runtime:ApplyGuardrail`
- Access to Claude Haiku 4.5 and Claude 5 Sonnet on Amazon Bedrock, plus `openai.gpt-oss-120b` in a Flex-supporting region for `02-5_service_tiers.py` (which authenticates the OpenAI SDK with a short-lived Bedrock API token minted from your IAM credentials via `aws-bedrock-token-generator`, already in `requirements.txt`)
- Dependencies installed via `pip install -r requirements.txt` from the repository root
- `AWS_REGION` set (defaults to `us-east-1` if unset)
- OPTIONAL, for the rtk row in `02-6_filter_tool_output.py`: the `rtk` CLI from [rtk-ai/rtk](https://github.com/rtk-ai/rtk), a single Rust binary (install the rtk-ai project, not another tool that shares the name). If `rtk` is not on your `PATH`, the sample prints the install command below and runs it for you (it needs network access to GitHub), then continues; if the install fails it skips the rtk row. To install it yourself first:

  ```bash
  curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh
  ```

  The script downloads a checksum-verified binary into `~/.local/bin`. At the top of the sample, `RTK_INSTALL` is `"auto"` by default; set it to `"ask"` to be asked first, or `"never"` to skip the install.

## Model Compatibility Notes

`temperature` is deprecated on Claude 5 Sonnet - supplying it returns a `ValidationException`. The samples omit `temperature` on Sonnet 5 calls and keep it only for Haiku 4.5.

Flex tier is supported only by a specific model set - OpenAI gpt-oss, DeepSeek V3.1, Qwen3, and Nova Pro/Premier - not Anthropic Claude or the lighter Nova models, which is why `02-5` runs on `openai.gpt-oss-120b`. Availability is region-gated, so `02-5` guards the Flex call defensively. `gpt-oss-120b` is a reasoning model: on Chat Completions its reasoning arrives inline in `<reasoning>...</reasoning>` tags before the answer, and the sample strips it.
