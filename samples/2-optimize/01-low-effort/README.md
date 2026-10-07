# Low Effort

Cost-and-latency optimization levers that require little or no application change and carry low risk. These are the first levers to pull, in order: start here before reaching for medium- or high-effort techniques.

## Overview

Each script is a standalone, runnable demonstration of one lever. They are independent - lift any one into your own application without the others. Every script prints the token, latency, or cost signal that makes the lever's payoff visible, so you can see the effect rather than take it on faith.

## Models Used

All samples use the workshop's allowed models via Global cross-region inference profiles:

| Alias | Model ID |
|-------|----------|
| Haiku 4.5 | `global.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Claude 5 Sonnet | `global.anthropic.claude-sonnet-5` |
| Claude 5 Opus | `global.anthropic.claude-opus-5` |

## Scripts

| Script | Lever | What it demonstrates |
|--------|-------|----------------------|
| `01-1_model_selection.py` | Model Selection | Runs the same task on Haiku and Claude 5 Sonnet, comparing input/output tokens, latency, and per-call cost side by side |
| `01-2_prompt_design.py` | Prompt Design | Clear vs vague instructions, zero-shot vs few-shot consistency, and three structured-output methods (prompt-based, tool use, native JSON schema) |
| `01-3_parameter_tuning.py` | Parameter Tuning | `max_tokens` (the TPM quota reserved up front), `stop_sequences` (early termination), and `temperature` (determinism vs variation) |
| `01-4_prompt_caching.py` | Prompt Caching | Caches a large static prefix with a `cachePoint` marker, showing a cache write on the first call and a cache read (billed ~0.1x) on the second |
| `01-5_adaptive_thinking.py` | Adaptive Thinking | Sweeps `effort` levels (low / medium / high) on Claude 5 Opus, showing how reasoning (output) tokens and latency scale with effort |
| `01-6_service_tiers.py` | Service Tiers (Flex) | Runs the same summarization task on Standard (`default`) vs Flex tier, comparing latency and tokens and showing the resolved tier; Flex trades latency for a pricing discount |

Run any script directly:

```bash
python 01-1_model_selection.py
python 01-2_prompt_design.py
python 01-3_parameter_tuning.py
python 01-4_prompt_caching.py
python 01-5_adaptive_thinking.py
python 01-6_service_tiers.py
```

## The Levers

### 1. Model Selection

The highest-leverage cost decision is *which* model runs each call - the price gap between Claude tiers is roughly 5x per tier. Start on the cheapest tier (Haiku), and escalate only when a scored eval shows it falls short on *your* task. Re-benchmark every model generation; "needs the big model" decisions go stale.

### 2. Prompt Design

The most under-used lever. Most production prompts carry 30-50% wasted tokens (hedging, boilerplate, vague instructions) that cost money on every call and degrade quality. Lead with the task, use few-shot examples to pin output format, and when downstream code consumes the result, *constrain* the output rather than *asking* for a shape.

### 3. Parameter Tuning

Four inference-config values that ship in minutes. `max_tokens` is load-bearing: on Bedrock it sets the TPM quota reserved up front (output burns at 5x for Claude 3.7+), so an oversized value throttles concurrency under load even though per-call cost is unchanged. Use `temperature=0` for classification/extraction and `stop_sequences` for a free latency win when a real terminator exists.

### 4. Prompt Caching

Requests that resend the same system prompt, tool definitions, or reference document pay full input price for tokens that never change. A `cachePoint` marker pays full price once (the write, ~1.25x), then ~0.1x for every read - break-even is around 3 reads. Put static content first and dynamic content last, and watch cache hit rate like an SLO (target >= 70%).

### 5. Adaptive Thinking

Set an `effort` level and the model decides, per request, how much to reason before answering. It is both a quality lever (think when it helps) and a cost/latency lever (don't pay when it doesn't). The practical pattern is to step down from the default `high` once an eval shows a lower level holds quality.

### 6. Service Tiers (Flex)

Amazon Bedrock offers four service tiers: Reserved, Priority, Standard (the default), and Flex. For latency-tolerant workloads - model evaluations, content summarization, agentic/batch jobs - the Flex tier earns a pricing discount versus the Standard on-demand price. It is a one-parameter change: set `service_tier` to `flex` (passed through Converse via `additionalModelRequestFields`). Your on-demand quota is shared across the priority, default, and flex tiers, while Reserved capacity is separate. The discount is a billing effect you confirm in Cost Explorer; the request-time tradeoff is higher or more variable latency. Monitor the `ResolvedServiceTier` dimension in CloudWatch to see the tier that actually served each request, since it can differ from the one you requested.

## Model Compatibility Notes

Two behaviors of the Claude 5 models affect these samples (and are worth knowing before you reuse the code):

- **`temperature` is deprecated on Claude 5 Sonnet and Opus.** Supplying it returns `ValidationException: temperature is deprecated for this model`. The samples omit `temperature` for those models and keep it only for Haiku 4.5 (which is why the parameter-tuning demo runs on Haiku).
- **Native structured output is not accepted by Claude 5 Sonnet.** The `outputConfig` / JSON-schema path returns `ValidationException: output_config.format: Extra inputs are not permitted` on Sonnet 5, so `01-2` runs that specific demo on Haiku 4.5. When you need a schema guarantee on a thinking-capable Claude 5 model, use tool use (forced tool call) instead.
- **Flex tier availability varies by model and region.** Not every model/region supports the Flex `service_tier`, so `01-6` guards the Flex call and falls back gracefully if it is rejected. Check "Models at a glance" for each model's supported tiers.

## Prerequisites

- Python 3.12+
- IAM credentials with `bedrock-runtime:Converse` permission
- Access to Claude Haiku 4.5, Claude 5 Sonnet, and Claude 5 Opus on Amazon Bedrock
- Dependencies installed via `pip install -r requirements.txt` from the repository root
- `AWS_REGION` set (defaults to `us-east-1` if unset)
