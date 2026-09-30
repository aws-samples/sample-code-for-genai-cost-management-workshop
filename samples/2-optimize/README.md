# Optimize

Sample code for the **Optimize** part of the workshop: reducing generative AI spend on Amazon Bedrock once you can see where it goes.

The [1-track](../1-track/) samples answer *who is spending what, and where*. The optimize samples answer *how do we spend less* without sacrificing quality, using Bedrock's built-in cost-reduction levers.

## Effort Tiers

The samples are organized by how much work each lever takes to adopt. Start LOW, earn HIGH - apply and measure the cheaper levers before reaching for the complex ones.

| Tier | Folder | What it covers |
|------|--------|----------------|
| Low effort | [01-low-effort](01-low-effort/) | One-line changes, parameter tweaks, and short prompt refactors - no new infrastructure |
| Medium effort | [02-medium-effort](02-medium-effort/) | Architectural moves on well-trodden paths - some app or config change plus modest validation |
| High effort | [03-high-effort](03-high-effort/) | Compound-AI patterns that add real complexity, earned on evidence (coming soon) |

## Low Effort Samples

The [01-low-effort](01-low-effort/) tier has runnable samples today, one per lever:

| Script | Lever | What it demonstrates |
|--------|-------|----------------------|
| [01-1_model_selection.py](01-low-effort/01-1_model_selection.py) | Model Selection | Same task on Haiku vs Claude 5 Sonnet, comparing tokens, latency, and per-call cost |
| [01-2_prompt_design.py](01-low-effort/01-2_prompt_design.py) | Prompt Design | Clear instructions, few-shot examples, and three structured-output methods |
| [01-3_parameter_tuning.py](01-low-effort/01-3_parameter_tuning.py) | Parameter Tuning | `max_tokens` TPM reservation, `stop_sequences`, and `temperature` |
| [01-4_prompt_caching.py](01-low-effort/01-4_prompt_caching.py) | Prompt Caching | `cachePoint` on a large static prefix - cache write then cache read at ~0.1x |
| [01-5_adaptive_thinking.py](01-low-effort/01-5_adaptive_thinking.py) | Adaptive Thinking | Sweeps `effort` levels on Claude 5 Opus, showing reasoning-token and latency scaling |

See the [low-effort README](01-low-effort/README.md) for details on each lever and model-compatibility notes.

## Medium Effort Samples

The [02-medium-effort](02-medium-effort/) tier has runnable samples today, one per lever:

| Script | Lever | What it demonstrates |
|--------|-------|----------------------|
| [02-1_llm_routing.py](02-medium-effort/02-1_llm_routing.py) | LLM Routing | A tiny Haiku classifier routes simple lookups to Haiku and complex queries to Claude 5 Sonnet |
| [02-2_bedrock_guardrails.py](02-medium-effort/02-2_bedrock_guardrails.py) | Bedrock Guardrails | Inline and standalone `ApplyGuardrail` - blocked traffic never pays for inference |
| [02-3_rag_indexing.py](02-medium-effort/02-3_rag_indexing.py) | RAG / Indexing | Send only the relevant catalog slice instead of the whole corpus (no vector DB), comparing input tokens |
| [02-4_batch_inference.py](02-medium-effort/02-4_batch_inference.py) | Batch Inference | A Converse-format JSONL batch job at 50% of the on-demand price |

See the [medium-effort README](02-medium-effort/README.md) for details on each lever, batch-job setup, and model-compatibility notes.

## Models Used

The samples use the workshop's allowed models via Global cross-region inference profiles:

| Alias | Model ID |
|-------|----------|
| Haiku 4.5 | `global.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Claude 5 Sonnet | `global.anthropic.claude-sonnet-5` |
| Claude 5 Opus | `global.anthropic.claude-opus-5` |

## Setup

See the [main README](../../README.md) for environment setup, virtual environment creation, and dependency installation.
