# Medium Effort

Cost-and-latency optimization levers that are architectural moves on well-trodden paths - a step up from the one-line tweaks of the [low-effort](../01-low-effort/) tier. Each requires some application or configuration change and modest validation before rollout.

## Overview

Each script is a standalone, runnable demonstration of one lever, matching the repo's Python sample style. They are independent - lift any one into your own application without the others. Every script prints the token, cost, or behavior signal that makes the lever's payoff visible.

## Models Used

The samples use the workshop's allowed models via Global cross-region inference profiles:

| Alias | Model ID |
|-------|----------|
| Haiku 4.5 | `global.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Claude 5 Sonnet | `global.anthropic.claude-sonnet-5` |

## Scripts

| Script | Lever | What it demonstrates |
|--------|-------|----------------------|
| `02-1_llm_routing.py` | LLM Routing | Labels each query simple/complex to route simple lookups to Haiku and complex ones to Claude 5 Sonnet - and compares two ways to make that call: a tiny Haiku LLM classifier vs the local Strands Decider decision model |
| `02-2_bedrock_guardrails.py` | Bedrock Guardrails | Creates a guardrail (denied topic + PII), applies it inline on a Converse call and standalone via `ApplyGuardrail`, then deletes it - blocked traffic never pays for inference |
| `02-3_rag_indexing.py` | RAG / Indexing | Sends the relevant slice of a catalog instead of the whole thing - naive full-catalog prompt vs loading only the relevant category, comparing input tokens |
| `02-4_batch_inference.py` | Batch Inference | Builds a Converse-format JSONL batch and submits a `create_model_invocation_job` (50% of on-demand price) - live when configured, demo otherwise |

Run any script directly:

```bash
python 02-1_llm_routing.py
python 02-2_bedrock_guardrails.py
python 02-3_rag_indexing.py
python 02-4_batch_inference.py
```

## The Levers

### 1. LLM Routing

Query complexity has a long tail: only ~20% of requests genuinely need a workhorse model; the rest are lookups a smaller model answers just as well at a fraction of the cost. A classifier labels each request, then routes it to the cheapest model that can handle it. Keep the classifier cheap and deterministic (low `maxTokens`, a stop sequence, cache its static prompt), and make sure the cheap path takes the majority of traffic. Confirm routed-down quality on your own eval set.

The sample implements the routing *decision* two ways and compares them on latency and cost:

- **LLM classifier** (`run_llm_classification`) - a tiny, deterministic Haiku Converse round-trip. Few-shot examples pin the output to one word and a stop sequence ends generation right after it, so it costs only a few output tokens per request.
- **Decision model** (`run_decider_classification`) - [Strands Decider 2B](https://strandsagents.com/blog/introducing-strands-decider/), a small open-source "system one" model that runs locally and picks between options with a *calibrated confidence* on every answer. The routing decision never touches Bedrock, so it costs no Bedrock tokens, and the confidence lets you escalate low-confidence decisions to the bigger model - something a bare label cannot do.

The decision model is served locally over HTTP (`POST /v1/systemone`). Serving it loads the 1.9B model once and keeps it warm, so each query pays only the real decision latency rather than a per-call model reload. The sample manages that server for you: `start_decider_server` launches `strands-decider serve` as a child process and waits until it is ready, and `stop_decider_server` shuts it down when the comparison finishes (an already-running server at `DECIDER_URL` is reused instead). If the `strands-decider` CLI is not installed, the sample skips the decision-model path and still runs the LLM classifier.

> This sample isolates the routing *decision* only - there is no answering step. In the source notebook the answering model is a tool-backed agent; the routing logic is the same lever either way.

### 2. Bedrock Guardrails

Guardrails are sold as safety, but they are equally a cost lever: a request blocked at the input layer never reaches the model, so you don't pay generation tokens on injection probes, jailbreaks, or off-topic traffic. The sample shows both application modes - inline on a Converse call, and the standalone `ApplyGuardrail` API that screens text without invoking a model, so you can place the check before retrieval, routing, or tool calls. Pricing is per ~1,000-character text unit, billed per policy evaluated, so blocking off-topic traffic saves the full request cost.

### 3. RAG / Indexing

Stuffing an entire knowledge source into the prompt is expensive and counterproductive - past a certain size accuracy drops as the model underweights information buried mid-prompt ("lost in the middle"). Retrieval inverts it: index the corpus once, then per query pull in only the relevant slice. This sample teaches the concept **without a vector database** - a fake 10-category, 100-product catalog stands in for the corpus, one file per category is the index, and picking the file for the question's category stands in for retrieval. The result: the same answer at roughly 89% fewer input tokens. A real system swaps file-selection for embedding search over chunks (e.g. Amazon Bedrock Knowledge Bases), adds metadata filtering and reranking, and keeps retrieved chunks *after* any `cachePoint`.

### 4. Batch Inference

For work that is not latency sensitive - embeddings, entity extraction, LLM-as-judge evaluations, bulk categorization or summarization - Bedrock batch inference runs asynchronously at 50% of the on-demand price. Same model, same output, half the cost. The sample builds a JSONL input in the Converse batch format (`recordId` + `modelInput`) and submits a `create_model_invocation_job`.

## Prerequisites

- Python 3.12+
- IAM credentials with the relevant permissions:
  - Routing & RAG: `bedrock-runtime:Converse`
  - Guardrails: `bedrock:CreateGuardrail`, `bedrock:DeleteGuardrail`, `bedrock-runtime:Converse`, `bedrock-runtime:ApplyGuardrail`
  - Batch inference: `bedrock:CreateModelInvocationJob`, an S3 bucket, and a Bedrock batch service role
- Access to Claude Haiku 4.5 and Claude 5 Sonnet on Amazon Bedrock
- Dependencies installed via `pip install -r requirements.txt` from the repository root
- `AWS_REGION` set (defaults to `us-east-1` if unset)
- OPTIONAL, for the decision-model path in `02-1_llm_routing.py`: `pip install strands-decider`. This installs the Python package only; the ~1.9B model weights are downloaded separately from Hugging Face into `~/.cache/huggingface/` the first time the server starts (not during `pip install`), so expect a one-time delay on that first run. It runs on CPU, GPU, or Apple silicon. The sample starts and stops the server itself and skips this path cleanly if the CLI is absent.

## Running the Batch Inference Sample

`02-4_batch_inference.py` always builds and shows the JSONL locally. It submits a real job only when both environment variables are set:

```bash
export BEDROCK_BATCH_BUCKET=your-s3-bucket
export BEDROCK_BATCH_ROLE_ARN=arn:aws:iam::<account>:role/<batch-service-role>
python 02-4_batch_inference.py
```

Without them it prints the sample records and the exact API call it would make, then exits cleanly - so you can inspect the format without provisioning S3 or an IAM role. A live batch job runs asynchronously (minutes to hours) and has a minimum record count (commonly 100 records per job).

## Running the Decision-Model Comparison

`02-1_llm_routing.py` always runs the LLM-classifier approach against Bedrock. It also runs the Strands Decider approach when the model is installed:

```bash
pip install strands-decider
python 02-1_llm_routing.py
```

The script starts the Decider server itself (first run downloads the ~1.9B model, which is slow once), runs both approaches on the same queries, prints the latency and Bedrock-cost comparison, then stops the server. If `strands-decider` is not installed it prints how to enable the path and runs the LLM classifier only. Override the port with `DECIDER_PORT` (default `8099`), or point at an external server with `DECIDER_URL`. The decision model needs local hardware to be useful; it runs on CPU but is fastest on a GPU or Apple silicon.

## Model Compatibility Note

`temperature` is deprecated on Claude 5 Sonnet - supplying it returns a `ValidationException`. The samples omit `temperature` on Sonnet 5 calls and keep it only for Haiku 4.5.
