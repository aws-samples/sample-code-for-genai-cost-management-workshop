# Low Effort

Cost-and-latency optimization levers that require little or no application change and carry low risk. These are the first levers to pull, in order: start here before reaching for medium- or high-effort techniques.

## Overview

Each sample is a runnable demonstration of one lever. Most are independent and need no new infrastructure. The AgentCore evaluation example uses two scripts and requires CloudWatch observability plus `pricing:DescribeServices`, `pricing:GetAttributeValues`, and `pricing:GetProducts` for its cost estimate.

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
| `01-6_strands_summarization.py` | Strands Agent | Runs a Bedrock-backed Strands summarization and sentiment agent, reporting tokens and latency with configurable model and adaptive-thinking effort |
| `01-7_agentcore_evaluation.py` | AgentCore Evaluations | Scores one or more `01-6` sessions, reports per-session trace tokens and cost estimates, and returns aggregate evaluation scores |

Run any script directly:

```bash
python 01-1_model_selection.py
python 01-2_prompt_design.py
python 01-3_parameter_tuning.py
python 01-4_prompt_caching.py
python 01-5_adaptive_thinking.py
```

The AgentCore example is run as a pair. Complete the
[observability setup](#agentcore-evaluations-setup) first, then run these commands
from this directory. The Strands script initializes OpenTelemetry automatically;
no environment exports are needed for the default setup.

```bash
python 01-6_strands_summarization.py \
  --session-id summarize-sonnet-high-workshop-run01

python 01-7_agentcore_evaluation.py \
  --session-ids summarize-sonnet-high-workshop-run01
```

To compare model or prompt changes in one batch, run `01-6` for each version
with a distinct session ID, then pass all IDs to `01-7` using `--session-ids`.
Use a fresh session ID for every agent invocation so separate runs do not
contribute to the same session's traces and token totals.
Use `--model-id` to change models and `--effort` to `none`, `low`, `medium`, or
`high`; `none` disables adaptive thinking for models that do not support it.
The defaults are Claude 5 Sonnet and `high`. Use `--review "..."` to replace
the built-in product review, or edit `SYSTEM_PROMPT` to change the instructions.
For example, compare the Sonnet session above with Haiku using `--effort none`:

```bash
python 01-6_strands_summarization.py \
  --session-id summarize-haiku-none-workshop-run01 \
  --model-id global.anthropic.claude-haiku-4-5-20251001-v1:0 \
  --effort none

python 01-7_agentcore_evaluation.py \
  --session-ids summarize-sonnet-high-workshop-run01 \
    summarize-haiku-none-workshop-run01
```

The evaluation script waits 300 seconds for CloudWatch ingestion before
submitting the batch job. Keep the default wait for fresh sessions; reducing
it can cause the evaluation to miss spans that have not reached CloudWatch yet.
Use `--ingestion-wait-seconds` to change this delay. Pass `--no-wait` on the
evaluation command to skip polling for completion; it still performs the
ingestion wait, submits the job, and reports session usage before returning.
By default, it polls every 30 seconds for up to 1,800 seconds after usage
reporting. Each invocation starts a new evaluation job and accepts up to
500 session IDs.

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

### 6. Strands Summarization Agent

`01-6` summarizes a product review and classifies its sentiment with a Strands
agent backed by Amazon Bedrock. It prints the session ID, model, effort,
input/output/total tokens, agent-call latency, and final response once.
Latency excludes instrumentation setup and log-resource creation.

The prompt requests a one-sentence summary and a sentiment label on exactly
two lines. This is a prompt instruction, not a format validator. The agent's
traces include the review, response, and session ID so `01-7` can evaluate
the same run.

### 7. AgentCore Evaluations

`01-7` starts a batch evaluation over the selected CloudWatch sessions using:

- `Builtin.Correctness`: judges the response against the conversation context.
  This sample does not supply an `expectedResponse` reference answer.
- `Builtin.GoalSuccessRate`: judges the two supplied assertions: a concise,
  accurate, one-sentence summary and exactly one supported sentiment label.

The script reports each session's trace token totals and estimated input/output
inference cost, in the order of the supplied session IDs. `utils.py` searches
the last 24 hours of traces and retrieves current on-demand list prices from
the AWS Price List API. These estimates exclude evaluation and observability
charges. If traces or unambiguous prices are unavailable, the script reports
that limitation and continues monitoring the evaluation.

After completion, the script prints the job status, completed-session count,
and each evaluator's average score and evaluated count across the batch.
Individual scores and judge explanations are in the CloudWatch log location
returned by `GetBatchEvaluation.outputConfig`; the script does not print them.
See [AWS: batch evaluation results](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/batch-evaluations-results.html).

A score of `1.0` reflects the evaluator's judgment on these sessions. The
assertions do not explicitly check the exact two-line format, and one review
per configuration does not establish general model quality or cost savings.
Use multiple representative reviews and compare per-session results when
choosing a configuration.

## Model Compatibility Notes

Two behaviors of the Claude 5 models affect these samples (and are worth knowing before you reuse the code):

- **`temperature` is deprecated on Claude 5 Sonnet and Opus.** Supplying it returns `ValidationException: temperature is deprecated for this model`. The samples omit `temperature` for those models and keep it only for Haiku 4.5 (which is why the parameter-tuning demo runs on Haiku).
- **Native structured output is not accepted by Claude 5 Sonnet.** The `outputConfig` / JSON-schema path returns `ValidationException: output_config.format: Extra inputs are not permitted` on Sonnet 5, so `01-2` runs that specific demo on Haiku 4.5. When you need a schema guarantee on a thinking-capable Claude 5 model, use tool use (forced tool call) instead.

## Prerequisites

- Python 3.12+
- IAM credentials with `bedrock:InvokeModel` for Converse and
  `bedrock:InvokeModelWithResponseStream` for streaming calls, including the
  Strands sample ([AWS permissions reference](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html))
- Access to Claude Haiku 4.5, Claude 5 Sonnet, and Claude 5 Opus on Amazon Bedrock
- Dependencies installed via `pip install -r requirements.txt` from the repository root
- Optional `AWS_REGION` override; the scripts default to `us-east-1`

## AgentCore Evaluations setup

The Strands agent runs locally; it does not use Amazon Bedrock AgentCore
Runtime. AgentCore Evaluations still needs the agent's supported Strands
telemetry in CloudWatch:

- `01-6` creates or reuses the CloudWatch log group
  `/aws/bedrock-agentcore/runtimes/low-effort-summarizer-local` and its
  `agent-logs` and `spans` streams. It sets the group's retention to seven days
  on each run. `01-7` leaves the logs intact by default.
- Enable CloudWatch Transaction Search. X-Ray's trace-segment destination must
  be `CloudWatchLogs` and its status must be `ACTIVE`. Configure the CloudWatch
  Logs resource policies that allow X-Ray to deliver traces to the custom log
  group and the Transaction Search log groups. Neither these scripts nor the
  checked-in [workshop stack](../../../cfn/workshop-stack.yaml) configure
  Transaction Search or these resource policies.
- The script applies local AgentCore Observability defaults before initializing
  OpenTelemetry: service name `LowEffortSummarizer.DEFAULT`, traces and logs
  directed to the log group above, and content capture enabled so evaluation
  can read prompts and responses. Run it with plain `python`; a separate
  `opentelemetry-instrument` launcher is unnecessary.
- Use AWS credentials with Bedrock model invocation, X-Ray/CloudWatch telemetry
  export, log-group/stream creation, retention updates, trace reading, and
  AgentCore batch-evaluation permissions. Cost lookup additionally uses the
  three AWS Price List permissions listed in the overview. Batch evaluation
  runs under the caller's credentials; it does not require an AgentCore Runtime
  or a separate evaluation service role.

See [AWS: observability setup](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-get-started.html)
for the CloudWatch Transaction Search, ADOT, environment-variable, and resource
policy setup. The traces contain the review text and model response because
AgentCore Evaluations needs that content to score the session.

Configure Transaction Search and the resource policies before running the
agent. The script handles creation of its log group and streams.
AWS credentials should come from the normal AWS credential chain, such as an
AWS profile. No environment exports are needed for the default setup.
`01-6_strands_summarization.py` contains the defaults and preserves existing
environment overrides. For a custom setup, these optional settings are shared
with the evaluation script:

```bash
export AWS_REGION=us-east-1
export AGENTCORE_EVAL_SERVICE_NAME=LowEffortSummarizer.DEFAULT
export AGENTCORE_EVAL_LOG_GROUP=/aws/bedrock-agentcore/runtimes/low-effort-summarizer-local
export AGENTCORE_EVAL_SPAN_LOG_STREAM=spans
```

The agent derives its OpenTelemetry service name and log destinations from
these settings unless the corresponding `OTEL_*` variables are already set.

Both scripts require session IDs of at least 33 characters, starting with an
ASCII letter or digit and containing only ASCII letters, digits, hyphens, and
underscores. AgentCore Runtime has this minimum length; the local example
keeps the same rule so its session IDs remain portable.
