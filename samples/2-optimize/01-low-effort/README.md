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
| Amazon Nova 2 Lite | `global.amazon.nova-2-lite-v1:0` |

Nova 2 Lite is an extra row in the comparison tables of the model selection, prompt design, output and reasoning limits, prompt caching, and evaluation samples. It uses the same Converse API; only a few request fields differ (see Model Compatibility Notes). The retry, turn and stop limits sample stays on Haiku.

| Script | Lever | What it demonstrates |
|--------|-------|----------------------|
| `01-1_model_selection.py` | Model Selection | Runs the same task on Haiku, Claude 5 Sonnet, and Nova 2 Lite, comparing input/output tokens, latency, and per-call cost side by side |
| `01-2_prompt_design.py` | Prompt Design | Clear vs vague instructions, zero-shot vs few-shot consistency, and three structured-output methods (prompt-based, tool use, native JSON schema), with Nova 2 Lite rows for every method it supports |
| `01-3_output_reasoning_limits.py` | Output and Reasoning Limits | `max_tokens` (the TPM quota reserved up front), `stop_sequences` (early termination), `temperature` (determinism vs variation), and reasoning `effort` swept on a simple and a complex task (Claude 5 Sonnet adaptive thinking, Nova 2 Lite `reasoningConfig`); `max_tokens`, `stop_sequences`, and `temperature` also run on Nova 2 Lite |
| `01-4_prompt_caching.py` | Prompt Caching | Caches a large static prefix with a `cachePoint` marker, showing a cache write on the first call and a cache read on the second (Claude billed ~0.1x, Nova 2 Lite 0.25x with no write premium) |
| `01-5_retry_turn_limits.py` | Retry, Turn and Stop Limits | Caps SDK retries (against a simulated throttling outage), caps an application retry loop whose validator always fails, and caps turns and tool calls on a stuck tool-use loop, printing the spend each cap bounds |
| `01-6_evaluation.py` | Evaluation (LLM-as-a-Judge) | A rubric-based judge (Claude 5 Sonnet, native structured output) scores Haiku, Nova 2 Lite, and Sonnet answers on a small eval set, after a calibration check with a known-bad answer; compares quality, generation cost, and judging cost to decide whether the cheaper model is good enough |
| `01-7_agentcore_evaluation.py` | Strands Agent + AgentCore Evaluations | Runs a Bedrock-backed Strands summarization and sentiment agent on Haiku 4.5, Claude 5 Sonnet (low effort), and Claude 5 Sonnet (high effort), scores all three sessions in one AgentCore evaluation, and compares tokens, estimated cost, latency, and evaluation scores |

Run any script directly:

```bash
python 01-1_model_selection.py
python 01-2_prompt_design.py
python 01-3_output_reasoning_limits.py
python 01-4_prompt_caching.py
python 01-5_retry_turn_limits.py
python 01-6_evaluation.py
```

The AgentCore example is a single script. Complete the
[observability setup](#agentcore-evaluations-setup) first, then run this command
from this directory. The script initializes OpenTelemetry automatically; no
environment exports are needed for the default setup.

```bash
python 01-7_agentcore_evaluation.py
```

The script takes no command-line arguments. It runs the same summarization
task three times, each in its own trace session:

| Run | Model | Effort |
|-----|-------|--------|
| 1 | Claude Haiku 4.5 (`global.anthropic.claude-haiku-4-5-20251001-v1:0`) | `none` |
| 2 | Claude 5 Sonnet (`global.anthropic.claude-sonnet-5`) | `low` |
| 3 | Claude 5 Sonnet (`global.anthropic.claude-sonnet-5`) | `high` |

Each session ID is generated as
`summarize-<model>-<effort>-workshop-run-<GMT epoch millis>`, for example
`summarize-sonnet-5-low-workshop-run-1792587720944`, so separate runs never
share a session's traces or token totals. After all three runs, the script
evaluates the sessions together and prints a side-by-side comparison of input,
output, and total tokens, estimated cost, latency, and the `Correctness` and
`GoalSuccessRate` scores.

Edit `CONFIGS` at the top of `01-7_agentcore_evaluation.py` to compare other
models or adaptive-thinking efforts (`none`, `low`, `medium`, or `high`; `none`
disables adaptive thinking for models that do not support it). Edit
`EXAMPLE_REVIEW` to replace the built-in product review, or `SYSTEM_PROMPT` to
change the instructions.

AgentCore evaluation is an asynchronous task: in development or production it
can run at a later time or periodically on traces already in CloudWatch. For the
demo, the script runs the summarizations and the evaluation together. It waits
300 seconds (`INGESTION_WAIT_SECONDS`) for CloudWatch ingestion before
submitting the batch job. Keep the default wait for fresh sessions; reducing it
can make the evaluation fail with "No spans were found". It then polls every 30
seconds for up to 1,800 seconds. Each run starts a new evaluation job.

## The Levers

### 1. Model Selection

The highest-leverage cost decision is *which* model runs each call - the price gap between Claude tiers is roughly 5x per tier. Start on the cheapest tier (Haiku), and escalate only when a scored eval shows it falls short on *your* task. Re-benchmark every model generation; "needs the big model" decisions go stale.

### 2. Prompt Design

The most under-used lever. Most production prompts carry 30-50% wasted tokens (hedging, boilerplate, vague instructions) that cost money on every call and degrade quality. Lead with the task, use few-shot examples to pin output format, and when downstream code consumes the result, *constrain* the output rather than *asking* for a shape.

### 3. Output and Reasoning Limits

Do not default to maximum output or reasoning; tune both to task complexity. `max_tokens` is load-bearing: on Bedrock it sets the TPM quota reserved up front (output burns at 5x on Haiku 4.5, 10x on Claude 5 Sonnet, and 1x on Nova), so an oversized value throttles concurrency under load even though per-call cost is unchanged. Use `temperature=0` for classification/extraction and `stop_sequences` for a free latency win when a real terminator exists.

Adaptive thinking is the reasoning side of the same lever: set an `effort` level and the model decides, per request, how much to reason before answering. Reasoning tokens are billed as output tokens and count against `max_tokens`. The practical pattern is to step down from the default `high` once an eval shows a lower level holds quality. On Claude 5 Sonnet a simple task costs about the same at every level, so only complex tasks pay for effort. Nova 2 Lite reasons only when you turn it on, and on a simple task even `low` multiplies the billed tokens, so enable it only where the task needs it.

### 4. Prompt Caching

Requests that resend the same system prompt, tool definitions, or reference document pay full input price for tokens that never change. A `cachePoint` marker pays full price once (the write, ~1.25x), then ~0.1x for every read - break-even is around 3 reads. Nova 2 Lite has no write premium and reads at 0.25x, so it pays back from the first read, though its cached tokens are cheaper to begin with. A read can miss right after the write, and a miss re-writes the prefix, so count retries in the total. Put static content first and dynamic content last, and watch cache hit rate like an SLO (target >= 70%).

### 5. Retry, Turn and Stop Limits

Do not retry or loop without a limit. SDK retries on throttling are not billed per attempt, but each one adds latency and holds capacity, so cap them from how long transient failures really last (botocore's retry `max_attempts` counts retries after the first call). Application retries - the model answered but your validator rejected it - are billed in full every time, and a mis-specified prompt or validator fails every attempt, so cap them and fix the cause. In a tool-use loop the whole history is resent each turn, so input tokens climb with every turn: cap max turns and max tool calls, check `stopReason` on every response, and return a graceful fallback when a cap is hit.

### 6. Evaluation (LLM-as-a-Judge)

Every lever above says "confirm on an eval set". This is the smallest useful eval: a handful of representative inputs, a rubric with per-criterion scores, and a second model that grades each answer. Write the rationale before the scores, calibrate the judge on a deliberately wrong answer so you know it can fail something, and remember the judge costs tokens too - run it on a sample of traffic or in CI, not on every request. Judges favor longer answers and their own style, so spot-check a few judgments by hand. Edit `CANDIDATES`, `JUDGE`, `EVAL_SET`, and `PASS_THRESHOLD` at the top of the file to try your own setup.

This sample hand-rolls the judge so every step is visible. For production, AWS offers two managed options:

- [Amazon Bedrock Evaluations](https://docs.aws.amazon.com/bedrock/latest/userguide/evaluation-judge.html) - a model evaluation job that uses a judge model. You supply a prompt dataset (JSONL in Amazon S3), a generator model, and an evaluator model; Bedrock scores each response with built-in or custom metrics and explains each score. You can also bring responses from a model outside Bedrock. Best for comparing models or prompts offline, which is what this sample does.
- [Amazon Bedrock AgentCore Evaluations](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/evaluations.html) - LLM-as-a-judge scoring of agent behavior from OpenTelemetry traces (Strands, LangGraph), with built-in evaluators such as `Builtin.Helpfulness`, custom evaluators, on-demand evaluation for CI/CD, and online evaluation of live traffic. Best when the thing you are evaluating is an agent rather than a single model call.

### 7. Strands Summarization Agent

`01-7` calls `summarize()` once per entry in `CONFIGS`. Each call summarizes a
product review and classifies its sentiment with a Strands agent backed by
Amazon Bedrock. The script prints the session ID, model, and effort before each
run starts, then `summarize()` prints the input/output/total tokens, agent-call
latency, and final response. Latency excludes instrumentation setup and
log-resource creation. `run_agentcore_evaluation()` then evaluates all the
sessions in one batch.

The prompt requests a one-sentence summary and a sentiment label on exactly
two lines. This is a prompt instruction, not a format validator. The agent's
traces include the review, response, and session ID so AgentCore Evaluations
can evaluate the same run.

### 8. AgentCore Evaluations

`run_agentcore_evaluation()` starts one batch evaluation over the sessions
that were just run, using:

- `Builtin.Correctness`: judges the response against the conversation context.
  This sample does not supply an `expectedResponse` reference answer.
- `Builtin.GoalSuccessRate`: judges the two supplied assertions: a concise,
  accurate, one-sentence summary and exactly one supported sentiment label.

The script reports each session's trace token totals and estimated input/output
inference cost. `utils.py` searches
the last 24 hours of traces and retrieves current on-demand list prices from
the AWS Price List API. These estimates exclude evaluation and observability
charges. If traces or unambiguous prices are unavailable, the script reports
that limitation (shown as `n/a` in the comparison) and continues.

After completion, the script prints the job status and completed-session count,
then reads each session's evaluator scores from the CloudWatch log location
returned by `GetBatchEvaluation.outputConfig` and shows them in the comparison
table. Judge explanations stay in that log location; the script does not print
them.
See [AWS: batch evaluation results](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/batch-evaluations-results.html).

A score of `1.0` reflects the evaluator's judgment on these sessions. The
assertions do not explicitly check the exact two-line format, and one review
per configuration does not establish general model quality or cost savings.
Use multiple representative reviews and compare per-session results when
choosing a configuration.

## Model Compatibility Notes

Two behaviors of the Claude 5 models affect these samples (and are worth knowing before you reuse the code):

- **`temperature` is deprecated on Claude 5 Sonnet.** Supplying it returns `ValidationException: temperature is deprecated for this model`. The samples omit `temperature` for Sonnet 5 and keep it for Haiku 4.5 and Nova 2 Lite (which is why the `temperature` demo runs on those two).
- **Native structured output works on Claude 5 Sonnet.** `01-2` passes the schema through Converse `outputConfig.textFormat` (as a `json.dumps` string, with `additionalProperties: false` on every object). Select the text block by key rather than `content[0]`, because thinking-capable models can lead with a reasoning block.

Amazon Nova 2 Lite (`global.amazon.nova-2-lite-v1:0`) works through the same Converse API, with these differences:

- **Reasoning is opt-in and uses a different field.** Pass `additionalModelRequestFields={"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low" | "medium" | "high"}}`. At `high`, `maxTokens` must be left unset. The reasoning text comes back redacted but is billed as output tokens.
- **No native JSON schema.** `outputConfig` is rejected with a `ValidationException`, so use prompt-based JSON, assistant prefill, or tool use, and validate the result in code. The `strict` field on a tool is also rejected. Prompt-based JSON often arrives wrapped in a markdown fence, so strip it before parsing.
- **Stop sequences work, but the marker stays in the output** and `stopReason` is `end_turn` (Claude returns `stop_sequence` and drops the marker).
- **Prompt caching uses the same `cachePoint`.** A call right after the write can miss the cache, so the caching sample asks again, a bounded number of times, until it sees a read.
- **Output burndown is 1x**, against 5x on Haiku 4.5 and 10x on Claude 5 Sonnet, so the same `max_tokens` reserves far less TPM quota.

## Prerequisites

- Python 3.12+
- IAM credentials with `bedrock:InvokeModel` for Converse and
  `bedrock:InvokeModelWithResponseStream` for streaming calls, including the
  Strands sample ([AWS permissions reference](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html))
- Access to Claude Haiku 4.5, Claude 5 Sonnet, and Amazon Nova 2 Lite on Amazon Bedrock
- Dependencies installed via `pip install -r requirements.txt` from the repository root
- Optional `AWS_REGION` override; the scripts default to `us-east-1`

## AgentCore Evaluations setup

The Strands agent runs locally; it does not use Amazon Bedrock AgentCore
Runtime. AgentCore Evaluations still needs the agent's supported Strands
telemetry in CloudWatch:

- `01-7` creates or reuses the CloudWatch log group
  `/aws/bedrock-agentcore/runtimes/low-effort-summarizer-local` and its
  `agent-logs` and `spans` streams. It sets the group's retention to seven days
  on each run. It leaves the logs intact by default (`DELETE_LOGS = False`).
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
`01-7_agentcore_evaluation.py` contains the defaults and preserves existing
environment overrides. For a custom setup, these optional settings are shared
with `utils.py`:

```bash
export AWS_REGION=us-east-1
export AGENTCORE_EVAL_SERVICE_NAME=LowEffortSummarizer.DEFAULT
export AGENTCORE_EVAL_LOG_GROUP=/aws/bedrock-agentcore/runtimes/low-effort-summarizer-local
export AGENTCORE_EVAL_SPAN_LOG_STREAM=spans
```

The agent derives its OpenTelemetry service name and log destinations from
these settings unless the corresponding `OTEL_*` variables are already set.

The script requires session IDs of at least 33 characters, starting with an
ASCII letter or digit and containing only ASCII letters, digits, hyphens, and
underscores. AgentCore Runtime has this minimum length; the local example
keeps the same rule so its session IDs remain portable.
