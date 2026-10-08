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

The AgentCore example is run as a pair. First run the Strands agent under
OpenTelemetry instrumentation, then evaluate that session:

```bash
opentelemetry-instrument python 01-6_strands_summarization.py \
  --session-id summarize-sonnet-high-workshop-run01

python 01-7_agentcore_evaluation.py \
  --session-ids summarize-sonnet-high-workshop-run01
```

To compare model or prompt changes in one batch, run `01-6` for each version
with a distinct session ID, then pass all IDs to `01-7` using `--session-ids`.
Use `--model-id` to change models and `--effort` to `none`, `low`, `medium`, or
`high`; `none` disables adaptive thinking for models that do not support it.
You can also edit `SYSTEM_PROMPT` in the agent script. For example:

```bash
opentelemetry-instrument python 01-6_strands_summarization.py \
  --session-id summarize-haiku-low-workshop-run01 \
  --model-id global.anthropic.claude-haiku-4-5-20251001-v1:0 \
  --effort none

python 01-7_agentcore_evaluation.py \
  --session-ids summarize-sonnet-high-workshop-run01 \
    summarize-haiku-low-workshop-run01
```

The evaluation script waits for CloudWatch ingestion before submitting the
batch job; pass `--no-wait` to return the batch job ID without polling for its
results. The job reports aggregate averages across the selected sessions;
per-session scores are available in its CloudWatch result logs.

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

## Model Compatibility Notes

Two behaviors of the Claude 5 models affect these samples (and are worth knowing before you reuse the code):

- **`temperature` is deprecated on Claude 5 Sonnet and Opus.** Supplying it returns `ValidationException: temperature is deprecated for this model`. The samples omit `temperature` for those models and keep it only for Haiku 4.5 (which is why the parameter-tuning demo runs on Haiku).
- **Native structured output is not accepted by Claude 5 Sonnet.** The `outputConfig` / JSON-schema path returns `ValidationException: output_config.format: Extra inputs are not permitted` on Sonnet 5, so `01-2` runs that specific demo on Haiku 4.5. When you need a schema guarantee on a thinking-capable Claude 5 model, use tool use (forced tool call) instead.

## Prerequisites

- Python 3.12+
- IAM credentials with `bedrock-runtime:Converse` permission
- Access to Claude Haiku 4.5, Claude 5 Sonnet, and Claude 5 Opus on Amazon Bedrock
- Dependencies installed via `pip install -r requirements.txt` from the repository root
- `AWS_REGION` set (defaults to `us-east-1` if unset)

## AgentCore Evaluations setup

The Strands agent runs locally; it does not use Amazon Bedrock AgentCore
Runtime. AgentCore Evaluations still needs the agent's supported Strands
telemetry in CloudWatch:

- Enable CloudWatch Transaction Search for the account.
- Create the CloudWatch log group
  `/aws/bedrock-agentcore/runtimes/low-effort-summarizer-local`.
- Configure the CloudWatch Logs resource policy that allows X-Ray to deliver
  traces to that custom log group.
- Configure local AgentCore Observability environment variables before using
  `opentelemetry-instrument`. Set the service name to
  `LowEffortSummarizer.DEFAULT`, direct traces and logs to the log group above,
  and enable content capture so evaluation can read prompts and responses.
- Use AWS credentials with Bedrock model invocation, CloudWatch Logs, and
  AgentCore batch-evaluation permissions. Batch evaluation runs under the
  caller's credentials; it does not require an AgentCore Runtime or a separate
  evaluation service role.

See [AWS: observability for agents hosted outside AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-configure.html)
for the CloudWatch Transaction Search, ADOT, environment-variable, and resource
policy setup. The traces contain the review text and model response because
AgentCore Evaluations needs that content to score the session.

For a local shell, set the telemetry variables after creating the log group
and its X-Ray resource policy. AWS credentials should come from the normal
AWS credential chain, such as an AWS profile:

```bash
export AWS_ACCOUNT_ID=<your-account-id>
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION="$AWS_REGION"

export AGENT_OBSERVABILITY_ENABLED=true
export AWS_GENAI_CONTENT_EXTRACTION_OPT_OUT=true
export OTEL_PYTHON_DISTRO=aws_distro
export OTEL_PYTHON_CONFIGURATOR=aws_configurator
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_TRACES_EXPORTER=otlp
export OTEL_RESOURCE_ATTRIBUTES=service.name=LowEffortSummarizer.DEFAULT
export OTEL_EXPORTER_OTLP_TRACES_HEADERS=x-aws-log-group=/aws/bedrock-agentcore/runtimes/low-effort-summarizer-local,x-aws-log-stream=spans
export OTEL_EXPORTER_OTLP_LOGS_HEADERS=x-aws-log-group=/aws/bedrock-agentcore/runtimes/low-effort-summarizer-local,x-aws-log-stream=agent-logs,x-aws-metric-namespace=bedrock-agentcore

export AGENTCORE_EVAL_SERVICE_NAME=LowEffortSummarizer.DEFAULT
export AGENTCORE_EVAL_LOG_GROUP=/aws/bedrock-agentcore/runtimes/low-effort-summarizer-local
```

The agent requires a session ID of at least 33 characters and permits only
letters, numbers, hyphens, and underscores. AgentCore Runtime has this minimum;
the local example keeps the same rule so its session IDs remain portable.
