"""
Low-Effort Optimization - Lever 06: Strands Agent + AgentCore Evaluations

Run a small Strands summarization and sentiment agent three times with Amazon
Bedrock: Claude Haiku 4.5 (no adaptive thinking), then Claude 5 Sonnet with low
and high adaptive-thinking effort. Evaluate all three sessions together with
AgentCore Evaluations, then compare token use, estimated cost, latency, and
evaluation scores side by side. Edit CONFIGS below to compare other choices.

Prerequisites:
- An AWS account with Amazon Bedrock model access
- IAM credentials with bedrock:Converse, CloudWatch telemetry, and
  AgentCore Evaluations permissions
- CloudWatch Transaction Search and AgentCore Observability configured
- Dependencies installed via: pip install -r requirements.txt
- Run with `python`; the script configures OpenTelemetry before loading Strands

A single response is an example, not proof that one configuration is better.
"""

import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError
from utils import (
    LOG_GROUP_NAME,
    REGION,
    SPAN_LOG_STREAM,
    delete_observability_log_resources,
    get_session_usage,
    read_evaluation_scores,
    validate_session_id,
)


# ============================================================
# Configuration
# ============================================================

# Each configuration is (model ID, adaptive-thinking effort). Effort is "low",
# "medium", or "high"; use "none" for a model without adaptive thinking.
CONFIGS = [
    ("global.anthropic.claude-haiku-4-5-20251001-v1:0", "none"),
    ("global.anthropic.claude-sonnet-5", "low"),
    ("global.anthropic.claude-sonnet-5", "high"),
]

SERVICE_NAME = os.environ.get(
    "AGENTCORE_EVAL_SERVICE_NAME", "LowEffortSummarizer.DEFAULT"
)
LOG_STREAM_NAMES = ("agent-logs", SPAN_LOG_STREAM)
LOG_RETENTION_DAYS = 7

EVALUATORS = ["Builtin.Correctness", "Builtin.GoalSuccessRate"]
# Wait for the traces to be ingested before evaluating; AgentCore Evaluations
# fails with "No spans were found" if it reads them too early.
INGESTION_WAIT_SECONDS = 300
POLL_INTERVAL_SECONDS = 30
POLL_TIMEOUT_SECONDS = 1800
TERMINAL_STATUSES = {
    "COMPLETED",
    "COMPLETED_WITH_ERRORS",
    "FAILED",
    "STOPPED",
}
DELETE_LOGS = False

OBSERVABILITY_DEFAULTS = {
    "AWS_REGION": REGION,
    "AWS_DEFAULT_REGION": REGION,
    "AGENT_OBSERVABILITY_ENABLED": "true",
    "AWS_GENAI_CONTENT_EXTRACTION_OPT_OUT": "true",
    "OTEL_PYTHON_DISTRO": "aws_distro",
    "OTEL_PYTHON_CONFIGURATOR": "aws_configurator",
    # Keep ADOT 0.21's exclusions. ADOT owns logging; Strands owns threading.
    "OTEL_PYTHON_DISABLED_INSTRUMENTATIONS": (
        "http,sqlalchemy,psycopg2,pymysql,sqlite3,aiopg,asyncpg,mysql_connector,"
        "urllib3,requests,system_metrics,google-genai,jinja2,logging,threading"
    ),
    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
    "OTEL_TRACES_EXPORTER": "otlp",
    "OTEL_RESOURCE_ATTRIBUTES": f"service.name={SERVICE_NAME}",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS": (
        f"x-aws-log-group={LOG_GROUP_NAME},x-aws-log-stream={SPAN_LOG_STREAM}"
    ),
    "OTEL_EXPORTER_OTLP_LOGS_HEADERS": (
        f"x-aws-log-group={LOG_GROUP_NAME},x-aws-log-stream=agent-logs,"
        "x-aws-metric-namespace=bedrock-agentcore"
    ),
}

EXAMPLE_REVIEW = (
    "I ordered this device because the product page promised reliable "
    "performance, easy setup, and a battery that would last through an entire "
    "workday for me. Delivery arrived two days early, and the sturdy packaging "
    "protected everything during shipping. Setup took only a few minutes, and "
    "the display looked sharp indoors. Unfortunately, the battery now lasts "
    "barely three hours, the charging cable disconnects if moved, and the "
    "device has frozen repeatedly during video calls. Customer support asked "
    "me to repeat troubleshooting steps, then closed my case without offering "
    "a replacement. The price seemed worthwhile, but reliability problems "
    "made this purchase frustrating."
)

SYSTEM_PROMPT = """\
You summarize customer product reviews.
For each review:
- Write a concise, accurate, one-sentence summary.
- Classify the overall sentiment as positive, neutral, or negative.

Return exactly two lines:
Summary: <one-sentence summary>
Sentiment: <positive, neutral, or negative>
"""

ASSERTIONS = [
    {
        "text": (
            "The response gives a concise, accurate, one-sentence summary "
            "of the product review."
        )
    },
    {
        "text": (
            "The response provides exactly one sentiment label: positive, "
            "neutral, or negative, and the label is supported by the review."
        )
    },
]


# ============================================================
# Helpers
# ============================================================

def log(message: str) -> None:
    """Print a timestamped progress message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def configure_observability() -> None:
    """Apply defaults and initialize instrumentation before importing Strands."""
    log("Configuring observability (OpenTelemetry + AgentCore defaults)...")
    for name, default in OBSERVABILITY_DEFAULTS.items():
        # The SDK reads environment variables; preserve any explicit overrides.
        os.environ.setdefault(name, default)

    from opentelemetry.instrumentation.auto_instrumentation import initialize

    log("Initializing OpenTelemetry auto-instrumentation...")
    initialize(swallow_exceptions=False)
    log(
        f"Observability ready: service={SERVICE_NAME}, "
        f"log group={LOG_GROUP_NAME}, span stream={SPAN_LOG_STREAM}"
    )


def model_label(model_id: str) -> str:
    """Return a short model name: "global.anthropic.claude-sonnet-5" -> "sonnet-5"."""
    name = re.sub(r"[^A-Za-z0-9_-]", "-", model_id.split(".")[-1])
    name = name.removeprefix("claude-")
    # Drop the release date and version suffix, e.g. "-20251001-v1-0".
    return re.sub(r"-20\d{6}.*$", "", name)


def default_session_id(model_id: str, effort: str) -> str:
    """Build summarize-<model>-<effort>-workshop-run-<epoch millis, GMT>."""
    return (
        f"summarize-{model_label(model_id)}-{effort}-workshop-run-"
        f"{int(time.time() * 1000)}"
    )


def extract_answer(result: Any) -> str:
    """Return text blocks from a Strands agent result."""
    content = result.message.get("content", [])
    return "\n".join(
        block["text"]
        for block in content
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )


def ensure_observability_log_resources() -> None:
    """Create the log group and streams that receive agent traces."""
    log(f"Ensuring CloudWatch log resources in {REGION}...")
    logs = boto3.client("logs", region_name=REGION)

    try:
        logs.create_log_group(logGroupName=LOG_GROUP_NAME)
        log(f"  Created log group {LOG_GROUP_NAME}")
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceAlreadyExistsException":
            raise
        log(f"  Log group {LOG_GROUP_NAME} already exists")

    # Match the seven-day retention configured by the workshop stack.
    logs.put_retention_policy(
        logGroupName=LOG_GROUP_NAME,
        retentionInDays=LOG_RETENTION_DAYS,
    )
    log(f"  Set retention to {LOG_RETENTION_DAYS} days")

    for stream_name in LOG_STREAM_NAMES:
        try:
            logs.create_log_stream(
                logGroupName=LOG_GROUP_NAME,
                logStreamName=stream_name,
            )
            log(f"  Created log stream {stream_name}")
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") != "ResourceAlreadyExistsException":
                raise
            log(f"  Log stream {stream_name} already exists")
    log("Log resources ready.")


# ============================================================
# Summarization agent
# ============================================================

def summarize(session_id: str, model_id: str, effort: str) -> dict[str, Any]:
    """Run the Strands summarization agent, print its results, and return them."""
    log(f"Summarizing with {model_id} (effort={effort})...")
    # Import after instrumentation has read the environment defaults.
    from opentelemetry import baggage, context
    from strands import Agent
    from strands.models import BedrockModel

    additional_request_fields = {}
    if effort != "none":
        additional_request_fields = {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }
        log(f"  Adaptive thinking enabled: {additional_request_fields}")
    else:
        log("  Adaptive thinking disabled (effort=none)")

    model = BedrockModel(
        model_id=model_id,
        region_name=REGION,
        max_tokens=2048,
        additional_request_fields=additional_request_fields,
    )
    log(f"  Created Bedrock model (region={REGION}, max_tokens=2048)")
    # Return the final answer instead of also streaming it to the console.
    agent = Agent(model=model, system_prompt=SYSTEM_PROMPT, callback_handler=None)
    log("  Created Strands agent")

    # AgentCore Observability uses this baggage value to group spans as a session.
    otel_context = baggage.set_baggage("session.id", session_id)
    context_token = context.attach(otel_context)
    log(f"  Tracing session.id={session_id}")
    log(f"  Invoking agent with a {len(EXAMPLE_REVIEW)}-character review...")
    started_at = time.perf_counter()
    try:
        result = agent(EXAMPLE_REVIEW)
    except Exception as exc:
        log(f"  Agent invocation failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        context.detach(context_token)

    answer = extract_answer(result)
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    usage = result.metrics.accumulated_usage
    log(
        f"  Agent responded in {latency_ms} ms "
        f"(stop reason: {getattr(result, 'stop_reason', 'unknown')})"
    )

    print(f"Latency:    {latency_ms} ms")
    print(
        f"Tokens:     input={usage['inputTokens']}, "
        f"output={usage['outputTokens']}, total={usage['totalTokens']}"
    )
    print("\nAgent response:")
    print(answer or result)
    return {"latency_ms": latency_ms}


# ============================================================
# AgentCore evaluation
# ============================================================

def wait_for_span_ingestion() -> None:
    """Wait for the sessions' spans to be ingested before evaluating them."""
    log(
        f"Waiting {INGESTION_WAIT_SECONDS} seconds for the sessions' spans to "
        "be ingested into CloudWatch..."
    )
    remaining = INGESTION_WAIT_SECONDS
    while remaining > 0:
        step = min(60, remaining)
        time.sleep(step)
        remaining -= step
        if remaining:
            log(f"  {remaining} seconds remaining...")
    log("Wait complete.")


def run_agentcore_evaluation(session_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Evaluate the sessions from their CloudWatch traces in one AgentCore job.

    Returns each session's evaluator scores, keyed by session ID.
    """
    client = boto3.client("bedrock-agentcore", region_name=REGION)
    job_name = (
        "summarization_eval_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:6]}"
    )
    response = client.start_batch_evaluation(
        batchEvaluationName=job_name,
        evaluators=[{"evaluatorId": evaluator} for evaluator in EVALUATORS],
        dataSourceConfig={
            "cloudWatchLogs": {
                "serviceNames": [SERVICE_NAME],
                "logGroupNames": [LOG_GROUP_NAME],
                "filterConfig": {"sessionIds": session_ids},
            }
        },
        evaluationMetadata={
            # Attach the assertions to each session in the batch.
            "sessionMetadata": [
                {
                    "sessionId": session_id,
                    "testScenarioId": "product_review_summarization",
                    "groundTruth": {"inline": {"assertions": ASSERTIONS}},
                }
                for session_id in session_ids
            ],
        },
        clientToken=str(uuid.uuid4()),
    )

    evaluation_id = response["batchEvaluationId"]
    print(f"Started AgentCore evaluation: {evaluation_id}")

    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        result = client.get_batch_evaluation(batchEvaluationId=evaluation_id)
        status = result["status"]
        print(f"Evaluation status: {status}")
        if status in TERMINAL_STATUSES:
            break
        time.sleep(POLL_INTERVAL_SECONDS)
    else:
        raise TimeoutError(
            f"Evaluation {evaluation_id} is still running. Check its status "
            "in the AgentCore console."
        )

    print(f"\nFinal status: {status}")
    results = result.get("evaluationResults") or {}
    print(
        f"Sessions completed: {results.get('numberOfSessionsCompleted', 0)} "
        f"of {len(session_ids)}"
    )
    for detail in result.get("errorDetails", []):
        print(f"Evaluation detail: {detail}")
    if status in {"FAILED", "STOPPED"}:
        raise RuntimeError(f"Evaluation finished with status {status}.")

    return read_evaluation_scores(
        result["outputConfig"], session_ids, EVALUATORS
    )


# ============================================================
# Comparison
# ============================================================

def format_score(scores: dict[str, Any], evaluator_id: str) -> str:
    """Format one evaluator result for the comparison table."""
    entry = scores.get(evaluator_id)
    if not entry:
        return "n/a"
    if entry["error"] or entry["score"] is None:
        return "error"
    return f"{entry['score']:.2f}"


def print_comparison(
    runs: list[dict[str, Any]], scores: dict[str, dict[str, Any]]
) -> None:
    """Print tokens, estimated cost, latency, and scores for each run."""
    headers = ["Configuration", "Input", "Output", "Total", "Cost ($)", "Latency"]
    headers += [evaluator.removeprefix("Builtin.") for evaluator in EVALUATORS]
    rows = []
    for run in runs:
        usage = get_session_usage(run["session_id"])
        if usage:
            tokens = [
                str(usage["input_tokens"]),
                str(usage["output_tokens"]),
                str(usage["total_tokens"]),
            ]
            cost = "n/a" if usage["cost"] is None else f"{usage['cost']:.6f}"
        else:
            tokens, cost = ["n/a"] * 3, "n/a"
        rows.append(
            [run["label"], *tokens, cost, f"{run['latency_ms']} ms"]
            + [
                format_score(scores.get(run["session_id"], {}), evaluator)
                for evaluator in EVALUATORS
            ]
        )

    widths = [
        max(len(row[column]) for row in [headers, *rows])
        for column in range(len(headers))
    ]
    print("\nComparison (tokens and cost from CloudWatch traces; cost excludes evaluation):")
    for row in [headers, *rows]:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))
    print(
        "\nScores are 0-1 judgments from AgentCore evaluators on one review per "
        "configuration; they are an example, not proof that one configuration "
        "is better."
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    # Step 1: Set up observability. The agent writes OpenTelemetry traces to
    # CloudWatch; AgentCore Evaluations reads those traces later to score each
    # session. Instrumentation must be initialized before Strands is imported.
    print("Step 1 of 4: Set up observability so the agent's traces reach CloudWatch")
    configure_observability()
    # Make sure the log group and streams that receive the traces exist.
    ensure_observability_log_resources()

    # Step 2: Run the same summarization task with each configuration. Every
    # run gets its own session ID, which groups its spans into one session that
    # can be evaluated and costed separately.
    print(
        f"\nStep 2 of 4: Summarize the same product review with {len(CONFIGS)} "
        "configurations, each in its own trace session:"
    )
    for model_id, effort in CONFIGS:
        print(f"  - {model_label(model_id)} with effort={effort}")

    runs = []
    for model_id, effort in CONFIGS:
        session_id = validate_session_id(default_session_id(model_id, effort))
        label = f"{model_label(model_id)}/{effort}"
        print(f"\n{'=' * 60}\nRun {len(runs) + 1} of {len(CONFIGS)}: {label}\n{'=' * 60}")
        print(f"Session ID: {session_id}")
        print(f"Model:      {model_id}")
        print(f"Effort:     {effort}\n")
        result = summarize(session_id, model_id, effort)
        runs.append(
            {"session_id": session_id, "label": label, "latency_ms": result["latency_ms"]}
        )

    # AgentCore evaluation is an asynchronous task that runs on the traces the
    # agents already wrote to CloudWatch. In development or production it can
    # run at a later time or periodically. For the sake of this demo, we run the
    # summarization and the evaluation together.
    print(
        "\n*** Note: AgentCore evaluation is an asynchronous task. In "
        "development or production it can run at a later time or "
        "periodically. For the sake of this demo, we run the summarization "
        "and evaluation together. ***\n"
    )

    # Step 3: Give CloudWatch time to ingest the spans, then score all the
    # sessions in a single AgentCore batch evaluation. The evaluators judge each
    # response against the assertions in ASSERTIONS.
    print(
        "Step 3 of 4: Evaluate all sessions with AgentCore Evaluations "
        f"({', '.join(evaluator.removeprefix('Builtin.') for evaluator in EVALUATORS)})"
    )
    wait_for_span_ingestion()
    scores = run_agentcore_evaluation([run["session_id"] for run in runs])

    # Step 4: Compare the configurations side by side: tokens and estimated
    # cost from the traces, latency from the run, and the evaluation scores.
    print("\nStep 4 of 4: Compare tokens, cost, latency, and evaluation scores")
    print_comparison(runs, scores)

    if DELETE_LOGS:
        delete_observability_log_resources()


if __name__ == "__main__":
    main()


# Defaults above replace the exports; existing environment variables override them.
# Use plain python: instrumentation is initialized inside the script.
# To compare other configurations, edit CONFIGS, then run again:
# python 01-7_agentcore_evaluation.py
