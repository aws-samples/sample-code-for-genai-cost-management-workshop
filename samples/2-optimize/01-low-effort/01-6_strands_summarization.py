"""
Low-Effort Optimization - Lever 06: Strands Summarization Agent

Run a small Strands summarization and sentiment agent with an Amazon Bedrock
model, then evaluate its session with the companion AgentCore Evaluations
script. Change the model or adaptive-thinking effort and compare separate
sessions to see how those choices affect the response, token use, and latency.

Prerequisites:
- An AWS account with Amazon Bedrock model access
- IAM credentials with bedrock:Converse and CloudWatch telemetry permissions
- CloudWatch Transaction Search and AgentCore Observability configured
- Dependencies installed via: pip install -r requirements.txt
- Run with `python`; the script configures OpenTelemetry before loading Strands

A single response is an example, not proof that one configuration is better.
"""

import argparse
import os
import re
import time
from typing import Any

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-5"
EFFORT_LEVELS = ("none", "low", "medium", "high")
MIN_SESSION_ID_LENGTH = 33
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SERVICE_NAME = os.environ.get(
    "AGENTCORE_EVAL_SERVICE_NAME", "LowEffortSummarizer.DEFAULT"
)
LOG_GROUP_NAME = os.environ.get(
    "AGENTCORE_EVAL_LOG_GROUP",
    "/aws/bedrock-agentcore/runtimes/low-effort-summarizer-local",
)
SPAN_LOG_STREAM = os.environ.get("AGENTCORE_EVAL_SPAN_LOG_STREAM", "spans")
LOG_STREAM_NAMES = ("agent-logs", SPAN_LOG_STREAM)
LOG_RETENTION_DAYS = 7

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


# ============================================================
# Helpers
# ============================================================

def configure_observability() -> None:
    """Apply defaults and initialize instrumentation before importing Strands."""
    for name, default in OBSERVABILITY_DEFAULTS.items():
        # The SDK reads environment variables; preserve any explicit overrides.
        os.environ.setdefault(name, default)

    from opentelemetry.instrumentation.auto_instrumentation import initialize

    initialize(swallow_exceptions=False)


def validate_session_id(session_id: str) -> str:
    """Validate a descriptive session ID suitable for AgentCore tracing."""
    if len(session_id) < MIN_SESSION_ID_LENGTH:
        raise ValueError(
            f"Session ID must be at least {MIN_SESSION_ID_LENGTH} characters "
            "to remain compatible with AgentCore Runtime session IDs."
        )
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError(
            "Session ID must start with a letter or digit and contain only "
            "letters, digits, hyphens, or underscores."
        )
    return session_id


def extract_answer(result: Any) -> str:
    """Return text blocks from a Strands agent result."""
    content = result.message.get("content", [])
    return "\n".join(
        block["text"]
        for block in content
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )


def ensure_observability_log_resources() -> None:
    import boto3
    from botocore.exceptions import ClientError

    logs = boto3.client("logs", region_name=REGION)

    try:
        logs.create_log_group(logGroupName=LOG_GROUP_NAME)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceAlreadyExistsException":
            raise

    # Match the seven-day retention configured by the workshop stack.
    logs.put_retention_policy(
        logGroupName=LOG_GROUP_NAME,
        retentionInDays=LOG_RETENTION_DAYS,
    )

    for stream_name in LOG_STREAM_NAMES:
        try:
            logs.create_log_stream(
                logGroupName=LOG_GROUP_NAME,
                logStreamName=stream_name,
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") != "ResourceAlreadyExistsException":
                raise


def parse_args() -> argparse.Namespace:
    """Parse the model, effort, input review, and required trace session ID."""
    parser = argparse.ArgumentParser(
        description="Summarize and classify a product review with a Strands agent."
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help=(
            "Descriptive ID for this run (33+ letters, digits, hyphens, or "
            "underscores). Reuse it with 01-7_agentcore_evaluation.py."
        ),
    )
    parser.add_argument(
        "--review",
        default=EXAMPLE_REVIEW,
        help="Review text to summarize. Defaults to the workshop example.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Amazon Bedrock model ID. Defaults to the workshop Sonnet model.",
    )
    parser.add_argument(
        "--effort",
        choices=EFFORT_LEVELS,
        default="high",
        help=(
            "Adaptive-thinking effort for compatible Claude models; use "
            "'none' for a model without adaptive-thinking support."
        ),
    )
    return parser.parse_args()


# ============================================================
# Agent
# ============================================================

def main() -> None:
    args = parse_args()
    session_id = validate_session_id(args.session_id)
    configure_observability()

    # Import after instrumentation has read the environment defaults.
    from opentelemetry import baggage, context
    from strands import Agent
    from strands.models import BedrockModel

    # make sure logs resources are ready
    ensure_observability_log_resources()

    additional_request_fields = {}
    if args.effort != "none":
        additional_request_fields = {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": args.effort},
        }

    model = BedrockModel(
        model_id=args.model_id,
        region_name=REGION,
        max_tokens=2048,
        additional_request_fields=additional_request_fields,
    )
    # Print the final answer below instead of also streaming it to the console.
    agent = Agent(model=model, system_prompt=SYSTEM_PROMPT, callback_handler=None)

    # AgentCore Observability uses this baggage value to group spans as a session.
    otel_context = baggage.set_baggage("session.id", session_id)
    context_token = context.attach(otel_context)
    started_at = time.perf_counter()
    try:
        result = agent(args.review)
    finally:
        context.detach(context_token)

    answer = extract_answer(result)
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    usage = result.metrics.accumulated_usage

    print(f"Session ID: {session_id}")
    print(f"Model:      {args.model_id}")
    print(f"Effort:     {args.effort}")
    print(f"Latency:    {latency_ms} ms")
    print(
        f"Tokens:     input={usage['inputTokens']}, "
        f"output={usage['outputTokens']}, total={usage['totalTokens']}"
    )
    print("\nAgent response:")
    print(answer or result)


if __name__ == "__main__":
    main()


# Defaults above replace the exports; existing environment variables override them.
# Use plain python: instrumentation is initialized inside the script.
# python 01-6_strands_summarization.py \
#   --session-id summarize-sonnet-high-workshop-run01
# python 01-6_strands_summarization.py \
#   --session-id summarize-haiku-low-workshop-run01 \
#   --model-id global.anthropic.claude-haiku-4-5-20251001-v1:0 --effort none
