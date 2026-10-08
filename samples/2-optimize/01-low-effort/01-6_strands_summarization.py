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
- Run with `opentelemetry-instrument` so Strands traces reach CloudWatch

A single response is an example, not proof that one configuration is better.
"""

import argparse
import os
import re
import time
from typing import Any

from opentelemetry import baggage, context
from strands import Agent
from strands.models import BedrockModel

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-5"
EFFORT_LEVELS = ("none", "low", "medium", "high")
MIN_SESSION_ID_LENGTH = 33
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

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
    agent = Agent(model=model, system_prompt=SYSTEM_PROMPT)

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


# export AWS_PROFILE=cost
# export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
# export AWS_REGION=us-east-1
# export AWS_DEFAULT_REGION="$AWS_REGION"
# export AGENT_OBSERVABILITY_ENABLED=true
# export AWS_GENAI_CONTENT_EXTRACTION_OPT_OUT=true
# export OTEL_PYTHON_DISTRO=aws_distro
# export OTEL_PYTHON_CONFIGURATOR=aws_configurator
# export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
# export OTEL_TRACES_EXPORTER=otlp
# export OTEL_RESOURCE_ATTRIBUTES=service.name=LowEffortSummarizer.DEFAULT
# export OTEL_EXPORTER_OTLP_TRACES_HEADERS="x-aws-log-group=/aws/bedrock-agentcore/runtimes/low-effort-summarizer-local,x-aws-log-stream=spans"
# export OTEL_EXPORTER_OTLP_LOGS_HEADERS="x-aws-log-group=/aws/bedrock-agentcore/runtimes/low-effort-summarizer-local,x-aws-log-stream=agent-logs,x-aws-metric-namespace=bedrock-agentcore"
# export AGENTCORE_EVAL_SERVICE_NAME=LowEffortSummarizer.DEFAULT
# export AGENTCORE_EVAL_LOG_GROUP=/aws/bedrock-agentcore/runtimes/low-effort-summarizer-local
# export AGENTCORE_EVAL_SPAN_LOG_STREAM=spans
# opentelemetry-instrument python 01-6_strands_summarization.py \
#   --session-id summarize-sonnet-high-workshop-run01
# opentelemetry-instrument python 01-6_strands_summarization.py \
#   --session-id summarize-haiku-low-workshop-run01 \
#   --model-id global.anthropic.claude-haiku-4-5-20251001-v1:0 --effort none
