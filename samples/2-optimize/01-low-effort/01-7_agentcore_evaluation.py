"""
Create and monitor an AgentCore evaluation for Strands sessions.

Run 01-6 once per session, then pass their IDs to this script.
"""

import argparse
import os
import time
import uuid
from datetime import datetime, timezone

import boto3

from utils import REGION, report_session_usage, validate_session_id


SERVICE_NAME = os.environ.get(
    "AGENTCORE_EVAL_SERVICE_NAME", "LowEffortSummarizer.DEFAULT"
)
LOG_GROUP_NAME = os.environ.get(
    "AGENTCORE_EVAL_LOG_GROUP",
    "/aws/bedrock-agentcore/runtimes/low-effort-summarizer-local",
)
EVALUATORS = ["Builtin.Correctness", "Builtin.GoalSuccessRate"]
INGESTION_WAIT_SECONDS = 300
POLL_INTERVAL_SECONDS = 30
POLL_TIMEOUT_SECONDS = 1800
TERMINAL_STATUSES = {
    "COMPLETED",
    "COMPLETED_WITH_ERRORS",
    "FAILED",
    "STOPPED",
}

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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one or more Strands agent sessions with AgentCore."
    )
    parser.add_argument(
        "--session-ids",
        "--session-id",
        dest="session_ids",
        nargs="+",
        required=True,
        help="One or more session IDs passed to 01-6_strands_summarization.py.",
    )
    parser.add_argument(
        "--ingestion-wait-seconds",
        type=int,
        default=INGESTION_WAIT_SECONDS,
        help=f"CloudWatch wait before evaluation (default: {INGESTION_WAIT_SECONDS}).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit the job and return without polling for results.",
    )
    args = parser.parse_args()
    session_ids = [validate_session_id(value) for value in args.session_ids]
    if len(session_ids) > 500:
        parser.error("A batch evaluation supports at most 500 session IDs.")
    if args.ingestion_wait_seconds < 0:
        parser.error("--ingestion-wait-seconds cannot be negative.")

    if args.ingestion_wait_seconds:
        print(
            f"Waiting {args.ingestion_wait_seconds} seconds for CloudWatch "
            "ingestion..."
        )
        time.sleep(args.ingestion_wait_seconds)

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
    print(f"Session IDs: {', '.join(session_ids)}")
    for session_id in session_ids:
        report_session_usage(session_id)
    if args.no_wait:
        print("Evaluation is running asynchronously.")
        return

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
    print(f"Sessions completed: {results.get('numberOfSessionsCompleted', 0)}")
    for summary in results.get("evaluatorSummaries", []):
        statistics = summary.get("statistics", {})
        print(
            f"{summary['evaluatorId']}: "
            f"average score={statistics.get('averageScore')}, "
            f"evaluated={summary.get('totalEvaluated', 0)}"
        )
    for detail in result.get("errorDetails", []):
        print(f"Evaluation detail: {detail}")
    if status in {"FAILED", "STOPPED", "COMPLETED_WITH_ERRORS"}:
        raise RuntimeError(f"Evaluation finished with status {status}.")


if __name__ == "__main__":
    main()


# export AWS_REGION=us-east-1
# export AWS_DEFAULT_REGION="$AWS_REGION"
# export AGENTCORE_EVAL_SERVICE_NAME=LowEffortSummarizer.DEFAULT
# export AGENTCORE_EVAL_LOG_GROUP=/aws/bedrock-agentcore/runtimes/low-effort-summarizer-local
# export AGENTCORE_EVAL_SPAN_LOG_STREAM=spans
# python 01-7_agentcore_evaluation.py \
#   --session-ids summarize-sonnet-high-workshop-run01 summarize-haiku-low-workshop-run01
