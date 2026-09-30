"""
Amazon Bedrock IAM Identity Log Attribution - Usage Cost Report

This script queries CloudWatch Logs Insights for the past 7 days of model
invocation logs, aggregates token usage by IAM role and model, and estimates
the cost using per-model pricing.

You will learn how to:
- Query CloudWatch Logs Insights programmatically
- Aggregate token counts per identity and model
- Estimate spend from token usage and known pricing

Prerequisites:
- Model invocation logging enabled (CloudWatch Logs destination)
- Run 6-2_invoke_and_query_logs.py first to generate log data
- IAM credentials with logs:StartQuery and logs:GetQueryResults permissions
- Dependencies installed via: pip install -r requirements.txt
"""

import time
import boto3
from datetime import datetime, timedelta

# ============================================================
# Configuration
# ============================================================

LOOKBACK_DAYS = 7

# Pricing per 1,000 tokens (USD)
PRICING = {
    "global.anthropic.claude-sonnet-5": {"input": 0.003, "output": 0.015},
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": {"input": 0.001, "output": 0.005},
    "global.amazon.nova-2-lite-v1:0": {"input": 0.0006, "output": 0.0024},
}

# CloudWatch Logs Insights query
QUERY = """
fields identity.arn, modelId, input.inputTokenCount, output.outputTokenCount
| filter identity.arn like /bedrock-workshop-developer/
| parse identity.arn "assumed-role/*/" as role_name
| stats sum(input.inputTokenCount) as input_tokens,
        sum(output.outputTokenCount) as output_tokens,
        count(*) as requests
  by role_name, modelId
"""


# ============================================================
# Query Functions
# ============================================================

def detect_log_group() -> str:
    """Detect the CloudWatch log group from Bedrock model invocation logging settings."""
    bedrock = boto3.client("bedrock")
    response = bedrock.get_model_invocation_logging_configuration()
    config = response.get("loggingConfig", {})

    cw_config = config.get("cloudWatchConfig", {})
    log_group = cw_config.get("logGroupName")

    if not log_group:
        raise RuntimeError(
            "No CloudWatch log group configured for Bedrock model invocation logging. "
            "Enable logging in the Amazon Bedrock console or via the API."
        )
    return log_group

def run_insights_query(logs_client, query: str, log_group: str, days: int) -> list:
    """Start a CloudWatch Logs Insights query and wait for results."""
    response = logs_client.start_query(
        logGroupName=log_group,
        startTime=int((datetime.now() - timedelta(days=days)).timestamp()),
        endTime=int(datetime.now().timestamp()),
        queryString=query,
    )
    query_id = response["queryId"]

    # Poll until the query completes
    while True:
        result = logs_client.get_query_results(queryId=query_id)
        if result["status"] == "Complete":
            break
        if result["status"] == "Failed":
            raise RuntimeError(f"Query failed: {result}")
        time.sleep(1)

    # Convert results from list-of-fields to list-of-dicts
    rows = []
    for entry in result["results"]:
        row = {field["field"]: field["value"] for field in entry}
        rows.append(row)
    return rows


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate cost based on token counts and model pricing."""
    pricing = PRICING.get(model, {"input": 0, "output": 0})
    return (input_tokens * pricing["input"] / 1000) + (output_tokens * pricing["output"] / 1000)


# ============================================================
# Main
# ============================================================

def main():
    logs_client = boto3.client("logs")

    # Detect log group from Bedrock settings
    log_group = detect_log_group()
    print(f"  Detected log group from Bedrock settings: {log_group}")

    print(f"\n--- Usage Report (last {LOOKBACK_DAYS} days) ---\n")
    print(f"  Log group: {log_group}")
    print(f"  Running query...\n")

    results = run_insights_query(logs_client, QUERY, log_group, LOOKBACK_DAYS)

    if not results:
        print("  No results found. Ensure model invocation logging is enabled")
        print("  and that 6-2_invoke_and_query_logs.py has been run.")
        return

    # Print report
    print(f"  {'Role':<40} {'Model':<50} {'Input':<10} {'Output':<10} {'Requests':<10} {'Est. Cost'}")
    print(f"  {'-'*40} {'-'*50} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    total_cost = 0.0
    for row in results:
        role = row["role_name"]
        model = row["modelId"]
        input_tokens = int(row["input_tokens"])
        output_tokens = int(row["output_tokens"])
        requests = int(row["requests"])

        cost = calculate_cost(input_tokens, output_tokens, model)
        total_cost += cost

        print(f"  {role:<40} {model:<50} {input_tokens:<10} {output_tokens:<10} {requests:<10} ${cost:.4f}")

    print(f"\n  {'Total estimated cost:':<103} ${total_cost:.4f}")
    print("\n--- Done ---")


if __name__ == "__main__":
    main()
