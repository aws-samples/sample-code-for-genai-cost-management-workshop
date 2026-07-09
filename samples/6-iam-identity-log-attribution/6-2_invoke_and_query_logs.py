"""
Amazon Bedrock IAM Identity Log Attribution - Invoke Models & Query Logs

This script assumes the IAM roles created by 6-1_setup_iam_roles.py, makes
Bedrock inference calls, and then queries CloudWatch Logs Insights to show
per-identity token usage from the model invocation logs.

You will learn how to:
- Assume IAM roles and make inference calls (identity appears in logs)
- Query CloudWatch Logs Insights for per-role token attribution
- Get near real-time visibility without waiting for CUR/Cost Explorer

Prerequisites:
- Run 6-1_setup_iam_roles.py first to create the developer roles
- Model invocation logging enabled in Amazon Bedrock (CloudWatch Logs destination)
- IAM credentials with sts:AssumeRole and logs:StartQuery permissions
- Access to Claude or Nova models on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import boto3

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

# Model to use for inference calls
# Using Global cross-region inference profiles for maximum throughput:
#   - global.amazon.nova-2-lite-v1:0 (Amazon Nova 2 Lite — fast, cost-effective)
#   - global.anthropic.claude-sonnet-4-6 (Claude Sonnet 4.6 — balanced)
#   - global.anthropic.claude-haiku-4-5-20251001-v1:0 (Claude Haiku 4.5 — low latency)
MODELS = {
    "nova-2-lite": "global.amazon.nova-2-lite-v1:0",
    "claude-sonnet-4-6": "global.anthropic.claude-sonnet-4-6",
    "claude-haiku-4-5": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
}

# Developer tasks — each developer assumes their role and sends a prompt
DEVELOPER_TASKS = [
    {
        "role_name": "bedrock-workshop-developer-alice",
        "session": "alice-coding-session",
        "model": MODELS["claude-sonnet-4-6"],
        "message": "Write a complete REST API in Python Flask with endpoints for user registration, login with JWT tokens, password reset, and profile update. Include input validation, error handling, and docstrings for each endpoint.",
    },
    {
        "role_name": "bedrock-workshop-developer-bob",
        "session": "bob-coding-session",
        "model": MODELS["claude-haiku-4-5"],
        "message": "Write a full React component with TypeScript that implements a data table with pagination, column sorting, text filtering, row selection, and export to CSV. Include all type definitions and styled-components for the layout.",
    },
    {
        "role_name": "bedrock-workshop-developer-carol",
        "session": "carol-coding-session",
        "model": MODELS["nova-2-lite"],
        "message": "Write a complete Python machine learning pipeline that loads a CSV dataset, performs exploratory data analysis with summary statistics, handles missing values, encodes categorical features, splits into train/test, trains a random forest classifier with hyperparameter tuning using GridSearchCV, evaluates with confusion matrix and classification report, and outputs feature importances as a sorted table.",
    },
]


# ============================================================
# Inference Functions
# ============================================================

def invoke_with_assumed_role(role_arn: str, session_name: str, user_message: str, model_id: str = None) -> str:
    """
    Assume an IAM role and make a Converse API call.
    The caller identity (role ARN + session name) is automatically recorded
    in the model invocation logs as identity.arn.
    """
    sts = boto3.client("sts")
    credentials = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName=session_name,
    )["Credentials"]

    # Create a bedrock-runtime client with the assumed role credentials
    assumed_runtime = boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )

    messages = [{"role": "user", "content": [{"text": user_message}]}]

    response = assumed_runtime.converse(
        modelId=model_id or MODELS["claude-haiku-4-5"],
        messages=messages,
        inferenceConfig={"maxTokens": 1024},
    )
    return response["output"]["message"]["content"][0]["text"]


# ============================================================
# Main
# ============================================================

def main():
    # Step 1: Invoke models as different developers
    print("--- Step 1: Invoking Models with Assumed Developer Roles ---")
    print("  Each call's identity is recorded in model invocation logs.\n")

    for task in DEVELOPER_TASKS:
        role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{task['role_name']}"
        print(f"  [{task['session']}] — model: {task['model']}")
        try:
            result = invoke_with_assumed_role(
                role_arn=role_arn,
                session_name=task["session"],
                user_message=task["message"],
                model_id=task["model"],
            )
            # Print just the first 200 chars to keep output manageable
            print(f"  Response: {result[:200]}...\n")
        except Exception as e:
            print(f"  Error: {e}\n")

    print("--- Done ---")
    print("  Identity ARNs were recorded in model invocation logs for each request.")
    print()
    print("  Next steps:")
    print("  1. Ensure model invocation logging is enabled in Amazon Bedrock settings")
    print("  2. Check CloudWatch Logs Insights for per-identity token usage")
    print("  3. Query logs with the sample queries in README.md")
    print("  4. Build CloudWatch dashboards for per-developer usage monitoring")


if __name__ == "__main__":
    main()
