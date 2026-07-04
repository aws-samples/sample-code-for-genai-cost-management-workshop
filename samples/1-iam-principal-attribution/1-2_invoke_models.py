"""
Amazon Bedrock IAM Principal Attribution - Invoke Models with Assumed Roles

This script assumes the IAM roles created by 01_setup_iam_roles.py and makes
Bedrock inference calls. The cost of each call is attributed to the assumed
role's tags in Cost Explorer.

You will learn how to:
- Assume tagged IAM roles
- Make inference calls as different developers
- See how cost attribution flows from role tags

Prerequisites:
- Run 1-1_setup_iam_roles.py first to create and tag the developer roles
- IAM credentials with sts:AssumeRole permission
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
        "team": "BackendEngineering",
        "model": MODELS["claude-sonnet-4-6"],
        "message": "Write a REST API endpoint in Python Flask that handles user authentication with JWT tokens.",
    },
    {
        "role_name": "bedrock-workshop-developer-bob",
        "session": "bob-coding-session",
        "team": "FrontendEngineering",
        "model": MODELS["claude-haiku-4-5"],
        "message": "Write a React component that displays a paginated data table with sorting and filtering.",
    },
    {
        "role_name": "bedrock-workshop-developer-carol",
        "session": "carol-coding-session",
        "team": "DataScience",
        "model": MODELS["nova-2-lite"],
        "message": "Write a Python script that loads a CSV, trains a random forest classifier, and outputs feature importances.",
    },
]


# ============================================================
# Inference Functions
# ============================================================

def invoke_with_assumed_role(role_arn: str, session_name: str, user_message: str, model_id: str = None) -> str:
    """
    Assume an IAM role and make a Converse API call.
    The cost is attributed to the assumed role's tags.
    The session_name identifies the individual developer.
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
    print("--- Invoking Models with Assumed Developer Roles ---")
    print("  Each call is attributed to the assumed role's tags in Cost Explorer.\n")

    for task in DEVELOPER_TASKS:
        role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{task['role_name']}"
        print(f"  [{task['session']}] ({task['team']}) — model: {task['model']}")
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
    print("  Next steps:")
    print("  1. Wait ~24 hours for tags to appear in AWS Billing > Cost Allocation Tags")
    print("  2. Activate the bedrock:iam-principal:* tags")
    print("  3. Make additional inference calls")
    print("  4. After ~24 hours, view per-developer costs in Cost Explorer")


if __name__ == "__main__":
    main()
