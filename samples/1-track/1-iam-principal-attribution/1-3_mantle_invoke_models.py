"""
Amazon Bedrock IAM Principal Attribution - Invoke Models via Responses API (bedrock-mantle)

This script assumes the IAM roles created by 1-1_setup_iam_roles.py and makes
inference calls using the Responses API on the bedrock-mantle endpoint.
The cost of each call is attributed to the assumed role's tags in Cost Explorer.

You will learn how to:
- Assume tagged IAM roles
- Generate bearer tokens from assumed-role credentials
- Make Responses API calls as different developers via bedrock-mantle
- See how cost attribution flows from role tags on the mantle endpoint

Prerequisites:
- Run 1-1_setup_iam_roles.py first to create and tag the developer roles
- IAM credentials with sts:AssumeRole permission
- Access to OpenAI models on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import boto3
from openai import OpenAI

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

# The bedrock-mantle endpoint for the OpenAI-compatible Responses API
MANTLE_BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws"
OPENAI_BASE_URL = f"{MANTLE_BASE_URL}/openai/v1"

# Models used in this sample (bedrock-mantle supports many models via the Responses API):
#   - openai.gpt-5.6-sol (frontier reasoning and agentic coding)
# Each developer assumes a different IAM role, so costs are attributed per
# developer even though they share the same model.
MODELS = {
    "gpt-5.6-sol": "openai.gpt-5.6-sol",
}

# Developer tasks - each developer assumes their role and sends a prompt
DEVELOPER_TASKS = [
    {
        "role_name": "bedrock-workshop-developer-alice",
        "session": "alice-coding-session",
        "team": "BackendEngineering",
        "model": MODELS["gpt-5.6-sol"],
        "message": "Write a REST API endpoint in Python Flask that handles user authentication with JWT tokens.",
    },
    {
        "role_name": "bedrock-workshop-developer-bob",
        "session": "bob-coding-session",
        "team": "FrontendEngineering",
        "model": MODELS["gpt-5.6-sol"],
        "message": "Write a React component that displays a paginated data table with sorting and filtering.",
    },
    {
        "role_name": "bedrock-workshop-developer-carol",
        "session": "carol-coding-session",
        "team": "DataScience",
        "model": MODELS["gpt-5.6-sol"],
        "message": "Write a Python script that loads a CSV, trains a random forest classifier, and outputs feature importances.",
    },
]


# ============================================================
# Inference Functions
# ============================================================

def get_bearer_token_for_role(role_arn: str, session_name: str) -> str:
    """
    Assume an IAM role and generate a bearer token for the bedrock-mantle endpoint.
    The cost is attributed to the assumed role's tags.
    """
    from botocore.credentials import DeferredRefreshableCredentials, CredentialProvider

    sts = boto3.client("sts")
    credentials = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName=session_name,
    )["Credentials"]

    # Build a credential provider that supplies the assumed-role credentials
    class AssumedRoleCredentialProvider(CredentialProvider):
        METHOD = "assumed-role"

        def load(self):
            from botocore.credentials import RefreshableCredentials
            from datetime import datetime, timezone

            return RefreshableCredentials(
                access_key=credentials["AccessKeyId"],
                secret_key=credentials["SecretAccessKey"],
                token=credentials["SessionToken"],
                expiry_time=credentials["Expiration"],
                refresh_using=lambda: {
                    "access_key": credentials["AccessKeyId"],
                    "secret_key": credentials["SecretAccessKey"],
                    "token": credentials["SessionToken"],
                    "expiry_time": credentials["Expiration"],
                },
                method=self.METHOD,
            )

    # Generate a bearer token from the assumed role credentials
    from aws_bedrock_token_generator import provide_token
    return provide_token(region=REGION, aws_credentials_provider=AssumedRoleCredentialProvider())


def invoke_with_assumed_role(role_arn: str, session_name: str, user_message: str, model_id: str = None) -> str:
    """
    Assume an IAM role and make a Responses API call via bedrock-mantle.
    The cost is attributed to the assumed role's tags.
    The session_name identifies the individual developer.
    """
    token = get_bearer_token_for_role(role_arn, session_name)

    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=token,
    )

    response = client.responses.create(
        model=model_id or MODELS["gpt-5.6-sol"],
        input=user_message,
    )

    return response.output_text


# ============================================================
# Main
# ============================================================

def main():
    print("--- Invoking Models via Responses API (bedrock-mantle) with Assumed Developer Roles ---")
    print("  Each call is attributed to the assumed role's tags in Cost Explorer.\n")

    for task in DEVELOPER_TASKS:
        role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{task['role_name']}"
        print(f"  [{task['session']}] ({task['team']}) - model: {task['model']}")
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
