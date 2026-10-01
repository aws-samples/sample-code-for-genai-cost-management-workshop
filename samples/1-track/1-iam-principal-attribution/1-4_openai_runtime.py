"""
Amazon Bedrock IAM Principal Attribution - Invoke OpenAI Models via Converse,
Chat Completions, and Responses APIs (bedrock-runtime)

This script assumes the IAM roles created by 1-1_setup_iam_roles.py and makes
inference calls to the OpenAI models on Amazon Bedrock using three different
API surfaces exposed by the bedrock-runtime endpoint:
- Converse API: the native Bedrock SDK call, SigV4-signed via boto3
- Chat Completions API: the OpenAI SDK, pointed at the bedrock-runtime
  OpenAI-compatible endpoint
- Responses API: the OpenAI SDK, also pointed at the bedrock-runtime
  OpenAI-compatible endpoint (unlike 1-3_mantle_invoke_models.py, which uses
  bedrock-mantle for the Responses API)

Each developer role in this sample invokes its model through a different
method, so you can compare how cost attribution flows from role tags
regardless of which API surface is used.

You will learn how to:
- Assume tagged IAM roles
- Make Converse, Chat Completions, and Responses API calls to OpenAI models
  as different developers, all via bedrock-runtime
- See how cost attribution flows from role tags across all three API surfaces

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

# The bedrock-runtime endpoint exposes an OpenAI-compatible surface for the
# Chat Completions and Responses APIs, alongside the native Converse API.
BEDROCK_RUNTIME_BASE_URL = f"https://bedrock-runtime.{REGION}.amazonaws.com"
OPENAI_BASE_URL = f"{BEDROCK_RUNTIME_BASE_URL}/openai/v1"

# OpenAI models on Amazon Bedrock used in this sample.
# GPT-5.6 models only support INFERENCE_PROFILE invocation on bedrock-runtime
# (no on-demand throughput for the bare model ID), so we use the "global."
# cross-region inference profile IDs, which route to any supported commercial
# AWS region for maximum throughput:
#   - global.openai.gpt-5.6-sol (frontier reasoning and agentic coding)
# This sample varies the API surface (Converse, Chat Completions, Responses)
# per developer rather than the model, so all three route to the same model.
MODELS = {
    "gpt-5.6-sol": "global.openai.gpt-5.6-sol",
}

# Developer tasks - each developer assumes their role and invokes their model
# through a different API surface, so you can compare cost attribution across
# Converse, Chat Completions, and Responses on the same bedrock-runtime
# endpoint.
DEVELOPER_TASKS = [
    {
        "role_name": "bedrock-workshop-developer-alice",
        "session": "alice-coding-session",
        "team": "BackendEngineering",
        "model": MODELS["gpt-5.6-sol"],
        "method": "converse",
        "message": "Write a REST API endpoint in Python Flask that handles user authentication with JWT tokens.",
    },
    {
        "role_name": "bedrock-workshop-developer-bob",
        "session": "bob-coding-session",
        "team": "FrontendEngineering",
        "model": MODELS["gpt-5.6-sol"],
        "method": "chat_completions",
        "message": "Write a React component that displays a paginated data table with sorting and filtering.",
    },
    {
        "role_name": "bedrock-workshop-developer-carol",
        "session": "carol-coding-session",
        "team": "DataScience",
        "model": MODELS["gpt-5.6-sol"],
        "method": "responses",
        "message": "Write a Python script that loads a CSV, trains a random forest classifier, and outputs feature importances.",
    },
]


# ============================================================
# Credential Helpers
# ============================================================

def assume_role(role_arn: str, session_name: str) -> dict:
    """Assume an IAM role and return the temporary credentials."""
    sts = boto3.client("sts")
    return sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName=session_name,
    )["Credentials"]


def get_bearer_token_for_credentials(credentials: dict) -> str:
    """
    Generate a bearer token for the bedrock-runtime OpenAI-compatible endpoint
    from a set of assumed-role credentials. The cost is attributed to the
    assumed role's tags.
    """
    from botocore.credentials import RefreshableCredentials, CredentialProvider
    from aws_bedrock_token_generator import provide_token

    class AssumedRoleCredentialProvider(CredentialProvider):
        METHOD = "assumed-role"

        def load(self):
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

    return provide_token(region=REGION, aws_credentials_provider=AssumedRoleCredentialProvider())


# ============================================================
# Inference Functions - one per API surface
# ============================================================

def invoke_via_converse(credentials: dict, user_message: str, model_id: str) -> str:
    """
    Make a native Converse API call via bedrock-runtime, SigV4-signed with
    the assumed role's credentials.
    """
    assumed_runtime = boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )

    messages = [{"role": "user", "content": [{"text": user_message}]}]

    response = assumed_runtime.converse(
        modelId=model_id,
        messages=messages,
        inferenceConfig={"maxTokens": 1024},
    )

    # GPT-5.6 models are reasoning models: the content list may include a
    # reasoningContent block before the text block, so scan for the text
    # block instead of assuming it is at index 0.
    content_blocks = response["output"]["message"]["content"]
    for block in content_blocks:
        if "text" in block:
            return block["text"]

    raise ValueError(f"No text content block found in response: {content_blocks}")


def invoke_via_chat_completions(credentials: dict, user_message: str, model_id: str) -> str:
    """
    Make a Chat Completions API call via the bedrock-runtime OpenAI-compatible
    endpoint, authenticated with a bearer token derived from the assumed
    role's credentials.
    """
    token = get_bearer_token_for_credentials(credentials)

    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=token,
    )

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": user_message}],
        max_completion_tokens=1024,
    )

    return response.choices[0].message.content


def invoke_via_responses(credentials: dict, user_message: str, model_id: str) -> str:
    """
    Make a Responses API call via the bedrock-runtime OpenAI-compatible
    endpoint, authenticated with a bearer token derived from the assumed
    role's credentials.
    """
    token = get_bearer_token_for_credentials(credentials)

    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=token,
    )

    response = client.responses.create(
        model=model_id,
        input=user_message,
    )

    return response.output_text


# Maps each developer task's "method" field to its corresponding invoke
# function, so main() can dispatch without a chain of if/elif branches.
INVOKE_METHODS = {
    "converse": invoke_via_converse,
    "chat_completions": invoke_via_chat_completions,
    "responses": invoke_via_responses,
}


# ============================================================
# Main
# ============================================================

def main():
    print("--- Invoking OpenAI Models via Converse, Chat Completions, and Responses APIs (bedrock-runtime) ---")
    print("  Each developer role uses a different API surface; costs are attributed to the role's tags.\n")

    for task in DEVELOPER_TASKS:
        role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{task['role_name']}"
        method_name = task["method"]
        print(f"  [{task['session']}] ({task['team']}) - method: {method_name} - model: {task['model']}")
        try:
            credentials = assume_role(role_arn, task["session"])
            invoke = INVOKE_METHODS[method_name]
            result = invoke(credentials, task["message"], task["model"])
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
