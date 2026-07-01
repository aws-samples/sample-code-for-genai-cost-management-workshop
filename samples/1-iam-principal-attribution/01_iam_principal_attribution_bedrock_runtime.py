"""
Amazon Bedrock IAM Principal Attribution - Per-Developer Cost Tracking

This sample demonstrates how to use IAM principal tags for per-developer
and per-team cost attribution on the bedrock-runtime endpoint.

You will learn how to:
- Tag IAM users/roles with cost allocation attributes
- Make inference calls as different tagged principals
- Verify tags are set correctly for cost attribution
- Simulate multiple developers making Bedrock calls

Tags used: bedrock:iam-principal:Application, bedrock:iam-principal:Environment,
           bedrock:iam-principal:Team, bedrock:iam-principal:CostCenter

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with permissions for iam:TagUser, iam:TagRole, iam:ListUserTags,
  iam:ListRoleTags, sts:AssumeRole, and bedrock-runtime:Converse
- Access to Claude or Nova models on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import json
import boto3

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]

# Boto3 clients
iam_client = boto3.client("iam")
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)

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


# ============================================================
# IAM Role Creation & Tagging Functions
# ============================================================

BEDROCK_ASSUME_ROLE_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": f"arn:aws:iam::{ACCOUNT_ID}:root"},
            "Action": "sts:AssumeRole",
        }
    ],
})

BEDROCK_INVOKE_POLICY = json.dumps({
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:Converse",
                "bedrock:ConverseStream",
            ],
            "Resource": "*",
        }
    ],
})


def get_or_create_role(role_name: str, tags: dict) -> str:
    """
    Get an existing IAM role or create a new one with Bedrock invoke permissions.
    Tags the role with cost allocation attributes.
    Returns the role ARN.
    """
    try:
        response = iam_client.get_role(RoleName=role_name)
        role_arn = response["Role"]["Arn"]
        print(f"  Role '{role_name}' already exists: {role_arn}")
    except iam_client.exceptions.NoSuchEntityException:
        # Create the role
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=BEDROCK_ASSUME_ROLE_POLICY,
            Description=f"Workshop role for developer cost attribution - {role_name}",
        )
        role_arn = response["Role"]["Arn"]
        print(f"  Created role '{role_name}': {role_arn}")

        # Attach inline policy for Bedrock invoke
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName="BedrockInvokeAccess",
            PolicyDocument=BEDROCK_INVOKE_POLICY,
        )
        print(f"  Attached BedrockInvokeAccess policy to '{role_name}'")

        # Tag the role with cost allocation attributes only on creation
        tag_iam_role(role_name, tags)

    return role_arn


def delete_developer_roles(role_names: list) -> None:
    """Delete IAM roles created for the workshop demo."""
    for role_name in role_names:
        try:
            # Remove inline policies first
            policies = iam_client.list_role_policies(RoleName=role_name)
            for policy_name in policies.get("PolicyNames", []):
                iam_client.delete_role_policy(RoleName=role_name, PolicyName=policy_name)

            iam_client.delete_role(RoleName=role_name)
            print(f"  Deleted role: {role_name}")
        except iam_client.exceptions.NoSuchEntityException:
            print(f"  Role '{role_name}' does not exist (skipping)")


# ============================================================
# IAM Tagging Functions
# ============================================================

def tag_iam_user(username: str, tags: dict) -> None:
    """
    Tag an IAM user with cost allocation attributes.
    These tags will appear in Cost Explorer and CUR 2.0 once activated.
    """
    tag_list = [{"Key": k, "Value": v} for k, v in tags.items()]

    iam_client.tag_user(UserName=username, Tags=tag_list)
    print(f"  Tagged user '{username}' with {len(tags)} tags")


def tag_iam_role(role_name: str, tags: dict) -> None:
    """
    Tag an IAM role with cost allocation attributes.
    Useful for service roles, Lambda functions, or assumed roles.
    """
    tag_list = [{"Key": k, "Value": v} for k, v in tags.items()]

    iam_client.tag_role(RoleName=role_name, Tags=tag_list)
    print(f"  Tagged role '{role_name}' with {len(tags)} tags")


def get_user_tags(username: str) -> list:
    """Get the tags currently set on an IAM user."""
    response = iam_client.list_user_tags(UserName=username)
    return response.get("Tags", [])


def get_role_tags(role_name: str) -> list:
    """Get the tags currently set on an IAM role."""
    response = iam_client.list_role_tags(RoleName=role_name)
    return response.get("Tags", [])


def get_current_identity() -> dict:
    """Get the current caller identity (user/role ARN)."""
    sts = boto3.client("sts")
    return sts.get_caller_identity()


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
    # Optional: Set to True to delete all developer roles before running
    CLEAN_START = False

    developer_roles = [
        "bedrock-workshop-developer-alice",
        "bedrock-workshop-developer-bob",
        "bedrock-workshop-developer-carol",
    ]

    if CLEAN_START:
        print("--- Deleting All Developer Roles ---")
        delete_developer_roles(developer_roles)
        print()

    # Step 1: Show current identity
    print("--- Step 1: Current IAM Identity ---")
    identity = get_current_identity()
    print(f"  Account: {identity['Account']}")
    print(f"  ARN: {identity['Arn']}")
    print()

    # Step 2: Create and tag IAM roles for each developer
    print("--- Step 2: Creating & Tagging Developer Roles ---")

    developers = [
        {
            "role_name": "bedrock-workshop-developer-alice",
            "tags": {
                "bedrock:iam-principal:Application": "CodeAssistant",
                "bedrock:iam-principal:Environment": "Development",
                "bedrock:iam-principal:Team": "BackendEngineering",
                "bedrock:iam-principal:CostCenter": "ENG-2200",
            },
        },
        {
            "role_name": "bedrock-workshop-developer-bob",
            "tags": {
                "bedrock:iam-principal:Application": "CodeAssistant",
                "bedrock:iam-principal:Environment": "Development",
                "bedrock:iam-principal:Team": "FrontendEngineering",
                "bedrock:iam-principal:CostCenter": "ENG-2300",
            },
        },
        {
            "role_name": "bedrock-workshop-developer-carol",
            "tags": {
                "bedrock:iam-principal:Application": "CodeAssistant",
                "bedrock:iam-principal:Environment": "Development",
                "bedrock:iam-principal:Team": "DataScience",
                "bedrock:iam-principal:CostCenter": "ENG-2400",
            },
        },
    ]

    role_arns = {}
    for dev in developers:
        role_arn = get_or_create_role(dev["role_name"], dev["tags"])
        role_arns[dev["role_name"]] = role_arn
    print()

    # Step 3: Verify tags on the roles
    print("--- Step 3: Verify Role Tags ---")
    for dev in developers:
        tags = get_role_tags(dev["role_name"])
        print(f"  {dev['role_name']}:")
        for tag in tags:
            print(f"    {tag['Key']} = {tag['Value']}")
    print()

    # Step 4: Make inference calls by assuming each developer's role
    print("--- Step 4: Inference as Different Developers (Assumed Roles) ---")
    print("  Each call is attributed to the assumed role's tags in Cost Explorer.\n")

    import time
    # Brief pause to allow IAM role propagation
    print("  Waiting 10 seconds for IAM role propagation...")
    time.sleep(10)

    developer_tasks = [
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

    for task in developer_tasks:
        role_arn = role_arns[task["role_name"]]
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
