"""
Amazon Bedrock IAM Principal Attribution - IAM Role Setup & Tagging

This script handles the IAM setup for per-developer cost attribution:
- Creates IAM roles for each developer
- Attaches Bedrock invoke permissions
- Tags roles with cost allocation attributes
- Verifies tags are set correctly

Tags used: bedrock:iam-principal:Application, bedrock:iam-principal:Environment,
           bedrock:iam-principal:Team, bedrock:iam-principal:CostCenter

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with permissions for iam:TagRole, iam:CreateRole,
  iam:PutRolePolicy, iam:ListRoleTags, iam:GetRole
- Dependencies installed via: pip install -r requirements.txt

After running this script, use 1-2_invoke_models.py to make inference calls
as different developers.
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

# Developer role definitions
DEVELOPERS = [
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


# ============================================================
# IAM Policies
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


# ============================================================
# IAM Role Creation & Tagging Functions
# ============================================================

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


def tag_iam_role(role_name: str, tags: dict) -> None:
    """
    Tag an IAM role with cost allocation attributes.
    Useful for service roles, Lambda functions, or assumed roles.
    """
    tag_list = [{"Key": k, "Value": v} for k, v in tags.items()]

    iam_client.tag_role(RoleName=role_name, Tags=tag_list)
    print(f"  Tagged role '{role_name}' with {len(tags)} tags")


def get_role_tags(role_name: str) -> list:
    """Get the tags currently set on an IAM role."""
    response = iam_client.list_role_tags(RoleName=role_name)
    return response.get("Tags", [])


def get_current_identity() -> dict:
    """Get the current caller identity (user/role ARN)."""
    sts = boto3.client("sts")
    return sts.get_caller_identity()


# ============================================================
# Main
# ============================================================

def main():
    # Optional: Set to True to delete all developer roles before running
    CLEAN_START = False

    developer_role_names = [dev["role_name"] for dev in DEVELOPERS]

    if CLEAN_START:
        print("--- Deleting All Developer Roles ---")
        delete_developer_roles(developer_role_names)
        print()

    # Step 1: Show current identity
    print("--- Step 1: Current IAM Identity ---")
    identity = get_current_identity()
    print(f"  Account: {identity['Account']}")
    print(f"  ARN: {identity['Arn']}")
    print()

    # Step 2: Create and tag IAM roles for each developer
    print("--- Step 2: Creating & Tagging Developer Roles ---")

    role_arns = {}
    for dev in DEVELOPERS:
        role_arn = get_or_create_role(dev["role_name"], dev["tags"])
        role_arns[dev["role_name"]] = role_arn
    print()

    # Step 3: Verify tags on the roles
    print("--- Step 3: Verify Role Tags ---")
    for dev in DEVELOPERS:
        tags = get_role_tags(dev["role_name"])
        print(f"  {dev['role_name']}:")
        for tag in tags:
            print(f"    {tag['Key']} = {tag['Value']}")
    print()

    # Wait for IAM role propagation before invoking models
    print("--- Waiting 10 seconds for IAM role propagation... ---")
    import time
    time.sleep(10)
    print()

    print("--- Setup Complete ---")
    print("  Roles created and tagged. You can now run 1-2_invoke_models.py")
    print("  to make inference calls as different developers.")
    print()
    print("  Next steps:")
    print("  1. Run 1-2_invoke_models.py to invoke models with assumed roles")
    print("  3. Wait ~24 hours for tags to appear in AWS Billing > Cost Allocation Tags")
    print("  4. Activate the bedrock:iam-principal:* tags")
    print("  5. View per-developer costs in Cost Explorer")


if __name__ == "__main__":
    main()
