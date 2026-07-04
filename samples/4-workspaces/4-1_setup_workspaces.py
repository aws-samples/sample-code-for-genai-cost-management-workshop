"""
Amazon Bedrock Workspaces - Workspace Setup & Tagging

This script handles the setup for per-application cost attribution using
Bedrock Workspaces on the Anthropic-compatible Messages API (bedrock-mantle endpoint).

You will learn how to:
- Create workspaces with cost allocation tags
- List and verify existing workspaces
- Create multiple workspaces for different support tiers

Tags used: bedrock:workspaces:Application, bedrock:workspaces:Environment,
           bedrock:workspaces:Team, bedrock:workspaces:CostCenter

Prerequisites:
- An AWS account with Amazon Bedrock access
- A Bedrock API key or IAM credentials for bearer token generation
- Dependencies installed via: pip install -r requirements.txt

After running this script, use 4-2_invoke_models.py to make inference calls
through the workspaces.
"""

import os
import json
import requests

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

# Authentication: Choose one of two methods
#
# Option 1: Bedrock API Key (static, created in the Bedrock console)
#   Export BEDROCK_API_KEY with your API key value.
#
# Option 2: Bearer Token from IAM credentials (recommended)
#   Uses your existing IAM role/credentials to generate a short-lived token.
#   No API key needed — just have valid AWS credentials configured.
#   Requires: pip install aws-bedrock-token-generator

BEDROCK_API_KEY = os.environ.get("BEDROCK_API_KEY")


def get_auth_token() -> str:
    """
    Get authentication token for the bedrock-mantle endpoint.
    Prefers BEDROCK_API_KEY if set, otherwise generates a bearer token
    from IAM credentials using aws-bedrock-token-generator.
    """
    if BEDROCK_API_KEY:
        return BEDROCK_API_KEY

    # Generate a bearer token from IAM credentials
    from aws_bedrock_token_generator import provide_token
    return provide_token(region=REGION)


AUTH_TOKEN = get_auth_token()

# The bedrock-mantle endpoint
MANTLE_BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws"


# ============================================================
# Workspace Management Functions
# ============================================================

def get_or_create_workspace(name: str, tags: dict) -> dict:
    """
    Get an existing workspace by name, or create a new one if it doesn't exist.
    Avoids creating duplicate workspaces with the same name.
    """
    # Check if a workspace with this name already exists
    existing = list_workspaces()
    for project in existing.get("data", []):
        if project.get("name") == name:
            print(f"  Workspace '{name}' already exists: {project['id']}")
            return project

    # Create a new one
    url = f"{MANTLE_BASE_URL}/v1/organization/projects"
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "name": name,
        "tags": tags,
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    print(f"  Created new workspace '{name}': {result['id']}")
    return result


def list_workspaces() -> dict:
    """List all workspaces (projects) in the account."""
    url = f"{MANTLE_BASE_URL}/v1/organization/projects"
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    result = response.json()

    for project in result.get("data", []):
        print(f"  {project.get('name', 'N/A')} — {project.get('arn', 'N/A')}")

    return result


def delete_all_workspaces() -> None:
    """Archive all workspaces (projects) in the account, skipping the default."""
    workspaces = list_workspaces()
    projects = workspaces.get("data", [])

    if not projects:
        print("  No workspaces to delete.")
        return

    count = 0
    for project in projects:
        project_id = project["id"]

        # Skip the default workspace — it cannot be archived
        if project_id == "default":
            print(f"  Skipping default workspace")
            continue

        url = f"{MANTLE_BASE_URL}/v1/organization/projects/{project_id}/archive"
        headers = {
            "Authorization": f"Bearer {AUTH_TOKEN}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers)
        if response.ok:
            print(f"  Archived workspace: {project.get('name', project_id)} ({project_id})")
            count += 1
        else:
            print(f"  Failed to archive {project_id}: {response.status_code} {response.text}")

    print(f"  Done. Archived {count} workspace(s).")


# ============================================================
# Main
# ============================================================

def main():
    # Optional: Set to True to delete all existing workspaces before running
    CLEAN_START = False

    if CLEAN_START:
        print("--- Deleting All Existing Workspaces ---")
        delete_all_workspaces()
        print()

    # Step 1: List existing workspaces
    print("--- Step 1: Listing Existing Workspaces ---")
    workspaces = list_workspaces()
    print(json.dumps(workspaces, indent=2))
    print()

    # Step 2: Create the primary workspace with tags for cost allocation
    print("--- Step 2: Create Primary Workspace ---")
    workspace = get_or_create_workspace(
        name="Customer Support Agent - Production",
        tags={
            "bedrock:workspaces:Application": "CustomerSupportAgent",
            "bedrock:workspaces:Environment": "Production",
            "bedrock:workspaces:Team": "CustomerExperience",
            "bedrock:workspaces:CostCenter": "CX-5500",
        },
    )
    print(json.dumps(workspace, indent=2))
    print()

    # Step 3: Create additional workspaces for different support tiers
    print("--- Step 3: Create Additional Workspaces (Support Tiers) ---")
    tiers = [
        {
            "name": "Support Agent - Tier 1 (General)",
            "tags": {
                "bedrock:workspaces:Application": "CustomerSupportAgent",
                "bedrock:workspaces:Environment": "Production",
                "bedrock:workspaces:Team": "Tier1Support",
                "bedrock:workspaces:CostCenter": "CX-5501",
            },
        },
        {
            "name": "Support Agent - Tier 2 (Technical)",
            "tags": {
                "bedrock:workspaces:Application": "CustomerSupportAgent",
                "bedrock:workspaces:Environment": "Production",
                "bedrock:workspaces:Team": "Tier2Technical",
                "bedrock:workspaces:CostCenter": "CX-5502",
            },
        },
        {
            "name": "Support Agent - Tier 3 (Escalation)",
            "tags": {
                "bedrock:workspaces:Application": "CustomerSupportAgent",
                "bedrock:workspaces:Environment": "Production",
                "bedrock:workspaces:Team": "Tier3Escalation",
                "bedrock:workspaces:CostCenter": "CX-5503",
            },
        },
    ]

    for tier in tiers:
        get_or_create_workspace(name=tier["name"], tags=tier["tags"])
    print()

    print("--- Setup Complete ---")
    print("  Workspaces created and tagged.")
    print("  You can now run 4-2_invoke_models.py to make inference calls through the workspaces.")
    print()
    print("  Next steps:")
    print("  1. Run 4-2_invoke_models.py to invoke models through workspaces")
    print("  2. Wait ~24 hours for tags to appear in AWS Billing > Cost Allocation Tags")
    print("  3. Activate the bedrock:workspaces:* tags")
    print("  4. View per-application costs in Cost Explorer")


if __name__ == "__main__":
    main()
