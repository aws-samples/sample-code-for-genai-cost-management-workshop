"""
Amazon Bedrock Projects - Project Setup & Tagging

This script handles the setup for per-application cost attribution using
Bedrock Projects on the OpenAI-compatible Responses API (bedrock-mantle endpoint).

You will learn how to:
- Create projects with cost allocation tags
- List and verify existing projects
- Create multiple projects for different environments

Tags used: bedrock:projects:Application, bedrock:projects:Environment,
           bedrock:projects:Team, bedrock:projects:CostCenter

Prerequisites:
- An AWS account with Amazon Bedrock access
- A Bedrock API key or IAM credentials for bearer token generation
- Dependencies installed via: pip install -r requirements.txt

After running this script, use 3-2_invoke_models.py to make inference calls
through the projects.
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
# Project Management Functions
# ============================================================

def get_or_create_project(name: str, tags: dict) -> dict:
    """
    Get an existing project by name, or create a new one if it doesn't exist.
    Avoids creating duplicate projects with the same name.
    """
    # Check if a project with this name already exists
    existing = list_projects()
    for project in existing.get("data", []):
        if project.get("name") == name:
            print(f"  Project '{name}' already exists: {project['id']}")
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
    if response.status_code == 401:
        print(f"  ERROR: Unauthorized (401). Your API key or bearer token does not have")
        print(f"  permission to create projects. Ensure your IAM identity has")
        print(f"  bedrock:CreateProject permission, or regenerate your API key with full access.")
        response.raise_for_status()
    response.raise_for_status()
    result = response.json()
    print(f"  Created new project '{name}': {result['id']}")
    return result


def list_projects() -> dict:
    """List all projects in the account."""
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


def delete_all_projects() -> None:
    """Archive all projects in the account, skipping the default."""
    projects_response = list_projects()
    projects = projects_response.get("data", [])

    if not projects:
        print("  No projects to delete.")
        return

    count = 0
    for project in projects:
        project_id = project["id"]

        # Skip the default project — it cannot be archived
        if project_id == "default":
            print(f"  Skipping default project")
            continue

        url = f"{MANTLE_BASE_URL}/v1/organization/projects/{project_id}/archive"
        headers = {
            "Authorization": f"Bearer {AUTH_TOKEN}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, headers=headers)
        if response.ok:
            print(f"  Archived project: {project.get('name', project_id)} ({project_id})")
            count += 1
        else:
            print(f"  Failed to archive {project_id}: {response.status_code} {response.text}")

    print(f"  Done. Archived {count} project(s).")


# ============================================================
# Main
# ============================================================

def main():
    # Optional: Set to True to delete all existing projects before running
    CLEAN_START = False

    if CLEAN_START:
        print("--- Deleting All Existing Projects ---")
        delete_all_projects()
        print()

    # Step 1: List existing projects
    print("--- Step 1: Listing Existing Projects ---")
    projects = list_projects()
    print(json.dumps(projects, indent=2))
    print()

    # Step 2: Create the primary project with tags for cost allocation
    print("--- Step 2: Create Primary Project ---")
    project = get_or_create_project(
        name="Order Fulfillment Agent - Staging",
        tags={
            "bedrock:projects:Application": "OrderFulfillmentAgent",
            "bedrock:projects:Environment": "Staging",
            "bedrock:projects:Team": "CommerceEngineering",
            "bedrock:projects:CostCenter": "ECOM-3100",
        },
    )
    print(json.dumps(project, indent=2))
    print()

    # Step 3: Create additional projects for different environments
    print("--- Step 3: Create Additional Projects (Environments) ---")
    environments = [
        {
            "name": "Order Fulfillment Agent - Development",
            "tags": {
                "bedrock:projects:Application": "OrderFulfillmentAgent",
                "bedrock:projects:Environment": "Development",
                "bedrock:projects:Team": "CommerceEngineering",
                "bedrock:projects:CostCenter": "ECOM-3101",
            },
        },
        {
            "name": "Order Fulfillment Agent - Production",
            "tags": {
                "bedrock:projects:Application": "OrderFulfillmentAgent",
                "bedrock:projects:Environment": "Production",
                "bedrock:projects:Team": "CommerceEngineering",
                "bedrock:projects:CostCenter": "ECOM-3102",
            },
        },
        {
            "name": "Inventory Forecasting Agent - Production",
            "tags": {
                "bedrock:projects:Application": "InventoryForecastAgent",
                "bedrock:projects:Environment": "Production",
                "bedrock:projects:Team": "SupplyChainOps",
                "bedrock:projects:CostCenter": "ECOM-3200",
            },
        },
    ]

    for env in environments:
        get_or_create_project(name=env["name"], tags=env["tags"])
    print()

    print("--- Setup Complete ---")
    print("  Projects created and tagged.")
    print("  You can now run 3-2_invoke_models.py to make inference calls through the projects.")
    print()
    print("  Next steps:")
    print("  1. Run 3-2_invoke_models.py to invoke models through projects")
    print("  2. Wait ~24 hours for tags to appear in AWS Billing > Cost Allocation Tags")
    print("  3. Activate the bedrock:projects:* tags")
    print("  4. View per-application costs in Cost Explorer")


if __name__ == "__main__":
    main()
