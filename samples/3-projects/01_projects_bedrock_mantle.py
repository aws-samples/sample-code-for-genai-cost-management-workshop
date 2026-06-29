"""
Amazon Bedrock Projects - Cost Attribution with the OpenAI-compatible Responses API

This sample demonstrates how to use Amazon Bedrock Projects for per-application
cost attribution on the OpenAI-compatible Responses API (bedrock-mantle endpoint).

You will learn how to:
- Create and tag projects with cost allocation attributes
- Route OpenAI SDK calls through projects using the project parameter
- Track costs across multiple environments using separate projects
- Simulate an order fulfillment agent with multi-step reasoning

Tags used: bedrock:projects:Application, bedrock:projects:Environment,
           bedrock:projects:Team, bedrock:projects:CostCenter

Prerequisites:
- An AWS account with Amazon Bedrock access
- A Bedrock API key or IAM credentials for bearer token generation
- Access to OpenAI models on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import json
import requests
from openai import OpenAI

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

# The bedrock-mantle endpoint for the OpenAI-compatible Responses API
MANTLE_BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws"
OPENAI_BASE_URL = f"{MANTLE_BASE_URL}/openai/v1"

# Models available on bedrock-mantle for the Responses API:
#   - openai.gpt-5.5 (most capable, advanced coding and reasoning)
#   - openai.gpt-5.4 (frontier reasoning, coding, tool use)
#   - openai.gpt-oss-120b (120B params, general purpose)
#   - openai.gpt-oss-20b (20B params, lower latency, cost-effective)
MODEL_ID = "openai.gpt-5.5"


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
# Inference Functions
# ============================================================

def invoke_with_project_sdk(project_id: str, user_message: str, instructions: str = None) -> str:
    """
    Send a Responses API request associated with a specific project
    using the OpenAI Python SDK.

    The project is specified via the `project` parameter in the OpenAI client.
    """
    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=AUTH_TOKEN,
        project=project_id,
    )

    response = client.responses.create(
        model=MODEL_ID,
        instructions=instructions,
        input=user_message,
    )

    return response.output_text


def invoke_with_project_http(project_id: str, user_message: str) -> dict:
    """
    Send a Responses API request associated with a specific project
    using raw HTTP requests (curl-equivalent).
    """
    url = f"{OPENAI_BASE_URL}/responses"
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "OpenAI-Project": project_id,
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "input": user_message,
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def order_fulfillment_agent(project_id: str) -> None:
    """
    Simulate an order fulfillment agent that processes a multi-step workflow.
    All calls are attributed to the same project for cost tracking.
    """
    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=AUTH_TOKEN,
        project=project_id,
    )

    instructions = (
        "You are an order fulfillment agent for an e-commerce platform. "
        "You help process orders by validating inventory, calculating shipping, "
        "and confirming order details. Be concise and structured in your responses."
    )

    # Step 1: Validate the order
    print("  Step 1: Validating order...")
    response = client.responses.create(
        model=MODEL_ID,
        instructions=instructions,
        input=(
            "New order received: 2x Wireless Headphones ($79.99 each), "
            "1x USB-C Charging Cable ($12.99). "
            "Customer: Jane Smith, Shipping: Seattle, WA. "
            "Validate this order and check for any issues."
        ),
    )
    print(f"  {response.output_text}\n")

    # Step 2: Calculate shipping options
    print("  Step 2: Calculating shipping...")
    response = client.responses.create(
        model=MODEL_ID,
        instructions=instructions,
        input=(
            "Order total: $172.97. Destination: Seattle, WA (ZIP 98101). "
            "Package weight: 1.2 lbs. "
            "Provide shipping options with estimated delivery dates and costs."
        ),
    )
    print(f"  {response.output_text}\n")

    # Step 3: Generate confirmation
    print("  Step 3: Generating order confirmation...")
    response = client.responses.create(
        model=MODEL_ID,
        instructions=instructions,
        input=(
            "Generate an order confirmation email for Jane Smith. "
            "Order #ORD-92847: 2x Wireless Headphones, 1x USB-C Cable. "
            "Total: $172.97 + $8.99 shipping = $181.96. "
            "Estimated delivery: 3-5 business days. "
            "Keep it brief and professional."
        ),
    )
    print(f"  {response.output_text}\n")


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

    # Step 2: Get or create a project with tags for cost allocation
    print("--- Step 2: Get or Create a Project ---")
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
    project_id = project["id"]
    print(f"Project ID: {project_id}\n")

    # Step 3: Inference using the OpenAI SDK with project
    print("--- Step 3: Inference with Project (OpenAI SDK) ---")
    result = invoke_with_project_sdk(
        project_id=project_id,
        user_message="Summarize the top 3 reasons why order fulfillment latency increases during peak shopping seasons.",
        instructions="You are an e-commerce operations expert. Be concise.",
    )
    print(result)
    print()

    # Step 4: Inference using raw HTTP with project header
    print("--- Step 4: Inference with Project (HTTP/requests) ---")
    result = invoke_with_project_http(
        project_id=project_id,
        user_message="What metrics should we track to optimize order-to-delivery time?",
    )
    print(json.dumps(result, indent=2))
    print()

    # Step 5: Multi-step order fulfillment agent
    print("--- Step 5: Order Fulfillment Agent (Multi-step) ---")
    order_fulfillment_agent(project_id)

    # Step 6: Multiple projects for different environments
    print("--- Step 6: Multiple Projects (Environments) ---")
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

    queries = [
        "How should I handle partial order cancellations in the fulfillment pipeline?",
        "A customer wants to change their shipping address after the order entered picking. What's the process?",
        "Based on last year's holiday season data, what inventory levels should we maintain for our top 10 SKUs?",
    ]

    for env, query in zip(environments, queries):
        proj = get_or_create_project(name=env["name"], tags=env["tags"])

        # Create a new client with the project context
        project_client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=AUTH_TOKEN,
            project=proj["id"],
        )

        response = project_client.responses.create(
            model=MODEL_ID,
            input=query,
        )
        print(f"  [{env['name']}]")
        print(f"  Query: {query}")
        print(f"  Response: {response.output_text}\n")


if __name__ == "__main__":
    main()
