"""
Amazon Bedrock Projects - Invoke Models Through Projects

This script invokes models through Bedrock Projects created by
4-1_setup_projects.py. Costs are attributed to each project's tags
in Cost Explorer.

You will learn how to:
- Route OpenAI SDK calls through projects using the project parameter
- Send raw HTTP requests with the OpenAI-Project header
- Run a multi-step order fulfillment agent

Prerequisites:
- Run 4-1_setup_projects.py first to create the projects
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

# Project names created by 4-1_setup_projects.py
PROJECT_NAMES = [
    "Order Fulfillment Agent - Staging",
    "Order Fulfillment Agent - Development",
    "Order Fulfillment Agent - Production",
    "Inventory Forecasting Agent - Production",
]


# ============================================================
# Helper Functions
# ============================================================

def find_project_by_name(name: str) -> dict | None:
    """Find an existing project by name."""
    url = f"{MANTLE_BASE_URL}/v1/organization/projects"
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    result = response.json()

    for project in result.get("data", []):
        if project.get("name") == name:
            return project
    return None


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
    # Resolve project IDs by name
    print("--- Resolving Projects ---")
    project_ids = {}
    for name in PROJECT_NAMES:
        project = find_project_by_name(name)
        if project:
            project_ids[name] = project["id"]
            print(f"  Found: {name} ({project['id']})")
        else:
            print(f"  NOT FOUND: {name} — run 4-1_setup_projects.py first")

    if not project_ids:
        print("\n  No projects found. Please run 4-1_setup_projects.py first.")
        return
    print()

    # Step 1: Inference using the OpenAI SDK with project
    staging_id = project_ids.get("Order Fulfillment Agent - Staging")
    if staging_id:
        print("--- Step 1: Inference with Project (OpenAI SDK) ---")
        result = invoke_with_project_sdk(
            project_id=staging_id,
            user_message="Summarize the top 3 reasons why order fulfillment latency increases during peak shopping seasons.",
            instructions="You are an e-commerce operations expert. Be concise.",
        )
        print(result)
        print()

        # Step 2: Inference using raw HTTP with project header
        print("--- Step 2: Inference with Project (HTTP/requests) ---")
        result = invoke_with_project_http(
            project_id=staging_id,
            user_message="What metrics should we track to optimize order-to-delivery time?",
        )
        print(json.dumps(result, indent=2))
        print()

        # Step 3: Multi-step order fulfillment agent
        print("--- Step 3: Order Fulfillment Agent (Multi-step) ---")
        order_fulfillment_agent(staging_id)

    # Step 4: Invoke through different projects
    print("--- Step 4: Invoke Through Multiple Projects ---")
    queries = {
        "Order Fulfillment Agent - Development": "How should I handle partial order cancellations in the fulfillment pipeline?",
        "Order Fulfillment Agent - Production": "A customer wants to change their shipping address after the order entered picking. What's the process?",
        "Inventory Forecasting Agent - Production": "Based on last year's holiday season data, what inventory levels should we maintain for our top 10 SKUs?",
    }

    for name, query in queries.items():
        pid = project_ids.get(name)
        if not pid:
            print(f"  [{name}] Skipped — project not found\n")
            continue

        client = OpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=AUTH_TOKEN,
            project=pid,
        )

        response = client.responses.create(
            model=MODEL_ID,
            input=query,
        )
        print(f"  [{name}]")
        print(f"  Query: {query}")
        print(f"  Response: {response.output_text}\n")

    print("--- Done ---")
    print("  Next steps:")
    print("  1. Wait ~24 hours for tags to appear in AWS Billing > Cost Allocation Tags")
    print("  2. Activate the bedrock:projects:* tags")
    print("  3. View per-application costs in Cost Explorer")


if __name__ == "__main__":
    main()
