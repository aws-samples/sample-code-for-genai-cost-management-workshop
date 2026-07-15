"""
Amazon Bedrock Workspaces - Invoke Models Through Workspaces

This script invokes models through Bedrock Workspaces created by
4-1_setup_workspaces.py. Costs are attributed to each workspace's tags
in Cost Explorer.

You will learn how to:
- Route Anthropic SDK calls through workspaces using the anthropic-workspace-id header
- Send raw HTTP requests with the anthropic-workspace-id header
- Run multi-turn conversations with full cost attribution

Prerequisites:
- Run 4-1_setup_workspaces.py first to create the workspaces
- A Bedrock API key or IAM credentials for bearer token generation
- Access to Claude models on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import json
import requests
import anthropic

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

# The bedrock-mantle endpoint for the Anthropic Messages API
MANTLE_BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws"
ANTHROPIC_BASE_URL = f"{MANTLE_BASE_URL}/anthropic"

# Models available on bedrock-mantle for the Messages API:
#   - anthropic.claude-haiku-4-5 (fast, cost-effective)
#   - anthropic.claude-opus-4-7 (powerful, 1M context)
#   - anthropic.claude-opus-4-8 (latest Opus)
MODEL_ID = "anthropic.claude-opus-4-8"

# Workspace names created by 4-1_setup_workspaces.py
WORKSPACE_NAMES = [
    "Customer Support Agent - Production",
    "Support Agent - Tier 1 (General)",
    "Support Agent - Tier 2 (Technical)",
    "Support Agent - Tier 3 (Escalation)",
]


# ============================================================
# Helper Functions
# ============================================================

def list_workspaces() -> dict:
    """List all workspaces (projects) in the account."""
    url = f"{MANTLE_BASE_URL}/v1/organization/projects"
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def find_workspace_by_name(name: str) -> dict | None:
    """Find an existing workspace by name."""
    result = list_workspaces()

    for project in result.get("data", []):
        if project.get("name") == name:
            return project
    return None


# ============================================================
# Inference Functions
# ============================================================

def invoke_with_workspace_sdk(workspace_id: str, user_message: str) -> str:
    """
    Send a Messages API request associated with a specific workspace
    using the Anthropic Python SDK.

    The workspace is specified via the `anthropic-workspace-id` default header,
    which ensures proper cost attribution to the workspace in Cost Explorer.
    """
    client = anthropic.Anthropic(
        base_url=ANTHROPIC_BASE_URL,
        api_key=AUTH_TOKEN,
        default_headers={"anthropic-workspace-id": workspace_id},
    )

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": user_message}
        ],
    )

    return response.content[0].text


def invoke_with_workspace_http(workspace_id: str, user_message: str) -> dict:
    """
    Send a Messages API request associated with a specific workspace
    using raw HTTP requests (curl-equivalent).
    """
    url = f"{ANTHROPIC_BASE_URL}/v1/messages"
    headers = {
        "x-api-key": AUTH_TOKEN,
        "anthropic-version": "2023-06-01",
        "anthropic-workspace-id": workspace_id,
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": user_message}
        ],
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def multi_turn_conversation(workspace_id: str) -> None:
    """
    Demonstrate a multi-turn customer support conversation within a workspace.
    All messages are attributed to the same workspace for cost tracking.
    """
    client = anthropic.Anthropic(
        base_url=ANTHROPIC_BASE_URL,
        api_key=AUTH_TOKEN,
        default_headers={"anthropic-workspace-id": workspace_id},
    )

    messages = []

    # Turn 1: Customer opens a support ticket
    messages.append({
        "role": "user",
        "content": "I placed an order 3 days ago (order #ORD-88421) and the tracking still shows 'Processing'. Can you check the status?",
    })

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=1024,
        system="You are a customer support agent for an e-commerce company. Be helpful, empathetic, and concise. If you need to look up information, explain what you're checking.",
        messages=messages,
    )

    assistant_reply = response.content[0].text
    messages.append({"role": "assistant", "content": assistant_reply})
    print(f"  Turn 1 (Agent): {assistant_reply}\n")

    # Turn 2: Customer provides additional context
    messages.append({
        "role": "user",
        "content": "Yes, it's shipping to 123 Main St, Seattle WA. I paid for express shipping.",
    })

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=1024,
        system="You are a customer support agent for an e-commerce company. Be helpful, empathetic, and concise. If you need to look up information, explain what you're checking.",
        messages=messages,
    )

    assistant_reply = response.content[0].text
    messages.append({"role": "assistant", "content": assistant_reply})
    print(f"  Turn 2 (Agent): {assistant_reply}\n")

    # Turn 3: Customer asks about refund policy
    messages.append({
        "role": "user",
        "content": "If it doesn't arrive by tomorrow, can I get a refund on the express shipping charge?",
    })

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=1024,
        system="You are a customer support agent for an e-commerce company. Be helpful, empathetic, and concise. If you need to look up information, explain what you're checking.",
        messages=messages,
    )

    assistant_reply = response.content[0].text
    print(f"  Turn 3 (Agent): {assistant_reply}\n")


# ============================================================
# Main
# ============================================================

def main():
    # Step 0: List all workspaces
    print("--- Listing All Workspaces ---")
    workspaces = list_workspaces()
    for project in workspaces.get("data", []):
        print(f"  {project.get('name', 'N/A')} — {project.get('id', 'N/A')}")
    print()

    # Resolve workspace IDs by name
    print("--- Resolving Workspaces ---")
    workspace_ids = {}
    for name in WORKSPACE_NAMES:
        ws = find_workspace_by_name(name)
        if ws:
            workspace_ids[name] = ws["id"]
            print(f"  Found: {name} ({ws['id']})")
        else:
            print(f"  NOT FOUND: {name} — run 4-1_setup_workspaces.py first")

    if not workspace_ids:
        print("\n  No workspaces found. Please run 4-1_setup_workspaces.py first.")
        return
    print()

    # Step 1: Inference using the Anthropic SDK with workspace header
    primary_id = workspace_ids.get("Customer Support Agent - Production")
    if primary_id:
        print("--- Step 1: Inference with Workspace (Anthropic SDK) ---")
        result = invoke_with_workspace_sdk(
            workspace_id=primary_id,
            user_message="A customer is asking about our return policy for electronics purchased more than 30 days ago. Summarize the key points I should mention.",
        )
        print(result)
        print()

        # Step 2: Inference using raw HTTP with workspace header
        print("--- Step 2: Inference with Workspace (HTTP/requests) ---")
        result = invoke_with_workspace_http(
            workspace_id=primary_id,
            user_message="Draft a polite response to a customer whose shipment was delayed by 2 days due to weather.",
        )
        print(json.dumps(result, indent=2))
        print()

        # Step 3: Multi-turn customer support conversation
        print("--- Step 3: Multi-turn Customer Support Conversation ---")
        multi_turn_conversation(primary_id)

    # Step 4: Invoke through different workspace tiers
    print("--- Step 4: Invoke Through Multiple Workspaces (Support Tiers) ---")

    support_queries = {
        "Support Agent - Tier 1 (General)": "How do I reset my password?",
        "Support Agent - Tier 2 (Technical)": "My API integration is returning 429 errors after the latest update. I'm using the v3 SDK with retry logic.",
        "Support Agent - Tier 3 (Escalation)": "I've been waiting 2 weeks for a resolution on ticket #ESC-1192. This is my third follow-up. I need to speak to a manager.",
    }

    for name, query in support_queries.items():
        ws_id = workspace_ids.get(name)
        if not ws_id:
            print(f"  [{name}] Skipped — workspace not found\n")
            continue

        client = anthropic.Anthropic(
            base_url=ANTHROPIC_BASE_URL,
            api_key=AUTH_TOKEN,
            default_headers={"anthropic-workspace-id": ws_id},
        )

        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=256,
            system="You are a customer support agent. Respond appropriately for your support tier level.",
            messages=[
                {"role": "user", "content": query}
            ],
        )
        print(f"  [{name}]")
        print(f"  Query: {query}")
        print(f"  Response: {response.content[0].text}\n")

    print("--- Done ---")
    print("  Next steps:")
    print("  1. Wait ~24 hours for tags to appear in AWS Billing > Cost Allocation Tags")
    print("  2. Activate the bedrock:workspaces:* tags")
    print("  3. View per-application costs in Cost Explorer")


if __name__ == "__main__":
    main()
