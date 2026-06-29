"""
Amazon Bedrock Workspaces - Cost Attribution with the bedrock-mantle endpoint

This sample demonstrates how to use Amazon Bedrock Workspaces for per-application
cost attribution on the Anthropic-compatible Messages API (bedrock-mantle endpoint).

You will learn how to:
- Create and tag workspaces with cost allocation attributes
- Route inference calls through workspaces using the `anthropic-workspace` header
- Track costs across multiple support tiers using separate workspaces
- Run multi-turn conversations with full cost attribution

Tags used: bedrock:workspaces:Application, bedrock:workspaces:Environment,
           bedrock:workspaces:Team, bedrock:workspaces:CostCenter

Prerequisites:
- An AWS account with Amazon Bedrock access
- A Bedrock API key (https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html)
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
BEDROCK_API_KEY = os.environ.get("BEDROCK_API_KEY", "<your bedrock api key>")

# The bedrock-mantle endpoint for the Anthropic Messages API
MANTLE_BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws"
ANTHROPIC_BASE_URL = f"{MANTLE_BASE_URL}/anthropic"

# Models available on bedrock-mantle for the Messages API:
#   - anthropic.claude-haiku-4-5 (fast, cost-effective)
#   - anthropic.claude-opus-4-7 (powerful, 1M context)
#   - anthropic.claude-opus-4-8 (latest Opus)
MODEL_ID = "anthropic.claude-haiku-4-5"


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
        "Authorization": f"Bearer {BEDROCK_API_KEY}",
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
        "Authorization": f"Bearer {BEDROCK_API_KEY}",
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
            "Authorization": f"Bearer {BEDROCK_API_KEY}",
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
# Inference Functions
# ============================================================

def invoke_with_workspace_sdk(workspace_id: str, user_message: str) -> str:
    """
    Send a Messages API request associated with a specific workspace
    using the Anthropic Python SDK.

    The workspace is specified via the `anthropic-workspace` extra header.
    """
    client = anthropic.Anthropic(
        base_url=ANTHROPIC_BASE_URL,
        api_key=BEDROCK_API_KEY,
    )

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=1024,
        extra_headers={"anthropic-workspace": workspace_id},
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
        "x-api-key": BEDROCK_API_KEY,
        "anthropic-version": "2023-06-01",
        "anthropic-workspace": workspace_id,
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
        api_key=BEDROCK_API_KEY,
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
        extra_headers={"anthropic-workspace": workspace_id},
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
        extra_headers={"anthropic-workspace": workspace_id},
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
        extra_headers={"anthropic-workspace": workspace_id},
        system="You are a customer support agent for an e-commerce company. Be helpful, empathetic, and concise. If you need to look up information, explain what you're checking.",
        messages=messages,
    )

    assistant_reply = response.content[0].text
    print(f"  Turn 3 (Agent): {assistant_reply}\n")


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

    # Step 2: Get or create a workspace with tags for cost allocation
    print("--- Step 2: Get or Create a Workspace ---")
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
    workspace_id = workspace["id"]
    print(f"Workspace ID: {workspace_id}\n")

    # Step 3: Inference using the Anthropic SDK with workspace header
    print("--- Step 3: Inference with Workspace (Anthropic SDK) ---")
    result = invoke_with_workspace_sdk(
        workspace_id=workspace_id,
        user_message="A customer is asking about our return policy for electronics purchased more than 30 days ago. Summarize the key points I should mention.",
    )
    print(result)
    print()

    # Step 4: Inference using raw HTTP with workspace header
    print("--- Step 4: Inference with Workspace (HTTP/requests) ---")
    result = invoke_with_workspace_http(
        workspace_id=workspace_id,
        user_message="Draft a polite response to a customer whose shipment was delayed by 2 days due to weather.",
    )
    print(json.dumps(result, indent=2))
    print()

    # Step 5: Multi-turn customer support conversation
    print("--- Step 5: Multi-turn Customer Support Conversation ---")
    multi_turn_conversation(workspace_id)

    # Step 6: Multiple workspaces for different support tiers
    print("--- Step 6: Multiple Workspaces (Support Tiers) ---")
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

    client = anthropic.Anthropic(
        base_url=ANTHROPIC_BASE_URL,
        api_key=BEDROCK_API_KEY,
    )

    support_queries = [
        "How do I reset my password?",
        "My API integration is returning 429 errors after the latest update. I'm using the v3 SDK with retry logic.",
        "I've been waiting 2 weeks for a resolution on ticket #ESC-1192. This is my third follow-up. I need to speak to a manager.",
    ]

    for tier, query in zip(tiers, support_queries):
        ws = get_or_create_workspace(name=tier["name"], tags=tier["tags"])

        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=256,
            extra_headers={"anthropic-workspace": ws["id"]},
            system="You are a customer support agent. Respond appropriately for your support tier level.",
            messages=[
                {"role": "user", "content": query}
            ],
        )
        print(f"  [{tier['name']}]")
        print(f"  Query: {query}")
        print(f"  Response: {response.content[0].text}\n")


if __name__ == "__main__":
    main()
