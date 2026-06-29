"""
Amazon Bedrock Per-Request Metadata Tagging - Fine-Grained Cost Attribution

This sample demonstrates how to attach metadata to individual inference calls
for per-tenant, per-feature, and per-task cost attribution on bedrock-runtime.

You will learn how to:
- Add requestMetadata to Converse API calls
- Add requestMetadata to InvokeModel calls
- Tag requests with tenant, feature, and session identifiers
- Simulate a multi-tenant application with per-request cost tracking

Metadata appears in model invocation logs (CloudWatch Logs / S3) alongside
token usage. You can query these logs with Athena or visualize in QuickSight.

NOTE: This provides token counts, not dollar costs. Convert tokens to cost
using published pricing for each model.

Prerequisites:
- An AWS account with Amazon Bedrock access
- Model invocation logging MUST be enabled in Bedrock settings (Settings > Model invocation logging)
  Without logging enabled, metadata attached to requests will not be recorded.
- IAM credentials with bedrock-runtime:Converse and bedrock-runtime:InvokeModel
- Access to Claude and Nova models on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import json
import boto3

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

# Boto3 client for Bedrock Runtime
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)

# Models: Using Global cross-region inference profiles for maximum throughput
MODELS = {
    "nova-2-lite": "global.amazon.nova-2-lite-v1:0",
    "claude-sonnet-4-6": "global.anthropic.claude-sonnet-4-6",
    "claude-haiku-4-5": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
}


# ============================================================
# Converse API with Request Metadata
# ============================================================

def converse_with_metadata(
    model_id: str,
    user_message: str,
    metadata: dict,
    system_prompt: str = None,
) -> dict:
    """
    Make a Converse API call with requestMetadata for cost attribution.

    The metadata dict is recorded in model invocation logs, enabling
    per-tenant, per-feature, or per-task cost tracking.
    """
    messages = [{"role": "user", "content": [{"text": user_message}]}]

    params = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": {"maxTokens": 1024},
        "requestMetadata": metadata,
    }
    if system_prompt:
        params["system"] = [{"text": system_prompt}]

    response = bedrock_runtime.converse(**params)

    # Extract token usage for cost tracking
    usage = response.get("usage", {})

    return {
        "text": response["output"]["message"]["content"][0]["text"],
        "input_tokens": usage.get("inputTokens", 0),
        "output_tokens": usage.get("outputTokens", 0),
        "metadata": metadata,
    }


# ============================================================
# InvokeModel with Request Metadata
# ============================================================

def invoke_model_with_metadata(client, model_id, body, metadata_tags):
    """
    Invokes a model using the InvokeModel API with per-request metadata tags.

    Args:
        client: The Boto3 Bedrock runtime client.
        model_id (str): The model ID to use.
        body (dict): The request body for the model.
        metadata_tags (dict): Key-value pairs for request metadata (max 16 entries).

    Returns:
        response (dict): The parsed response body, or None on error.
    """
    try:
        response = client.invoke_model(
            body=json.dumps(body),
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
            requestMetadata=json.dumps(metadata_tags),
        )
        return json.loads(response["body"].read())
    except Exception as e:
        print(f"  Error invoking {model_id}: {e}")
        return None


# ============================================================
# Demo Steps
# ============================================================

def demo_per_tenant_converse():
    """Step 1: Converse API with per-tenant metadata across different models."""
    print("--- Step 1: Converse API with Per-Tenant Metadata ---")
    print("  Each call is tagged with tenant, feature, and session info.\n")

    tenants = [
        {
            "model": MODELS["claude-haiku-4-5"],
            "metadata": {
                "tenant_id": "tenant-acme-corp",
                "feature": "document-summarization",
                "session_id": "sess-a1b2c3d4",
                "environment": "production",
            },
            "message": "Summarize the key points of a quarterly earnings report in 3 bullet points.",
        },
        {
            "model": MODELS["claude-sonnet-4-6"],
            "metadata": {
                "tenant_id": "tenant-globex-inc",
                "feature": "code-review",
                "session_id": "sess-e5f6g7h8",
                "environment": "production",
            },
            "message": "Review this code for security issues: `cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')`",
        },
        {
            "model": MODELS["nova-2-lite"],
            "metadata": {
                "tenant_id": "tenant-initech-llc",
                "feature": "email-drafting",
                "session_id": "sess-i9j0k1l2",
                "environment": "production",
            },
            "message": "Draft a professional follow-up email after a product demo meeting.",
        },
    ]

    for tenant in tenants:
        result = converse_with_metadata(
            model_id=tenant["model"],
            user_message=tenant["message"],
            metadata=tenant["metadata"],
            system_prompt="Be concise and professional.",
        )
        print(f"  Tenant: {tenant['metadata']['tenant_id']}")
        print(f"  Feature: {tenant['metadata']['feature']}")
        print(f"  Model: {tenant['model']}")
        print(f"  Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
        print(f"  Response: {result['text'][:150]}...\n")


def demo_agent_steps_converse():
    """Step 2: Converse API with per-agent-step metadata to track agentic workflows."""
    print("--- Step 2: Converse API with Per-Agent-Step Metadata ---")
    print("  Track costs for each step in an agentic workflow.\n")

    agent_steps = [
        {
            "step": "plan",
            "metadata": {
                "agent_id": "travel-planner-agent",
                "task_id": "task-trip-paris-2026",
                "step": "planning",
                "step_number": "1",
                "customer_id": "cust-12345",
            },
            "message": "Plan a 5-day trip to Paris for a family of 4 in September. Budget: $5000. List the main activities.",
        },
        {
            "step": "research",
            "metadata": {
                "agent_id": "travel-planner-agent",
                "task_id": "task-trip-paris-2026",
                "step": "research",
                "step_number": "2",
                "customer_id": "cust-12345",
            },
            "message": "What are the best family-friendly hotels near the Eiffel Tower under $200/night?",
        },
        {
            "step": "synthesize",
            "metadata": {
                "agent_id": "travel-planner-agent",
                "task_id": "task-trip-paris-2026",
                "step": "synthesis",
                "step_number": "3",
                "customer_id": "cust-12345",
            },
            "message": "Create a day-by-day itinerary for the Paris trip with hotel, activities, and estimated costs per day.",
        },
    ]

    total_input_tokens = 0
    total_output_tokens = 0

    for step in agent_steps:
        result = converse_with_metadata(
            model_id=MODELS["claude-sonnet-4-6"],
            user_message=step["message"],
            metadata=step["metadata"],
            system_prompt="You are a travel planning agent. Be helpful and structured.",
        )
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        print(f"  Step {step['metadata']['step_number']} ({step['step']}): "
              f"{result['input_tokens']} in / {result['output_tokens']} out tokens")

    print(f"\n  Total for task: {total_input_tokens} input + {total_output_tokens} output tokens")
    print(f"  All steps tagged with task_id: {agent_steps[0]['metadata']['task_id']}\n")


def demo_invoke_model_metadata():
    """Step 3: InvokeModel API with requestMetadata parameter."""
    print("--- Step 3: InvokeModel with Per-Request Metadata ---")
    print("  Using requestMetadata parameter in InvokeModel API.\n")

    # Example with Amazon Nova 2 Lite (messages-v1 schema)
    nova_metadata = {
        "team": "data-science",
        "application": "customer-support-bot",
        "environment": "development",
        "experiment_id": "exp-2026-001",
    }

    nova_body = {
        "schemaVersion": "messages-v1",
        "messages": [
            {"role": "user", "content": [{"text": "Explain request metadata in APIs in two sentences."}]}
        ],
        "inferenceConfig": {"max_new_tokens": 100, "temperature": 0},
    }

    print(f"  Model: {MODELS['nova-2-lite']}")
    print(f"  Metadata: {nova_metadata}")
    response_body = invoke_model_with_metadata(
        bedrock_runtime, MODELS["nova-2-lite"], nova_body, nova_metadata
    )
    if response_body:
        print(f"  Response: {response_body['output']['message']['content'][0]['text']}\n")

    # Example with Claude Haiku 4.5 (Anthropic Messages format)
    claude_metadata = {
        "team": "platform-engineering",
        "application": "code-assistant",
        "environment": "production",
        "user_id": "dev-alice",
    }

    claude_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "What are the SOLID principles? List them briefly."}
        ],
    }

    print(f"  Model: {MODELS['claude-haiku-4-5']}")
    print(f"  Metadata: {claude_metadata}")
    response_body = invoke_model_with_metadata(
        bedrock_runtime, MODELS["claude-haiku-4-5"], claude_body, claude_metadata
    )
    if response_body:
        print(f"  Response: {response_body['content'][0]['text'][:200]}...\n")


def demo_multi_model_comparison():
    """Step 4: Compare token usage across models for the same prompt."""
    print("--- Step 4: Multi-Model Comparison (Same Request, Different Models) ---")
    print("  Compare token usage across models for the same prompt.\n")

    comparison_metadata = {
        "tenant_id": "tenant-globex-inc",
        "feature": "model-evaluation",
        "experiment_id": "exp-model-compare-001",
        "environment": "staging",
    }

    comparison_prompt = "Explain the CAP theorem in distributed systems in 2-3 sentences."

    for model_name, model_id in MODELS.items():
        result = converse_with_metadata(
            model_id=model_id,
            user_message=comparison_prompt,
            metadata={**comparison_metadata, "model": model_name},
        )
        print(f"  {model_name}:")
        print(f"    Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
        print(f"    Response: {result['text'][:120]}...\n")


# ============================================================
# Main
# ============================================================

def main():
    demo_per_tenant_converse()
    demo_agent_steps_converse()
    demo_invoke_model_metadata()
    demo_multi_model_comparison()

    print("--- Done ---")
    print("  Metadata tags were attached to all requests and will appear in model invocation logs.")
    print()
    print("  Next steps:")
    print("  1. Ensure model invocation logging is enabled in Amazon Bedrock settings")
    print("  2. Check CloudWatch Logs or S3 for the metadata in invocation logs")
    print("  3. Query logs with Athena to aggregate token usage by tenant/feature/task")
    print("  4. Build QuickSight dashboards for per-tenant cost visualization")


if __name__ == "__main__":
    main()
