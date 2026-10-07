"""
Low-Effort Optimization - Lever 07: Service Tiers (Flex)

Amazon Bedrock offers four service tiers for model inference: Reserved,
Priority, Standard (the default), and Flex. The low-effort cost lever is Flex:
for workloads that tolerate longer, variable processing times - model
evaluations, content summarization, batch and agentic workflows - routing a
request to the Flex tier earns a pricing discount versus the Standard on-demand
price. It is a near-zero-code-change lever: one optional request parameter.

You will learn how to:
- Pass service_tier through the Converse API via additionalModelRequestFields
- Run the same summarization task on Standard (default) and Flex, side by side
- Read the resolved service tier back from the response when it is present

Key points:
- service_tier is an optional field that accepts: reserved, priority, default
  (Standard), flex. Omitting it is the same as "default" (Standard).
- Flex trades latency for a lower price. Priority is a price PREMIUM for the
  fastest response. Reserved needs a capacity reservation via your AWS account
  team.
- Your on-demand quota is SHARED across the priority, default, and flex tiers;
  the reserved tier capacity is separate.
- The tier that actually served a request is observable: it appears in the API
  response and in AWS CloudTrail events, and in Amazon CloudWatch metrics under
  the ResolvedServiceTier dimension (alongside ModelId and ServiceTier).
  ResolvedServiceTier is the ACTUAL tier that served the request, which can
  differ from the one you requested.
- The Flex DISCOUNT is a billing effect (seen on your bill and in Cost
  Explorer), not a per-request number returned by the API. The visible
  request-time tradeoff is typically higher or more variable latency, so a
  single run may or may not show a latency gap.
- Model and region support varies. Check "Models at a glance" for each model's
  supported tiers:
  https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude Haiku 4.5 on Amazon Bedrock in a region/model that supports
  the Flex tier
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import time
import boto3
from botocore.exceptions import ClientError

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

RUNTIME = boto3.client("bedrock-runtime", region_name=REGION)

# Allowed workshop model (Global cross-region inference profile). A cheap model
# fits the batch/summarization framing that Flex is built for.
HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# A latency-tolerant task: summarize a chunk of content. Content summarization
# is an explicit Flex use case per the Bedrock docs.
ARTICLE = (
    "Quarterly operations review. In the first quarter, the fulfillment team "
    "cut average order-to-ship time from 48 hours to 31 hours by moving the two "
    "busiest regional warehouses to a wave-picking model and adding a second "
    "evening shift. Returns rose slightly, from 4.1% to 4.6%, driven almost "
    "entirely by a single apparel vendor whose sizing ran small; that vendor has "
    "since updated its size charts. Customer support ticket volume fell 12% "
    "after the self-service returns portal launched in February, though average "
    "handle time for the remaining tickets went up because the easy cases now "
    "resolve themselves. Shipping cost per order was flat despite higher fuel "
    "surcharges, because the faster ship times let more orders consolidate into "
    "ground shipments instead of air. The team flagged two risks for next "
    "quarter: the evening shift relies on temporary staffing that is hard to "
    "retain, and the wave-picking model has not yet been tested against a peak "
    "holiday volume."
)

SUMMARY_PROMPT = (
    "Summarize the following operations review in three concise bullet points, "
    "then state the single biggest risk for next quarter:\n\n" + ARTICLE
)


# ============================================================
# Helper Functions
# ============================================================

def extract_resolved_tier(resp: dict) -> str:
    """Best-effort read of the resolved service tier from a Converse response.

    The docs state the served tier is visible in the API response, but do not
    pin down the exact field name, and it can vary by model/region. Look in the
    likely locations and fall back to a clear "not reported" marker rather than
    hard-failing when the field is absent.
    """
    # Top-level field (some responses surface it directly).
    for key in ("serviceTier", "service_tier", "resolvedServiceTier", "ResolvedServiceTier"):
        if key in resp:
            return str(resp[key])

    # Model-specific response fields (mirror of additionalModelRequestFields).
    extra = resp.get("additionalModelResponseFields") or {}
    if isinstance(extra, dict):
        for key in ("service_tier", "serviceTier", "resolved_service_tier", "resolvedServiceTier"):
            if key in extra:
                return str(extra[key])

    # Response metadata headers can carry it on some paths.
    meta = resp.get("ResponseMetadata", {}) or {}
    headers = meta.get("HTTPHeaders", {}) or {}
    for key in ("x-amzn-bedrock-service-tier", "x-amzn-bedrock-resolved-service-tier"):
        if key in headers:
            return str(headers[key])

    return "not reported in response (observe via CloudWatch ResolvedServiceTier / CloudTrail)"


def converse_with_tier(model_id: str, prompt: str, tier: str, max_tokens: int = 400) -> dict:
    """Call Converse requesting a service tier and return usage, latency, tier.

    service_tier is a model-specific request field, so it is passed through the
    Converse API via additionalModelRequestFields - the same mechanism
    01-5_adaptive_thinking.py uses for thinking and output_config.
    """
    t0 = time.time()
    resp = RUNTIME.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens},
        additionalModelRequestFields={"service_tier": tier},  # 'default' or 'flex'
    )
    usage = resp["usage"]
    return {
        "text": resp["output"]["message"]["content"][0]["text"],
        "input_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],
        "latency_ms": int((time.time() - t0) * 1000),
        "requested_tier": tier,
        "resolved_tier": extract_resolved_tier(resp),
    }


# ============================================================
# Demo
# ============================================================

def compare_tiers() -> None:
    """Run the same summarization task on Standard (default) and Flex."""
    print("--- Service Tiers: same task, Standard (default) vs Flex ---\n")

    rows = []

    # Standard (default) tier - always runs so the sample produces useful output.
    std = converse_with_tier(HAIKU, SUMMARY_PROMPT, "default")
    rows.append(std)
    print("=== service_tier=default (Standard) ===")
    print(f"{std['text']}\n")

    # Flex tier - may be unsupported for a given model/region, so guard the call.
    # If the tier is rejected, explain it clearly instead of crashing.
    flex = None
    try:
        flex = converse_with_tier(HAIKU, SUMMARY_PROMPT, "flex")
        rows.append(flex)
        print("=== service_tier=flex ===")
        print(f"{flex['text']}\n")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        print("=== service_tier=flex ===")
        print(
            "  The Flex request was rejected (" + (code or "ClientError") + ")."
            "\n  Flex tier availability varies by model and region. Confirm this"
            "\n  model/region supports Flex via 'Models at a glance':"
            "\n  https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html\n"
        )

    print("Summary:")
    print(f"{'requested':<12} {'resolved':<16} {'In':>6} {'Out':>6} {'ms':>7}")
    for r in rows:
        resolved = r["resolved_tier"]
        print(
            f"{r['requested_tier']:<12} {resolved[:16]:<16} "
            f"{r['input_tokens']:>6} {r['output_tokens']:>6} {r['latency_ms']:>7}"
        )

    print(
        "\n  The Flex DISCOUNT is a billing effect: it shows up on your bill and in"
        "\n  Cost Explorer, not as a dollar figure in the API response - so this"
        "\n  sample intentionally does not invent a discount number. The visible"
        "\n  request-time tradeoff is typically higher or more variable latency, so a"
        "\n  single run may or may not show a latency gap. Confirm the actual tier"
        "\n  that served each request via CloudWatch's ResolvedServiceTier dimension."
    )


# ============================================================
# Main
# ============================================================

def main():
    compare_tiers()

    print("\n--- Done ---")
    print("  Next steps:")
    print("  1. Identify latency-tolerant workloads (evals, summarization, agentic/batch)")
    print("  2. Route them to Flex by setting service_tier='flex' - a one-parameter change")
    print("  3. Verify the discount in Cost Explorer and monitor ResolvedServiceTier in CloudWatch")


if __name__ == "__main__":
    main()
