"""
Medium-Effort Optimization - Lever 05: Service Tiers (Flex)

Amazon Bedrock offers four service tiers for model inference: Reserved,
Priority, Standard (the default), and Flex. The cost lever here is Flex:
for workloads that tolerate longer, variable processing times - model
evaluations, content summarization, labeling/annotation, multistep agentic
workflows - routing a request to the Flex tier earns a pricing discount versus
the Standard on-demand price. Adopting it is a workload-routing decision: the
code change is a single optional request parameter, but you must identify
latency-tolerant traffic, accept longer and more variable latency on it, and
validate the tradeoff before rollout.

You will learn how to:
- Pass service_tier through the Converse API via additionalModelRequestFields
- Run the same summarization task on Standard (default) and Flex, side by side
- Compare input/output tokens and latency across the two tiers

Key points:
- service_tier is an optional field that accepts: reserved, priority, default
  (Standard), flex. Omitting it is the same as "default" (Standard).
- Flex trades latency for a lower price: Flex requests get lower priority during
  high demand, so latency is longer and more variable. Priority is a price
  PREMIUM for the fastest response. Reserved needs a capacity reservation via
  your AWS account team.
- Your on-demand quota is SHARED across the priority, default, and flex tiers;
  the reserved tier capacity is separate.
- The resolved tier that actually served a request is NOT returned anywhere in
  the Converse response: not as a top-level field, not in
  additionalModelResponseFields, and not in any HTTP response header. It is
  observable ONLY via Amazon CloudWatch (the ResolvedServiceTier dimension,
  alongside ModelId and ServiceTier) and AWS CloudTrail events.
- The Flex DISCOUNT is a billing effect (seen on your bill and in Cost
  Explorer), not a per-request number returned by the API. The visible
  request-time tradeoff is typically higher or more variable latency, so a
  single run may or may not show a latency gap.

Model choice:
- This workshop otherwise standardizes on a constrained allowed-model list
  (Nova 2 Lite, Claude Haiku 4.5, Claude 5 Sonnet/Opus, OpenAI GPT-5.6 Sol).
  None of those support the Flex tier today. This sample uses
  openai.gpt-oss-120b precisely BECAUSE it is one of the models that serves
  Flex. The Flex/Priority launch set is OpenAI gpt-oss (20b/120b), DeepSeek
  V3.1, Qwen3 variants, and Amazon Nova Pro/Premier - not Anthropic Claude and
  not the lighter Nova/GPT-5.x models.
- Support is also region-gated. Check "Models at a glance" for the current,
  authoritative per-model and per-region supported-tier list:
  https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to openai.gpt-oss-120b on Amazon Bedrock in a region that supports the
  Flex tier
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

# openai.gpt-oss-120b is an OpenAI open-weight model on Bedrock. It is used here
# because it serves the Flex tier - the allowed workshop Claude/Nova/GPT-5.x
# models do not. It is also a reasoning model, so the response carries a
# reasoningContent block before the answer text block.
GPT_OSS = "openai.gpt-oss-120b-1:0"

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

def converse_with_tier(model_id: str, prompt: str, tier: str, max_tokens: int = 512) -> dict:
    """Call Converse requesting a service tier and return answer, usage, latency.

    service_tier is a model-specific request field, so it is passed through the
    Converse API via additionalModelRequestFields - the same mechanism
    01-5_adaptive_thinking.py uses for thinking and output_config. snake_case
    service_tier is correct; camelCase is rejected.

    gpt-oss-120b is a reasoning model: output.message.content holds a
    reasoningContent block FIRST, then the text answer block. So we iterate the
    content blocks and pick the block that has a text key - reading content[0]
    would KeyError on the reasoning block. maxTokens is 512 because the reasoning
    block consumes output tokens before the answer; too small a budget truncates
    before any answer text appears.

    The resolved tier is intentionally not returned: it is not present anywhere
    in the Converse response (no field, no header). Observe the actual tier that
    served the request via CloudWatch ResolvedServiceTier / CloudTrail.
    """
    t0 = time.time()
    resp = RUNTIME.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens},
        additionalModelRequestFields={"service_tier": tier},  # 'default' or 'flex'
    )

    # Separate the reasoning block from the final answer text.
    reasoning, answer = "", ""
    for block in resp["output"]["message"]["content"]:
        if "reasoningContent" in block:
            reasoning = (
                block["reasoningContent"].get("reasoningText", {}).get("text", "")
            )
        elif "text" in block:
            answer = block["text"]

    usage = resp["usage"]
    return {
        "text": answer,
        "reasoning": reasoning,
        "input_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],
        "latency_ms": int((time.time() - t0) * 1000),
        "requested_tier": tier,
    }


# ============================================================
# Demo
# ============================================================

def compare_tiers() -> None:
    """Run the same summarization task on Standard (default) and Flex."""
    print("--- Service Tiers: same task, Standard (default) vs Flex ---\n")

    rows = []

    # Standard (default) tier.
    std = converse_with_tier(GPT_OSS, SUMMARY_PROMPT, "default")
    rows.append(std)
    print("=== service_tier=default (Standard) ===")
    print(f"{std['text']}\n")

    # Flex tier - the expected, happy path on gpt-oss-120b. The try/except is
    # defensive hygiene only: model/region Flex availability can change, so if
    # the tier is ever rejected we point at the service-tiers doc instead of
    # crashing. Success is the primary path here.
    try:
        flex = converse_with_tier(GPT_OSS, SUMMARY_PROMPT, "flex")
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
    print(f"{'requested':<12} {'In':>6} {'Out':>6} {'ms':>7}")
    for r in rows:
        print(
            f"{r['requested_tier']:<12} "
            f"{r['input_tokens']:>6} {r['output_tokens']:>6} {r['latency_ms']:>7}"
        )

    print(
        "\n  The Flex DISCOUNT is a billing effect: it shows up on your bill and in"
        "\n  Cost Explorer, not as a dollar figure in the API response - so this"
        "\n  sample intentionally does not invent a discount number. The visible"
        "\n  request-time tradeoff is typically higher or more variable latency, so a"
        "\n  single run may or may not show a latency gap. The tier that actually"
        "\n  served each request is NOT in the API response; confirm it via"
        "\n  CloudWatch's ResolvedServiceTier dimension or CloudTrail events."
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
