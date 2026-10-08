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
- Pass service_tier on the OpenAI Chat Completions API (bedrock-runtime endpoint)
- Run the same summarization task on Standard (default) and Flex, side by side
- Read the tier that served each request back from the response
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
- This sample calls the OpenAI Chat Completions API rather than Converse
  because Chat Completions returns a top-level service_tier field in the
  response, so you can see which tier each request used. The Converse API
  returns no tier field at all (no top-level field, no
  additionalModelResponseFields entry, no HTTP header).
- For fleet-wide confirmation use Amazon CloudWatch (the ResolvedServiceTier
  dimension, alongside ModelId and ServiceTier) and AWS CloudTrail events.
- The Flex DISCOUNT is a billing effect (seen on your bill and in Cost
  Explorer), not a per-request number returned by the API. The visible
  request-time tradeoff is typically higher or more variable latency, so a
  single run may or may not show a latency gap.

Model choice:
- This workshop otherwise standardizes on a constrained allowed-model list
  (Nova 2 Lite, Claude Haiku 4.5, Claude 5 Sonnet).
  None of those support the Flex tier today. This sample uses
  openai.gpt-oss-120b precisely BECAUSE it is one of the models that serves
  Flex. The Flex/Priority launch set is OpenAI gpt-oss (20b/120b), DeepSeek
  V3.1, Qwen3 variants, and Amazon Nova Pro/Premier - not Anthropic Claude and
  not the lighter Nova models.
- The newer OpenAI frontier models (GPT-6.1 Sol, GPT-6 Astra/Sol/Luna, and the
  GPT-5.6 and GPT-5.5 families) do not serve Flex either: tested on Converse,
  Responses, and Chat Completions, they accept only service_tier "default" and
  reject "flex" and "priority" with a validation error.
- Support is also region-gated. Check "Models at a glance" for the current,
  authoritative per-model and per-region supported-tier list:
  https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials able to call Amazon Bedrock (used to mint a short-lived
  Bedrock API token for the OpenAI SDK)
- Access to openai.gpt-oss-120b on Amazon Bedrock in a region that supports the
  Flex tier
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import re
import time
from openai import APIStatusError, OpenAI
from aws_bedrock_token_generator import provide_token

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

# OpenAI-compatible endpoint on bedrock-runtime. The OpenAI SDK authenticates
# with a short-lived Bedrock API token minted from your IAM credentials.
OPENAI_BASE_URL = f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1"

# openai.gpt-oss-120b is an OpenAI open-weight model on Bedrock. It is used here
# because it serves the Flex tier - the allowed workshop Claude/Nova
# models do not. It is also a reasoning model: on Chat Completions the
# reasoning arrives inline at the start of the message content, wrapped in
# <reasoning>...</reasoning> tags, followed by the answer.
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

def split_reasoning(content: str) -> tuple[str, str]:
    """Split Chat Completions content into (reasoning, answer).

    gpt-oss is a reasoning model. On Chat Completions the reasoning comes first
    inside <reasoning>...</reasoning> tags and the answer follows. If the token
    budget runs out while still reasoning, the closing tag may be missing and
    there is no answer yet - return an empty answer in that case.
    """
    match = re.match(r"\s*<reasoning>(.*?)</reasoning>(.*)", content, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    if content.lstrip().startswith("<reasoning>"):
        return content.strip(), ""
    return "", content.strip()


def chat_with_tier(
    client: OpenAI, model_id: str, prompt: str, tier: str, max_tokens: int = 512
) -> dict:
    """Call Chat Completions requesting a service tier; return answer, usage, tier.

    service_tier is a first-class parameter on the OpenAI Chat Completions API
    ('default' or 'flex'). max_tokens is 512 because the reasoning consumes
    output tokens before the answer; too small a budget truncates before any
    answer text appears.

    The response carries a top-level service_tier field naming the tier that
    served the request. (The Converse API returns no such field, which is why
    this sample uses Chat Completions.)
    """
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens,
        service_tier=tier,
    )
    reasoning, answer = split_reasoning(resp.choices[0].message.content or "")
    return {
        "text": answer,
        "reasoning": reasoning,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "latency_ms": int((time.time() - t0) * 1000),
        "requested_tier": tier,
        "served_tier": getattr(resp, "service_tier", None) or "not reported",
    }


# ============================================================
# Demo
# ============================================================

def compare_tiers() -> None:
    """Run the same summarization task on Standard (default) and Flex."""
    print("--- Service Tiers: same task, Standard (default) vs Flex ---\n")

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=provide_token(region=REGION))
    rows = []

    # Standard (default) tier.
    std = chat_with_tier(client, GPT_OSS, SUMMARY_PROMPT, "default")
    rows.append(std)
    print("=== service_tier=default (Standard) ===")
    print(f"{std['text']}\n")
    print(f"  Service tier returned in response: {std['served_tier']}\n")

    # Flex tier - the expected, happy path on gpt-oss-120b. The try/except is
    # defensive hygiene only: model/region Flex availability can change, so if
    # the tier is ever rejected we point at the service-tiers doc instead of
    # crashing. Success is the primary path here.
    try:
        flex = chat_with_tier(client, GPT_OSS, SUMMARY_PROMPT, "flex")
        rows.append(flex)
        print("=== service_tier=flex ===")
        print(f"{flex['text']}\n")
        print(f"  Service tier returned in response: {flex['served_tier']}\n")
    except APIStatusError as e:
        print("=== service_tier=flex ===")
        print(
            f"  The Flex request was rejected (HTTP {e.status_code})."
            "\n  Flex tier availability varies by model and region. Confirm this"
            "\n  model/region supports Flex via 'Models at a glance':"
            "\n  https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html\n"
        )

    print("Summary:")
    print(f"{'requested':<11} {'served':<9} {'In':>6} {'Out':>6} {'ms':>7}")
    for r in rows:
        print(
            f"{r['requested_tier']:<11} {r['served_tier']:<9} "
            f"{r['input_tokens']:>6} {r['output_tokens']:>6} {r['latency_ms']:>7}"
        )

    print(
        "\n  'served' is the service_tier field returned in the Chat Completions"
        "\n  response - the tier that handled each request."
        "\n  The Flex DISCOUNT is a billing effect: it shows up on your bill and in"
        "\n  Cost Explorer, not as a dollar figure in the API response - so this"
        "\n  sample intentionally does not invent a discount number. The visible"
        "\n  request-time tradeoff is typically higher or more variable latency, so a"
        "\n  single run may or may not show a latency gap. For fleet-wide"
        "\n  confirmation use CloudWatch's ResolvedServiceTier dimension or"
        "\n  CloudTrail events."
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
    print("  3. Check the served tier in each response and monitor ResolvedServiceTier in CloudWatch")
    print("  4. Verify the discount in Cost Explorer")


if __name__ == "__main__":
    main()
