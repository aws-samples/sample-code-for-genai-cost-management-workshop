"""
Medium-Effort Optimization - Lever 08: Bedrock Guardrails

Guardrails are sold as safety - block toxic prompts, refuse off-scope topics,
redact PII. They are equally a cost-and-latency lever: a request blocked at the
input layer never reaches the model, so you don't pay generation tokens on
injection probes, jailbreaks, or off-topic traffic. On any public endpoint,
abuse traffic is constant, and an off-topic "write me a poem" costs the same
tokens as a real question.

You will learn how to:
- Create a guardrail with a denied topic and PII policies
- Apply it inline on a Converse call (blocked input never pays for inference)
- Apply it standalone with ApplyGuardrail (screen text WITHOUT invoking a model)
- Clean up the guardrail afterwards

Pricing is per "text unit" (~1,000 characters), billed per policy evaluated - so
blocking off-topic traffic at the input layer saves the full request cost
(retrieval + generation), which more than funds the eval fee.

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock:CreateGuardrail, bedrock:DeleteGuardrail,
  bedrock-runtime:Converse, and bedrock-runtime:ApplyGuardrail
- Access to Claude 5 Sonnet on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import time
import boto3

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

BEDROCK = boto3.client("bedrock", region_name=REGION)
RUNTIME = boto3.client("bedrock-runtime", region_name=REGION)

SONNET = "global.anthropic.claude-sonnet-5"

SUPPORT_SYSTEM = (
    "You are TechMart's customer-support agent. Be concise and friendly. If a "
    "customer shares contact details, acknowledge them back and say you've noted "
    "them on their ticket. Answer product and warranty questions directly."
)


# ============================================================
# Create / delete an ephemeral guardrail
# ============================================================

def create_guardrail() -> str:
    """Create a minimal guardrail: deny competitor questions, handle PII."""
    guardrail = BEDROCK.create_guardrail(
        name=f"playbook-workshop-{int(time.time())}",
        description="Workshop demo - blocks competitor questions and redacts PII",
        topicPolicyConfig={
            "topicsConfig": [
                {
                    "name": "Competitor questions",
                    "definition": "Questions comparing us to or recommending named competitors",
                    "examples": [
                        "Why is the other store cheaper?",
                        "Should I shop somewhere else instead?",
                    ],
                    "type": "DENY",
                }
            ]
        },
        sensitiveInformationPolicyConfig={
            "piiEntitiesConfig": [
                {"type": "EMAIL", "action": "ANONYMIZE"},
                {"type": "PHONE", "action": "ANONYMIZE"},
                {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK"},
            ]
        },
        blockedInputMessaging="Sorry, I can't answer questions about competitors.",
        blockedOutputsMessaging="Output blocked by safety policy.",
    )
    return guardrail["guardrailId"]


def delete_guardrail(guardrail_id: str) -> None:
    BEDROCK.delete_guardrail(guardrailIdentifier=guardrail_id)


# ============================================================
# 1. Inline guardrail on a Converse call
# ============================================================

def demo_inline(guardrail_id: str) -> None:
    """Attach the guardrail to converse - input screened before the model runs."""
    print("--- 1. Inline guardrail on converse ---")
    print("  A blocked request never pays for inference.\n")

    def converse_with_guardrail(query: str) -> dict:
        return RUNTIME.converse(
            modelId=SONNET,
            system=[{"text": SUPPORT_SYSTEM}],
            messages=[{"role": "user", "content": [{"text": query}]}],
            inferenceConfig={"maxTokens": 300},  # Claude 5 Sonnet: no temperature
            guardrailConfig={
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": "DRAFT",
                "trace": "enabled",
            },
        )

    queries = [
        "How does your warranty compare to the store down the street?",  # blocked (competitor)
        "My email is alice@example.com - can you confirm receipt?",  # passes, email redacted
        "What's your warranty on a refurbished laptop?",  # passes - real answer
    ]
    for q in queries:
        r = converse_with_guardrail(q)
        stop = r.get("stopReason", "end_turn")
        text = r["output"]["message"]["content"][0].get("text", "<no text>")
        shown = text if stop == "guardrail_intervened" else text[:200] + "..."
        print(f"[{stop}] {q}")
        print(f"  -> {shown}\n")


# ============================================================
# 2. Standalone ApplyGuardrail - screen input with NO model call
# ============================================================

def demo_apply_guardrail(guardrail_id: str) -> None:
    """Screen input as its own call, before any retrieval/routing/tool work."""
    print("--- 2. Standalone ApplyGuardrail (no model invocation) ---")
    print("  Same screening, placed earlier in the pipeline; runs in front of any model.\n")

    cases = [
        ("competitor", "Should I just shop at the store down the street instead?"),  # denied
        ("legitimate", "What's your warranty on a refurbished laptop?"),  # passes
    ]
    for label, text in cases:
        r = RUNTIME.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion="DRAFT",
            source="INPUT",
            content=[{"text": {"text": text}}],
        )
        print(f"[{label:11}] action={r['action']:22} (no model called)")
        if r["action"] == "GUARDRAIL_INTERVENED":
            print("              -> refused for the price of one guardrail eval, not a full inference")
    print()


# ============================================================
# Main
# ============================================================

def main():
    print("Creating an ephemeral guardrail...\n")
    guardrail_id = create_guardrail()
    print(f"Created guardrail: {guardrail_id}\n")

    # Guardrails can take a moment to become ready for use.
    time.sleep(5)

    try:
        demo_inline(guardrail_id)
        demo_apply_guardrail(guardrail_id)
    finally:
        delete_guardrail(guardrail_id)
        print(f"Deleted guardrail: {guardrail_id}\n")

    print("--- Done ---")
    print("  Blocked traffic bills a fraction of a full inference. At a few percent")
    print("  probe/off-topic rate, those early refusals are real savings.")
    print()
    print("  Next steps:")
    print("  1. Define denied topics and PII policies for your own domain")
    print("  2. Place ApplyGuardrail before retrieval/routing to reject bad traffic early")
    print("  3. Publish a guardrail version and reference it across your models")


if __name__ == "__main__":
    main()
