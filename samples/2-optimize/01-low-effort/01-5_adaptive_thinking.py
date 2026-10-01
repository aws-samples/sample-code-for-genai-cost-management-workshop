"""
Low-Effort Optimization - Lever 06: Adaptive Thinking

"Thinking" lets a Claude model spend extra internal tokens reasoning before it
answers - better on hard reasoning, coding, and planning, at the cost of more
output tokens and latency. Adaptive thinking replaces the old fixed
budget_tokens control: you set an effort level and the model decides, per
request, how much to think. It is both a quality lever (think when it helps)
and a cost/latency lever (don't pay when it doesn't).

You will learn how to:
- Enable adaptive thinking on the Converse API via additionalModelRequestFields
- Sweep effort levels (low / medium / high) and watch output tokens and latency
- Read the reasoning trace from the response's reasoning content block

Key points:
- One field, same Converse API - add thinking and output_config to
  additionalModelRequestFields. No new endpoint, no SDK swap.
- effort levels: low, medium, high (default), max. The practical pattern is to
  step DOWN from high once evals show a lower level holds quality.
- Placement matters: effort goes in a SEPARATE output_config object, NOT inside
  thinking. Putting it inside returns a ValidationException.
- Reasoning tokens are billed as OUTPUT tokens at the standard output rate.
- Model support: Claude 5 Opus / Sonnet (thinking capable). Not Haiku 4.5.

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
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

RUNTIME = boto3.client("bedrock-runtime", region_name=REGION)

# Adaptive thinking requires a thinking-capable model. Claude 5 Sonnet supports
# adaptive thinking.
SONNET = "global.anthropic.claude-sonnet-5"

# An ambiguous support ticket - the kind of task where more thinking can help.
TICKET = (
    "A customer bought a laptop a couple of months ago and now the battery "
    "drains in under an hour. They're out of the 30-day return window but within "
    "the 1-year manufacturer warranty. They're frustrated and asking for a full "
    "refund. What should we do, and how should we phrase the reply? Reason it "
    "through, then give the recommended action and a short customer message."
)


# ============================================================
# Adaptive thinking with an effort level
# ============================================================

def converse_with_effort(model_id: str, prompt: str, effort: str, max_tokens: int = 2048) -> dict:
    """Call Converse with adaptive thinking at the given effort level.

    effort goes in a SEPARATE output_config object inside
    additionalModelRequestFields - not inside the thinking object.
    """
    t0 = time.time()
    resp = RUNTIME.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens},
        additionalModelRequestFields={
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        },
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
        "reasoning": reasoning,
        "answer": answer,
        "input_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],  # includes reasoning tokens
        "latency_ms": int((time.time() - t0) * 1000),
    }


# ============================================================
# Demo
# ============================================================

def demo_effort_sweep() -> None:
    """Sweep low -> medium -> high on one ambiguous ticket."""
    print("--- Adaptive Thinking: sweep effort on one ambiguous ticket ---")
    print(f"TICKET:\n  {TICKET}\n")

    rows = []
    for effort in ["low", "medium", "high"]:
        r = converse_with_effort(SONNET, TICKET, effort)
        rows.append((effort, r["output_tokens"], r["latency_ms"]))

        print("=" * 70)
        print(f"effort={effort}  output_tokens={r['output_tokens']}  ms={r['latency_ms']}")
        print("=" * 70)
        if r["reasoning"]:
            trace = r["reasoning"].strip().replace("\n", " ")
            print(f"  reasoning trace ({len(r['reasoning'])} chars): {trace[:300]}...")
        print(f"  answer: {r['answer'][:400]}\n")

    print("Summary (output tokens include reasoning, billed at the output rate):")
    print(f"{'effort':<10} {'out_tokens':>11} {'ms':>8}")
    for effort, out_tokens, ms in rows:
        print(f"{effort:<10} {out_tokens:>11} {ms:>8}")
    print(
        "\n  Higher effort spends more reasoning (output) tokens and latency. Start at"
        "\n  high, then step down once an eval shows a lower level holds quality for"
        "\n  your task - that is where the cost and latency savings come from.\n"
    )


# ============================================================
# Main
# ============================================================

def main():
    demo_effort_sweep()

    print("--- Done ---")
    print("  Next steps:")
    print("  1. Enable adaptive thinking on reasoning-heavy tasks (thinking-capable models)")
    print("  2. Benchmark effort levels on your eval set")
    print("  3. Step down from the default 'high' to the lowest level that holds quality")


if __name__ == "__main__":
    main()
