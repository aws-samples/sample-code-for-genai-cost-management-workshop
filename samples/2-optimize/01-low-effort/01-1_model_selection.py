"""
Low-Effort Optimization - Lever 01: Model Selection

The highest-leverage cost decision is *which* model runs each call. The price
gap between Claude tiers is roughly 5x per tier, so running a frontier model on
a task a smaller one handles is pure waste - and usually slower too.

You will learn how to:
- Run the same task on a small model (Haiku) and a mid-tier model (Sonnet)
- Compare input/output tokens, latency, and per-call cost side by side
- See why you confirm the cheaper choice on an eval set, not on one output

The rule:
- Start small, escalate on evidence. Try Haiku; move up only when a scored eval
  shows it fails on your task.
- Re-benchmark every model generation. Today's small model is close to last
  year's large one, so "needs the big model" decisions go stale.
- Never decide by vibes. One good (or bad) output is not evidence; a held-out
  eval set is.

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude models on Amazon Bedrock
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

# Allowed workshop models (Global cross-region inference profiles)
HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET = "global.anthropic.claude-sonnet-5"

# Pricing per 1M tokens (USD). Illustrative workshop values - always verify
# current pricing at https://aws.amazon.com/bedrock/pricing/
PRICING = {
    HAIKU: {"input": 1.00, "output": 5.00},
    SONNET: {"input": 3.00, "output": 15.00},
}

# A neutral example task, reused so the same-input comparison is fair.
EXAMPLE_TASK = (
    "Summarize this product review in one sentence, then rate its sentiment "
    "as positive, neutral, or negative:\n\n"
    "'Delivery was fast and the packaging was solid, but the device stopped "
    "working after two weeks and support was slow to respond.'"
)


# ============================================================
# Helper Functions
# ============================================================

def calculate_cost(input_tokens: int, output_tokens: int, model_id: str) -> float:
    """Return the dollar cost of a single request for the given model."""
    prices = PRICING[model_id]
    return (input_tokens / 1_000_000) * prices["input"] + (
        output_tokens / 1_000_000
    ) * prices["output"]


def build_inference_config(model_id: str, max_tokens: int) -> dict:
    """Build inferenceConfig, omitting temperature for models that reject it.

    Claude 5 models (Sonnet 5, Opus 5) deprecated the temperature parameter and
    return a ValidationException if it is supplied. Haiku 4.5 still accepts it.
    """
    cfg = {"maxTokens": max_tokens}
    if "claude-sonnet-5" not in model_id and "claude-opus-5" not in model_id:
        cfg["temperature"] = 0
    return cfg


def converse_simple(model_id: str, query: str, max_tokens: int = 400) -> dict:
    """Make a Converse API call and return text, token usage, and latency."""
    t0 = time.time()
    resp = RUNTIME.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": query}]}],
        inferenceConfig=build_inference_config(model_id, max_tokens),
    )
    usage = resp["usage"]
    return {
        "text": resp["output"]["message"]["content"][0]["text"],
        "input_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],
        "latency_ms": int((time.time() - t0) * 1000),
        "model": model_id,
    }


# ============================================================
# Demo
# ============================================================

def compare_models() -> None:
    """Run the same task on Haiku and Sonnet and compare cost/latency/tokens."""
    print("--- Model Selection: same task, two model tiers ---\n")

    rows = []
    for model_id in [HAIKU, SONNET]:
        r = converse_simple(model_id, EXAMPLE_TASK)
        cost = calculate_cost(r["input_tokens"], r["output_tokens"], r["model"])
        rows.append(
            (
                model_id.split(".")[-1][:24],
                r["input_tokens"],
                r["output_tokens"],
                r["latency_ms"],
                cost,
            )
        )
        print(f"=== {model_id} ===")
        print(f"{r['text']}\n")

    print("\nSummary:")
    print(f"{'Model':<26} {'In':>6} {'Out':>6} {'ms':>7} {'$':>11}")
    for name, in_tok, out_tok, ms, cost in rows:
        print(f"{name:<26} {in_tok:>6} {out_tok:>6} {ms:>7} {cost:>11.6f}")

    print(
        "\n  For a well-defined single-step task, the smaller model often produces"
        "\n  an answer indistinguishable from the larger one at a fraction of the"
        "\n  cost and latency. Confirm the cheaper choice on an eval set, not on one"
        "\n  output - quality gaps show up on ambiguous prompts."
    )


# ============================================================
# Main
# ============================================================

def main():
    compare_models()

    print("\n--- Done ---")
    print("  Next steps:")
    print("  1. Build a small held-out eval set for your own task")
    print("  2. Start on the cheapest tier (Haiku) and score it")
    print("  3. Escalate a tier only when the eval shows the cheaper model falls short")


if __name__ == "__main__":
    main()
