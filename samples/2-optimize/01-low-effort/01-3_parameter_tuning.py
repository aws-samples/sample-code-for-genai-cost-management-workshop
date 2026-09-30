"""
Low-Effort Optimization - Lever 03: Parameter Tuning

Four inference-config values that ship in minutes and cost nothing to change,
but the defaults favor "won't surprise you," not "fast, cheap, predictable."
Getting them right buys 10-30% on latency and concurrency before you touch a
prompt.

You will learn how to:
- See how max_tokens sets the TPM quota RESERVED up front (affects concurrency)
- Use stop_sequences to halt generation the instant a terminator appears
- See how temperature controls determinism vs variation

Key points:
- max_tokens is the load-bearing one. On Bedrock it reserves quota up front
  (output burns at 5x for Claude 3.7+), refunded after. A too-large value
  throttles concurrency under load. Per-call cost is unchanged (you pay for
  tokens generated); concurrency is not. Right-size it.
- temperature: 0 for classification/extraction/structured output; 0.3-0.7 for
  generation; higher for brainstorming.
- stop_sequences: a free latency win when a real terminator exists.

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

HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# Output-token burndown rate for TPM quota (5x for Claude 3.7 and newer).
OUTPUT_BURNDOWN_RATE = 5

EXAMPLE_TASK = (
    "Summarize this product review in one sentence, then rate its sentiment "
    "as positive, neutral, or negative:\n\n"
    "'Delivery was fast and the packaging was solid, but the device stopped "
    "working after two weeks and support was slow to respond.'"
)

EMAIL = (
    "Hi, the wireless headphones I ordered last week arrived with a cracked case "
    "and the left earbud won't charge. I'd like a replacement before the weekend."
)


# ============================================================
# 1. max_tokens - TPM reservation
# ============================================================

def demo_max_tokens() -> None:
    """Same output, very different TPM reserved: max_tokens=4096 vs 200."""
    print("--- 1. max_tokens: identical output, different reserved TPM ---")
    print(f"INPUT:\n  {EXAMPLE_TASK}\n")

    rows = []
    answer = None
    for mt in [4096, 200]:
        resp = RUNTIME.converse(
            modelId=HAIKU,
            messages=[{"role": "user", "content": [{"text": EXAMPLE_TASK}]}],
            inferenceConfig={"maxTokens": mt, "temperature": 0},
        )
        answer = resp["output"]["message"]["content"][0]["text"]
        out_tokens = resp["usage"]["outputTokens"]
        reserved_tpm = resp["usage"]["inputTokens"] + (mt * OUTPUT_BURNDOWN_RATE)
        rows.append((mt, out_tokens, reserved_tpm))

    print("OUTPUT (identical regardless of max_tokens):")
    print(f"  {answer}\n")

    print(f"{'max_tokens':>10}  {'out':>4}  {'reserved TPM':>13}")
    for mt, out, tpm in rows:
        print(f"{mt:>10}  {out:>4}  {tpm:>13,}")
    print(
        "\n  Same output, but max_tokens=4096 reserves ~20x the TPM of max_tokens=200"
        "\n  - that is the concurrency you silently give up under load.\n"
    )


# ============================================================
# 2. stop_sequences - early termination
# ============================================================

def demo_stop_sequences() -> None:
    """Generation halts the instant a known terminator appears - fewer tokens, faster."""
    print("--- 2. stop_sequences: halt on a known boundary ---\n")

    prompt = (
        "Categorize the support email into one word (shipping, product, billing, other). "
        "Reply with the category, then ###STOP\n\n"
        f"Email: {EMAIL}\n\n"
        "Category: "
    )

    for use_stop in [False, True]:
        cfg = {"maxTokens": 200, "temperature": 0}
        if use_stop:
            cfg["stopSequences"] = ["###STOP"]
        t0 = time.time()
        resp = RUNTIME.converse(
            modelId=HAIKU,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig=cfg,
        )
        text = resp["output"]["message"]["content"][0]["text"]
        print(f"stop_sequences={use_stop}")
        print(f"  output: {text!r}")
        print(
            f"  out_tokens={resp['usage']['outputTokens']}  ms={int((time.time() - t0) * 1000)}\n"
        )


# ============================================================
# 3. temperature - determinism vs variation
# ============================================================

def demo_temperature() -> None:
    """Same creative prompt 3x per setting to show consistency vs variation."""
    print("--- 3. temperature: determinism vs variation ---\n")

    tagline = "Write a one-sentence tagline for a coffee shop."

    for temp in [0.0, 0.7, 1.0]:
        print(f"temperature={temp}")
        for run in range(3):
            resp = RUNTIME.converse(
                modelId=HAIKU,
                messages=[{"role": "user", "content": [{"text": tagline}]}],
                inferenceConfig={"maxTokens": 40, "temperature": temp},
            )
            print(f"  run {run + 1}: {resp['output']['message']['content'][0]['text'].strip()}")
        print()

    print("  At temperature=0 the runs are near-identical; at 0.7-1.0 they diverge -")
    print("  which is why classification/extraction use 0 and brainstorming uses higher.\n")


# ============================================================
# Main
# ============================================================

def main():
    demo_max_tokens()
    demo_stop_sequences()
    demo_temperature()

    print("--- Done ---")
    print("  Right-size max_tokens (set it tight, raise only on truncation), use")
    print("  stop_sequences when a real terminator exists, and match temperature to")
    print("  the task.")


if __name__ == "__main__":
    main()
