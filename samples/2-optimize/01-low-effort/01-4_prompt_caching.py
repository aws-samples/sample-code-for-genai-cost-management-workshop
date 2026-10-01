"""
Low-Effort Optimization - Lever 04: Prompt Caching

Most requests resend the same system prompt, tool definitions, or reference
document every time, paying full input price for tokens that never change.
Caching pays full price once (the write, billed at ~1.25x), then ~0.1x for
every read - break-even is around 3 reads, after which it is pure savings.

You will learn how to:
- Place a cachePoint marker to cache a large, stable prefix on the Converse API
- Read cacheWriteInputTokens (miss) and cacheReadInputTokens (hit) from usage
- See the input-token accounting shift from full price to cache-read price

How it works:
- Bedrock caches explicitly for Claude. You place a cachePoint marker; no marker
  means nothing is cached (no implicit fallback).
- The cache key is the longest token prefix from position 0. Put static content
  first (system prompt, tool schemas, documents) and dynamic content (the user
  question) last, so the largest possible prefix is cacheable.
- Caching is exact-prefix. Editing one word inside the cached region busts it and
  the next call pays the write premium again. Watch cache hit rate like an SLO.

NOTE: Each cache checkpoint must meet a per-model minimum token threshold to
activate (varies by model - check the model card). This sample builds a large
static prefix in code so it clears the threshold.

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to a prompt-caching capable Claude model
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import boto3

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

RUNTIME = boto3.client("bedrock-runtime", region_name=REGION)

SONNET = "global.anthropic.claude-sonnet-5"

# Input price per 1M tokens (USD, illustrative). Cache reads bill at ~0.1x,
# cache writes at ~1.25x. Verify at https://aws.amazon.com/bedrock/pricing/
INPUT_PRICE_PER_M = 3.00


# ============================================================
# A large, stable document worth caching
# ============================================================

def build_reference_manual() -> str:
    """Build a large static support manual to use as the cached prefix.

    In a real application this is your product docs, policy manual, tool
    schemas, or retrieved RAG context - anything large and reused across calls.
    """
    sections = []
    for i in range(1, 41):
        sections.append(
            f"Section {i}. Support procedure {i}: verify the customer's identity, "
            f"check the order status in the fulfillment system, confirm the return "
            f"window for the product category, and record the interaction with a "
            f"category code. Escalate to tier two when a refund exceeds the standard "
            f"limit or a contracted service level agreement must change."
        )
    manual = "\n".join(sections)
    return (
        "You are the support operations assistant for Example Corp. Answer strictly "
        "according to the following internal support manual. Be concise and cite the "
        "relevant section number.\n\n"
        f"{manual}"
    )


# ============================================================
# Converse with a cachePoint on the static prefix
# ============================================================

def ask_with_cache(manual: str, question: str) -> dict:
    """Cache the manual (static, first), then ask a volatile question (last)."""
    content = [
        {"text": f"Use the support manual below to answer questions.\n\nMANUAL:\n{manual}"},
        {"cachePoint": {"type": "default"}},  # everything before this is cached
        {"text": f"\n\nQUESTION: {question}"},
    ]
    resp = RUNTIME.converse(
        modelId=SONNET,
        messages=[{"role": "user", "content": content}],
        inferenceConfig={"maxTokens": 200},  # Claude 5 Sonnet: temperature deprecated
    )
    usage = resp["usage"]
    return {
        "answer": resp["output"]["message"]["content"][0]["text"],
        "input": usage.get("inputTokens", 0),
        "cache_read": usage.get("cacheReadInputTokens", 0),
        "cache_write": usage.get("cacheWriteInputTokens", 0),
        "output": usage.get("outputTokens", 0),
    }


def show(label: str, manual: str, question: str) -> dict:
    r = ask_with_cache(manual, question)
    print(label)
    print(f"  Q: {question}")
    print(f"  A: {r['answer'][:200]}")
    print(
        f"  usage: input={r['input']} cache_read={r['cache_read']} "
        f"cache_write={r['cache_write']} output={r['output']}\n"
    )
    return r


# ============================================================
# Demo
# ============================================================

def demo_cache_write_then_read() -> None:
    """First call writes to cache (miss); second reads from cache (hit)."""
    manual = build_reference_manual()
    approx_tokens = len(manual) // 4
    print(
        f"--- Prompt Caching: cache a ~{approx_tokens}-token manual, ask two questions ---\n"
    )

    first = show(
        "Call 1 (cache WRITE expected - first time the manual is seen):",
        manual,
        "What should an agent do when a refund exceeds the standard limit?",
    )
    second = show(
        "Call 2 (cache READ expected - same manual, different question):",
        manual,
        "How should an agent verify a customer's identity before making changes?",
    )

    if second["cache_read"] > 0:
        saved_tokens = second["cache_read"]
        full_price = (saved_tokens / 1_000_000) * INPUT_PRICE_PER_M
        cache_price = full_price * 0.1
        print(
            f"  Cache hit: {saved_tokens} input tokens served from cache at ~0.1x."
            f"\n  Those tokens cost ~${cache_price:.6f} instead of ~${full_price:.6f} this call."
            "\n  The larger and more frequently reused the prefix, the bigger the savings.\n"
        )
    else:
        print(
            "  No cache read recorded. The prefix may be below the model's token"
            "\n  threshold, the TTL may have expired, or caching is not active for this"
            "\n  model/Region. Enlarge the prefix or check the model card token minimum.\n"
        )


# ============================================================
# Main
# ============================================================

def main():
    demo_cache_write_then_read()

    print("--- Done ---")
    print("  Next steps:")
    print("  1. Identify high-volume prompts that share a large static prefix")
    print("  2. Place static content first, dynamic content last, add a cachePoint between")
    print("  3. Confirm cacheReadInputTokens > 0 and track cache hit rate (target >= 70%)")


if __name__ == "__main__":
    main()
