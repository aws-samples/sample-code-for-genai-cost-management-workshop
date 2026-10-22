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
- Compare Claude's cache pricing (1.25x write, 0.1x read) with Amazon Nova 2 Lite's
  (no write premium, 0.25x read) - same cachePoint, same Converse API

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
- Access to Claude 5 Sonnet and Amazon Nova 2 Lite on Amazon Bedrock (both support caching)
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

SONNET = "global.anthropic.claude-sonnet-5"
NOVA = "global.amazon.nova-2-lite-v1:0"

# Prompt caching is best effort: a call right after the write can still miss, and
# the miss writes (and bills) the prefix again. Waiting does not reliably help, so
# the demo simply asks again, a bounded number of times, until it sees a read.
MAX_READ_ATTEMPTS = 3

# Input price per 1M tokens (USD, illustrative) and cache price multipliers.
# Claude bills a cache write at ~1.25x and a cache read at ~0.1x. Nova 2 Lite has
# no cache-write premium (the AWS price list shows a $0 cache-write meter; this
# sample conservatively bills written tokens at the standard input rate, 1x) and
# bills a cache read at 0.25x ($0.075 vs $0.30 per 1M). Verify at
# https://aws.amazon.com/bedrock/pricing/
MODELS = {
    SONNET: {"label": "Sonnet 5", "input_per_m": 3.00, "write_mult": 1.25, "read_mult": 0.10},
    NOVA: {"label": "Nova 2 Lite", "input_per_m": 0.30, "write_mult": 1.00, "read_mult": 0.25},
}


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
    # The revision stamp makes the prefix new on every run. Without it, running the
    # sample twice within the cache TTL (about 5 minutes) would show a cache READ on
    # "call 1", because the previous run's entry is still warm.
    return (
        "You are the support operations assistant for Example Corp. Answer strictly "
        "according to the following internal support manual. Be concise and cite the "
        "relevant section number.\n\n"
        f"Manual revision: {int(time.time())}\n"
        f"{manual}"
    )


# ============================================================
# Converse with a cachePoint on the static prefix
# ============================================================

def ask_with_cache(model_id: str, manual: str, question: str) -> dict:
    """Cache the manual (static, first), then ask a volatile question (last)."""
    content = [
        {"text": f"Use the support manual below to answer questions.\n\nMANUAL:\n{manual}"},
        {"cachePoint": {"type": "default"}},  # everything before this is cached
        {"text": f"\n\nQUESTION: {question}"},
    ]
    resp = RUNTIME.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": content}],
        inferenceConfig={"maxTokens": 200},  # no temperature: deprecated on Claude 5
    )
    usage = resp["usage"]
    return {
        "answer": resp["output"]["message"]["content"][0]["text"],
        "input": usage.get("inputTokens", 0),
        "cache_read": usage.get("cacheReadInputTokens", 0),
        "cache_write": usage.get("cacheWriteInputTokens", 0),
        "output": usage.get("outputTokens", 0),
    }


def input_cost(model_id: str, r: dict) -> tuple[float, float]:
    """Return (cost with caching, cost if nothing were cached) for one call's input."""
    m = MODELS[model_id]
    price = m["input_per_m"] / 1_000_000
    with_cache = price * (
        r["input"]
        + r["cache_write"] * m["write_mult"]
        + r["cache_read"] * m["read_mult"]
    )
    without_cache = price * (r["input"] + r["cache_write"] + r["cache_read"])
    return with_cache, without_cache


def print_comparison_table(calls: list[tuple[str, str, dict]]) -> None:
    """Print each call's token breakdown and input cost, with and without caching."""
    print("--- Comparison: input cost with and without caching ---")
    print(
        f"{'Model':<12} {'Call':<15} {'Input':>7} {'Cache wr':>9} {'Cache rd':>9} {'Output':>7} "
        f"{'Input $ cached':>15} {'Input $ no cache':>17}"
    )
    totals = {}
    for label, model_id, r in calls:
        with_cache, without_cache = input_cost(model_id, r)
        t = totals.setdefault(model_id, [0.0, 0.0])
        t[0] += with_cache
        t[1] += without_cache
        print(
            f"{MODELS[model_id]['label']:<12} {label:<15} {r['input']:>7} {r['cache_write']:>9} "
            f"{r['cache_read']:>9} {r['output']:>7} {with_cache:>15.6f} {without_cache:>17.6f}"
        )
    print()
    for model_id, (with_cache, without_cache) in totals.items():
        change = (with_cache - without_cache) / without_cache if without_cache else 0.0
        print(
            f"{MODELS[model_id]['label']:<12} {'Total':<15} {'':>7} {'':>9} {'':>9} {'':>7} "
            f"{with_cache:>15.6f} {without_cache:>17.6f}   ({change:+.0%} vs no caching)"
        )
    print(
        "\n  Claude pays a 1.25x write premium once, then 0.1x per read, so it breaks even"
        "\n  after about 3 reads. Nova 2 Lite has no write premium and reads at 0.25x, so"
        "\n  it saves from the first read - but the cached tokens are cheap to begin with."
        "\n  Compare the dollars, not just the percentages. A read can miss (Nova 2 Lite"
        "\n  often does right after the write); a miss re-writes the prefix, so its retry"
        "\n  rows count toward the total.\n"
    )


def show(label: str, model_id: str, manual: str, question: str) -> dict:
    r = ask_with_cache(model_id, manual, question)
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
        f"--- Prompt Caching: cache a ~{approx_tokens}-token manual, ask two questions per model ---\n"
    )

    calls = []
    for model_id in MODELS:
        label = MODELS[model_id]["label"]
        print(f"=== {label} ({model_id}) ===\n")
        first = show(
            "Call 1 (cache WRITE expected - first time the manual is seen):",
            model_id,
            manual,
            "What should an agent do when a refund exceeds the standard limit?",
        )
        calls.append(("Call 1 (write)", model_id, first))
        for attempt in range(1, MAX_READ_ATTEMPTS + 1):
            second = show(
                "Call 2 (cache READ expected - same manual, different question):"
                if attempt == 1
                else f"Call 2, attempt {attempt} (the previous call missed the cache):",
                model_id,
                manual,
                "How should an agent verify a customer's identity before making changes?",
            )
            calls.append(
                ("Call 2 (read)" if attempt == 1 else f"Call 2 retry {attempt - 1}", model_id, second)
            )
            if second["cache_read"] > 0:
                break

        if second["cache_read"] > 0:
            m = MODELS[model_id]
            saved_tokens = second["cache_read"]
            full_price = (saved_tokens / 1_000_000) * m["input_per_m"]
            cache_price = full_price * m["read_mult"]
            print(
                f"  Cache hit: {saved_tokens} input tokens served from cache at {m['read_mult']}x."
                f"\n  Those tokens cost ~${cache_price:.6f} instead of ~${full_price:.6f} this call.\n"
            )
        else:
            print(
                "  No cache read recorded. The prefix may be below the model's token"
                "\n  threshold, the TTL may have expired, or caching is not active for this"
                "\n  model/Region. Enlarge the prefix or check the model card token minimum.\n"
            )

    print_comparison_table(calls)
    print("  The larger and more frequently reused the prefix, the bigger the savings.\n")


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
