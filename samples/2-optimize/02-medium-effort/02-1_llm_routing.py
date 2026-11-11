"""
Medium-Effort Optimization - Lever 07: LLM Routing

Query complexity has a long tail. In a typical workload only ~20% of requests
genuinely need a workhorse like Claude 5 Sonnet; the other ~80% are factual
lookups or single-step classifications Haiku 4.5 answers just as well at roughly
a third of the cost. Routing classifies each request, then sends it to the
cheapest model that can handle it.

The routing decision - labeling each query simple/complex so you can send it to
the cheapest capable model - is made by a tiny, deterministic Haiku Converse
round-trip per query (run_llm_classification). Few-shot examples pin the output
to one word and a stop sequence ends generation right after it, so it costs only
a few output tokens per request.

You will learn how to:
- Build a tiny, deterministic LLM classifier that labels a query simple/complex
- Map each label to the cheapest capable model (Haiku 4.5 or Claude 5 Sonnet)
- Measure what the classifier itself costs in latency and Bedrock spend

Two rules keep routing a win:
- The classifier must be cheap and deterministic - a low maxTokens, a stop
  sequence, and temperature 0 on models that allow it.
- The cheap path must take the majority of traffic, and you must confirm
  routed-down quality on your own eval set, not a vendor benchmark.

The failure mode to watch is the floor: if the classifier makes the "simple"
path slower or pricier than just calling the big model, routing has cost you.

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude Haiku 4.5 and Claude 5 Sonnet on Amazon Bedrock
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
SONNET = "global.anthropic.claude-sonnet-5"

# Illustrative input pricing per 1M tokens (USD). Verify current pricing at
# https://aws.amazon.com/bedrock/pricing/
PRICING = {HAIKU: {"input": 1.00, "output": 5.00}, SONNET: {"input": 3.00, "output": 15.00}}


# ============================================================
# Helpers
# ============================================================

def supports_temperature(model_id: str) -> bool:
    """Claude 5 models (Sonnet 5, Opus 5) deprecated temperature."""
    return "claude-sonnet-5" not in model_id and "claude-opus-5" not in model_id


def converse(model_id: str, prompt: str, max_tokens: int, stop=None) -> dict:
    """Single-turn Converse call returning text and usage."""
    cfg = {"maxTokens": max_tokens}
    if supports_temperature(model_id):
        cfg["temperature"] = 0
    if stop:
        cfg["stopSequences"] = stop
    resp = RUNTIME.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig=cfg,
    )
    return {
        "text": resp["output"]["message"]["content"][0]["text"],
        "input_tokens": resp["usage"]["inputTokens"],
        "output_tokens": resp["usage"]["outputTokens"],
    }


# ============================================================
# The classifier and the router
# ============================================================

def classify_complexity(query: str) -> dict:
    """Label a query 'simple' or 'complex' with a tiny, deterministic classifier.

    Few-shot examples pin the output to one word; the '###' stop sequence ends
    generation right after it, so the classifier costs only a few output tokens.
    """
    prompt = (
        "Classify the customer query as exactly one word: 'simple' (a factual "
        "lookup or single-step question, including a straightforward policy or "
        "product question) or 'complex' (multi-step reasoning, an ambiguous "
        "situation, or a policy edge case needing interpretation). Output only "
        "the word.\n\n"
        "Query: What are your store hours?\nClassification: simple ###\n"
        "Query: What's your return policy on headphones?\nClassification: simple ###\n"
        "Query: My order arrived damaged but it's past the return window; what can you do?\n"
        "Classification: complex ###\n"
        "Query: I returned 2 of 3 bundled items; what refund is owed on the third?\n"
        "Classification: complex ###\n"
        f"Query: {query}\nClassification:"
    )
    r = converse(HAIKU, prompt, max_tokens=5, stop=["###"])
    return {
        "label": r["text"].strip().lower(),
        "input_tokens": r["input_tokens"],
        "output_tokens": r["output_tokens"],
    }


# ============================================================
# Sample queries - two clear-simple, two clear-complex
# ============================================================

QUERIES = [
    "What are your store hours?",  # simple - factual lookup
    "What's your return policy on opened headphones?",  # simple - policy lookup
    "My laptop arrived damaged but it's been 35 days; what can you do?",  # complex - policy edge case
    "I bought 3 phones, returned 2 last week, and want to return the third now, "
    "but the receipt says they were a bundle deal - what refund am I owed?",  # complex
]


def route_target(label: str) -> str:
    """Pick the cheapest capable model for a complexity label."""
    return HAIKU if "simple" in label else SONNET


# ============================================================
# Classify with an LLM
# ============================================================

def run_llm_classification(queries) -> dict:
    """Classify every query with the Haiku LLM classifier.

    Measures total latency and Bedrock token spend for the routing decision
    alone (no answering call), prints the per-query label, and returns a summary.
    """
    print("LLM classifier (Haiku Converse):")
    latency = 0.0
    input_tokens = 0
    output_tokens = 0
    for q in queries:
        t0 = time.perf_counter()
        c = classify_complexity(q)
        latency += time.perf_counter() - t0
        input_tokens += c["input_tokens"]
        output_tokens += c["output_tokens"]
        print(f"  [{c['label']:8}] -> {route_target(c['label']).split('.')[-1][:24]}  {q[:52]}")

    p = PRICING[HAIKU]
    cost = (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
    print(
        f"  total: {latency * 1000:.0f}ms for {len(queries)} queries, "
        f"{input_tokens} input + {output_tokens} output tokens, "
        f"~${cost:.6f} in Bedrock spend\n"
    )
    return {"latency_ms": latency * 1000, "cost_usd": cost,
            "input_tokens": input_tokens, "output_tokens": output_tokens}


# ============================================================
# Main
# ============================================================

def main():
    print("--- Routing decision: classify each query with Haiku ---\n")
    result = run_llm_classification(QUERIES)

    print("Takeaway:")
    print(
        f"  Classifying {len(QUERIES)} queries took {result['latency_ms']:.0f}ms and "
        f"~${result['cost_usd']:.6f} in Bedrock spend.\n"
        "  That overhead is the price of routing: it pays off only when the cheap\n"
        "  path takes the majority of traffic and the classifier stays cheaper than\n"
        "  the savings. Confirm routed-down quality on your own eval set before\n"
        "  shipping.\n"
    )

    print("--- Done ---")
    print("  Next steps:")
    print("  1. Cache the classifier's static prompt (see 01-4_prompt_caching.py)")
    print("  2. Measure the routed-down quality on your eval set before shipping")
    print("  3. Ensure the cheap path takes the majority of traffic")


if __name__ == "__main__":
    main()
