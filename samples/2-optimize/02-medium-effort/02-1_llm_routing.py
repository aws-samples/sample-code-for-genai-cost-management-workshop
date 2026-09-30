"""
Medium-Effort Optimization - Lever 07: LLM Routing

Query complexity has a long tail. In a typical workload only ~20% of requests
genuinely need a workhorse like Claude 5 Sonnet; the other ~80% are factual
lookups or single-step classifications Haiku 4.5 answers just as well at roughly
a third of the cost. Routing classifies each request, then sends it to the
cheapest model that can handle it.

You will learn how to:
- Build a tiny, deterministic classifier that labels a query simple/complex
- Route simple queries to Haiku and complex ones to Claude 5 Sonnet
- See the classifier cost only a few output tokens per request

Two rules keep routing a win:
- The classifier must be tiny and deterministic - temperature 0 (on models that
  allow it), a low maxTokens, a stop sequence, and cache its static prompt.
- The cheap path must take the majority of traffic, and you must confirm
  routed-down quality on your own eval set, not a vendor benchmark.

The failure mode to watch is the floor: if the classifier round-trip makes the
"simple" path slower or pricier than just calling the big model, routing has
cost you.

NOTE: The notebook this is adapted from wires a tool-backed Strands agent as the
answering model. To keep this sample self-contained (plain boto3, no extra
dependencies), the answering step here is a direct Converse call - the routing
logic (classify, then pick the model) is the actual lever.

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude Haiku 4.5 and Claude 5 Sonnet on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
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
    return {"label": r["text"].strip().lower(), "classifier_output_tokens": r["output_tokens"]}


def route_and_answer(query: str) -> dict:
    """Classify the query, route to the cheapest capable model, then answer."""
    c = classify_complexity(query)
    target = HAIKU if "simple" in c["label"] else SONNET

    answer = converse(
        target,
        "You are a concise customer-support agent. Answer the customer's question "
        f"directly and briefly.\n\nQuestion: {query}",
        max_tokens=300,
    )
    return {
        "label": c["label"],
        "classifier_output_tokens": c["classifier_output_tokens"],
        "routed_to": target.split(".")[-1][:24],
        "answer": answer["text"],
        "answer_input_tokens": answer["input_tokens"],
        "answer_output_tokens": answer["output_tokens"],
    }


# ============================================================
# Demo
# ============================================================

def demo_routing() -> None:
    print("--- LLM Routing: classify, then route to the cheapest capable model ---\n")

    queries = [
        "What are your store hours?",  # simple - factual lookup
        "What's your return policy on opened headphones?",  # simple - policy lookup
        "My laptop arrived damaged but it's been 35 days; what can you do?",  # complex - policy edge case
        "I bought 3 phones, returned 2 last week, and want to return the third now, "
        "but the receipt says they were a bundle deal - what refund am I owed?",  # complex
    ]

    routed_haiku = 0
    for q in queries:
        r = route_and_answer(q)
        if "haiku" in r["routed_to"]:
            routed_haiku += 1
        print(f"[{r['label']}] routed to {r['routed_to']}  "
              f"(classifier used {r['classifier_output_tokens']} output tokens)")
        print(f"  Q: {q}")
        print(f"  A: {r['answer'][:200]}\n")

    print(
        f"  {routed_haiku}/{len(queries)} queries routed to the cheaper model."
        "\n  In a real workload ~80% land on Haiku at ~3x lower cost, while the"
        "\n  classifier adds only a few output tokens per request. Confirm routed-down"
        "\n  quality on your own eval set.\n"
    )


# ============================================================
# Main
# ============================================================

def main():
    demo_routing()

    print("--- Done ---")
    print("  Next steps:")
    print("  1. Cache the classifier's static prompt (see 01-4_prompt_caching.py)")
    print("  2. Measure the routed-down quality on your eval set before shipping")
    print("  3. Ensure the cheap path takes the majority of traffic")


if __name__ == "__main__":
    main()
