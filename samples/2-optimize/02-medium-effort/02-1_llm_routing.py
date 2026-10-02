"""
Medium-Effort Optimization - Lever 07: LLM Routing

Query complexity has a long tail. In a typical workload only ~20% of requests
genuinely need a workhorse like Claude 5 Sonnet; the other ~80% are factual
lookups or single-step classifications Haiku 4.5 answers just as well at roughly
a third of the cost. Routing classifies each request, then sends it to the
cheapest model that can handle it.

The routing decision - labeling each query simple/complex so you can send it to
the cheapest capable model - can be made two ways, and this sample implements
and compares both:
- An LLM classifier (run_llm_classification) - a tiny, deterministic Haiku
  Converse round-trip per query. Few-shot examples pin the output to one word
  and a stop sequence ends generation right after it, so it costs only a few
  output tokens per request.
- A decision model (run_decider_classification) - Strands Decider 2B, a small
  open-source "system one" model that runs locally and picks between options
  with a calibrated confidence on every answer, so the routing decision never
  touches Bedrock at all. See
  https://strandsagents.com/blog/introducing-strands-decider/ . It reads the
  query once and answers in ~115-150ms on local hardware at near-zero marginal
  cost; its confidence also lets you escalate uncertain decisions to the bigger
  model, which a bare label cannot.

You will learn how to:
- Build a tiny, deterministic LLM classifier that labels a query simple/complex
- Make the same decision with a local decision model at no Bedrock cost
- Compare the two on latency and cost so you can pick the right one

Two rules keep routing a win:
- The classifier must be cheap and deterministic - a low maxTokens, a stop
  sequence, and (for the LLM) temperature 0 on models that allow it.
- The cheap path must take the majority of traffic, and you must confirm
  routed-down quality on your own eval set, not a vendor benchmark.

The failure mode to watch is the floor: if the classifier makes the "simple"
path slower or pricier than just calling the big model, routing has cost you.

The decision model is reached over its local HTTP server (POST /v1/systemone).
Running it as a server loads the 1.9B model once and keeps it warm, so each
query pays only the real decision latency - not a per-call model reload. This
script manages that server for you: start_decider_server launches it as a child
process and waits until it is ready, and stop_decider_server shuts it down when
the comparison is done. If one is already running at DECIDER_URL it is reused.
All it needs is the model installed (pip install strands-decider); if the CLI
is absent the sample skips the decision-model path and still runs the LLM
classifier. To run the server yourself instead, start it before this script:

    strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude Haiku 4.5 and Claude 5 Sonnet on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
- OPTIONAL, for the decision-model path: pip install strands-decider, then run
  the serve command above (first run downloads the weights). The sample
  degrades gracefully and skips this path if the server is not reachable.
"""

import os
import shutil
import subprocess
import time

import boto3
import requests

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

# Strands Decider 2B - a small open-source decision model served locally. The
# decision runs on your own hardware, so the routing step costs no Bedrock
# tokens. Install with: pip install strands-decider, then start the server:
#   strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099
DECIDER_MODEL = "StrandsAgents/strands-decider-2B-hobson-v19"
DECIDER_PORT = int(os.environ.get("DECIDER_PORT", "8099"))
DECIDER_URL = os.environ.get("DECIDER_URL", f"http://localhost:{DECIDER_PORT}/v1/systemone")


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


def classify_complexity_decider(query: str) -> dict:
    """Label a query 'simple' or 'complex' with the Strands Decider model.

    Instead of a Bedrock round-trip, this POSTs a two-option `choice` question to
    the locally served decider (POST /v1/systemone). The server loads the 1.9B
    model once and keeps it warm, so each call pays only the real decision
    latency. The decision runs on local hardware, so it costs no Bedrock tokens,
    and the model returns a calibrated confidence on every answer - letting us
    escalate low-confidence calls to the bigger model rather than trusting a bare
    label.

    Returns the parsed label, confidence, and the server-reported latency, or
    {"available": False} when the server is not reachable, so the caller can
    fall back gracefully.
    """
    payload = {
        "state": query,
        "questions": {
            "complexity": {
                "type": "choice",
                "instructions": "How complex is this customer query?",
                "criteria": {
                    "simple": "a factual lookup or single-step question",
                    "complex": "multi-step reasoning, an ambiguous situation, "
                               "or a policy edge case needing interpretation",
                },
            }
        },
    }
    try:
        resp = requests.post(DECIDER_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        answer = data["answers"]["complexity"]
    except (requests.RequestException, KeyError, ValueError):
        return {"available": False}

    return {
        "available": True,
        "label": str(answer["choice"]).lower(),
        "confidence": float(answer["confidence"]),
        "server_latency_ms": float(data.get("latency_ms", 0.0)),
    }


# ============================================================
# Managing the local Decider server
# ============================================================

def decider_server_up() -> bool:
    """True if a Decider server is already answering at DECIDER_URL."""
    try:
        resp = requests.post(
            DECIDER_URL,
            json={"state": "ping", "questions": {"q": {"type": "noul", "instructions": "ready?"}}},
            timeout=5,
        )
        return resp.status_code == 200
    except requests.RequestException:
        return False


def start_decider_server(ready_timeout: float = 600.0) -> subprocess.Popen | None:
    """Start `strands-decider serve` as a child process and wait until it is ready.

    The timeout is generous (10 min) because the very first run downloads the
    model weights from Hugging Face before the server can answer; warm starts
    take seconds.

    Returns the process handle to pass to stop_decider_server, or None if the
    CLI is not installed or the server never became ready (the sample then skips
    the decision-model path and still runs the LLM classifier). If a server is
    already up at DECIDER_URL, reuses it and returns None (nothing to stop).
    """
    if decider_server_up():
        print(f"  Reusing a Decider server already running at {DECIDER_URL}\n")
        return None

    if shutil.which("strands-decider") is None:
        print(
            "  strands-decider CLI not found - skipping the decision-model path.\n"
            "  Install it with `pip install strands-decider` to enable it.\n"
        )
        return None

    print(
        f"  Starting Decider server on port {DECIDER_PORT}...\n"
        "  This may take a while: the ~1.9B model loads once, and the very first\n"
        "  run also downloads the weights from Hugging Face (several GB). Later\n"
        "  runs start faster from the local cache. Waiting for it to be ready..."
    )
    proc = subprocess.Popen(
        ["strands-decider", "serve", DECIDER_MODEL, "--port", str(DECIDER_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + ready_timeout
    while time.time() < deadline:
        if proc.poll() is not None:  # process exited before becoming ready
            print("  Decider server exited before it was ready - skipping that path.\n")
            return None
        if decider_server_up():
            print("  Decider server is ready.\n")
            return proc
        time.sleep(2)

    print("  Decider server did not become ready in time - skipping that path.\n")
    stop_decider_server(proc)
    return None


def stop_decider_server(proc: subprocess.Popen | None) -> None:
    """Terminate a Decider server started by start_decider_server."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print(f"  Stopped the Decider server on port {DECIDER_PORT}.")


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
# Approach 1: classify with an LLM
# ============================================================

def run_llm_classification(queries) -> dict:
    """Classify every query with the Haiku LLM classifier.

    Measures total latency and Bedrock token spend for the routing decision
    alone (no answering call), prints the per-query label, and returns a summary.
    """
    print("Approach 1 - LLM classifier (Haiku Converse):")
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
# Approach 2: classify with the Strands Decider model
# ============================================================

def run_decider_classification(queries) -> dict | None:
    """Classify every query with the locally served Strands Decider model.

    Measures round-trip latency and the server-reported model time, prints the
    per-query label with its calibrated confidence, and returns a summary. If
    the server is not reachable, prints setup instructions and returns None so
    the sample degrades gracefully.
    """
    print("Approach 2 - Strands Decider (local decision model):")

    probe = classify_complexity_decider(queries[0])
    if not probe.get("available"):
        print(
            f"  Decider server not reachable at {DECIDER_URL} - skipping this approach.\n"
            "  Install the model with `pip install strands-decider`, then start the\n"
            "  server (it loads the 1.9B model once and keeps it warm):\n"
            f"    strands-decider serve {DECIDER_MODEL} --port 8099\n"
        )
        return None

    latency = 0.0
    server_latency = 0.0
    for q in queries:
        t0 = time.perf_counter()
        d = classify_complexity_decider(q)
        latency += time.perf_counter() - t0
        if not d.get("available"):
            print(f"  [{'?':8}] (server error)  {q[:52]}")
            continue
        server_latency += d["server_latency_ms"]
        print(f"  [{d['label']:8}] -> {route_target(d['label']).split('.')[-1][:24]}  "
              f"(confidence {d['confidence']:.3f})  {q[:40]}")

    print(
        f"  total: {latency * 1000:.0f}ms round-trip for {len(queries)} queries "
        f"({server_latency:.0f}ms spent in the model), "
        f"0 Bedrock tokens, ~$0.000000 in Bedrock spend\n"
    )
    return {"latency_ms": latency * 1000, "server_latency_ms": server_latency, "cost_usd": 0.0}


# ============================================================
# Compare the two approaches
# ============================================================

def compare_routing_decision() -> None:
    """Run both classification approaches on the same queries and compare them."""
    print("--- Routing decision: LLM classifier vs decision model ---\n")

    llm = run_llm_classification(QUERIES)
    decider = run_decider_classification(QUERIES)

    print("Takeaway:")
    if decider is None:
        print(
            "  The decision model moves routing OFF Bedrock entirely - no classifier\n"
            "  tokens and no per-request Bedrock spend on the routing step - and returns\n"
            "  a calibrated confidence you can use to escalate uncertain calls to the\n"
            "  bigger model. Start the server above to see the live comparison.\n"
        )
        return

    print(
        f"  LLM classifier: {llm['latency_ms']:.0f}ms, ~${llm['cost_usd']:.6f} in Bedrock spend.\n"
        f"  Decision model: {decider['latency_ms']:.0f}ms round-trip "
        f"({decider['server_latency_ms']:.0f}ms in the model), $0 in Bedrock spend.\n"
        "  The decision model makes the routing call with no Bedrock tokens, and its\n"
        "  calibrated confidence lets you escalate low-confidence decisions to the\n"
        "  bigger model - something a bare LLM label cannot do. Confirm routed-down\n"
        "  quality on your own eval set before shipping either one.\n"
    )


# ============================================================
# Main
# ============================================================

def main():
    print("--- Starting the local Decider server ---")
    server = start_decider_server()
    try:
        compare_routing_decision()
    finally:
        stop_decider_server(server)

    print("--- Done ---")
    print("  Next steps:")
    print("  1. Cache the LLM classifier's static prompt (see 01-4_prompt_caching.py)")
    print("  2. Measure the routed-down quality on your eval set before shipping")
    print("  3. Ensure the cheap path takes the majority of traffic")
    print("  4. For the routing decision itself, weigh a local decision model")
    print("     (Strands Decider) against an LLM classifier using the comparison above")


if __name__ == "__main__":
    main()
