"""
Low-Effort Optimization - Lever 05: Retry, Turn and Stop Limits

Do not retry or loop without a limit. A retry that never ends, an agent loop
with no turn cap, or a tool that is called forever turns one bad request into an
unbounded bill. Cap retries, turns, and tool calls based on your task's real
success rates, and always decide what happens when a cap is hit.

You will learn how to:
- Cap SDK retries (botocore retry config) and see what a cap trades off
- Cap an application-level retry loop, where every retry pays for tokens again
- Cap a tool-use loop by turns and tool calls, and see input tokens grow each turn
- Read stopReason so a loop knows why the model stopped

Key points:
- SDK retries (throttling, 5xx) are cheap per attempt - a throttled request is
  not billed - but each retry adds latency and holds capacity. Set the cap from
  how long transient failures actually last, not from habit.
- Application retries (the model answered, your validator rejected it) are
  billed in full on every attempt. Retry only failures that are transient. A
  mis-specified prompt or validator fails every time, so retrying only burns
  tokens. Fix the cause instead.
- A tool-use loop resends the whole conversation each turn, so input tokens grow
  with every turn. Cap max turns and max tool calls, and return a graceful
  fallback (escalate, ask the user) when a cap is hit.
- stopReason tells you why generation stopped: end_turn, tool_use, max_tokens,
  or stop_sequence. A loop must handle each one explicitly.

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude Haiku 4.5 on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import json
import os
import time
import boto3
from botocore.awsrequest import AWSResponse
from botocore.config import Config
from botocore.exceptions import ClientError

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# Pricing per 1M tokens (USD). Illustrative workshop values - always verify
# current pricing at https://aws.amazon.com/bedrock/pricing/
INPUT_PRICE_PER_M = 1.00
OUTPUT_PRICE_PER_M = 5.00

# --- Part 1: SDK retry cap -----------------------------------------------
# The demo simulates a short throttling outage: the first N attempts are
# answered with ThrottlingException before the request reaches Bedrock, so no
# tokens are billed for them and the demo never depends on real throttling.
SIMULATED_THROTTLED_ATTEMPTS = 3
# Retry caps to compare: a low cap that gives up during the outage, and a higher
# cap that rides it out. In botocore's retry config the value counts RETRIES, on
# top of the first call, so a cap of 1 means at most 2 attempts in total.
SDK_MAX_RETRIES = [1, 4]

# --- Part 2: application retry cap ---------------------------------------
# Retries of a call whose answer fails YOUR validation. Every attempt is billed.
MAX_APP_RETRIES = 3

# --- Part 3: tool-use loop caps ------------------------------------------
# (label, max_turns, max_tool_calls). The tool below never reports DELIVERED, so
# an agent told to wait for it keeps calling until a cap stops it.
LOOP_CAPS = [
    ("tight caps", 4, 2),
    ("loose caps", 10, 10),
]

PROMPT = "Reply with exactly one word: ready"

# Results collected as the demos run and printed as one comparison table at the end.
# Each entry: (lever, setting, attempts_or_turns, seconds or None, billed_usd, outcome)
RESULTS: list[tuple[str, str, int, float | None, float, str]] = []

ORDER_MESSAGE = "Hi, where is my order? The order number is A-48213."

# ============================================================
# Helper Functions
# ============================================================


def cost(input_tokens: int, output_tokens: int) -> float:
    """Return the dollar cost of a request on Haiku 4.5."""
    return (input_tokens / 1_000_000) * INPUT_PRICE_PER_M + (
        output_tokens / 1_000_000
    ) * OUTPUT_PRICE_PER_M


def make_client(max_retries: int, throttled_attempts: int):
    """Build a Bedrock client with a capped retry config.

    max_retries is passed as botocore's retries["max_attempts"], which counts
    retries after the first call. "standard" mode retries throttling and
    transient errors with exponential backoff and jitter.

    A before-send hook answers the first `throttled_attempts` attempts with a
    simulated ThrottlingException. Botocore's retry handler then treats them like
    real throttles, so the retry cap behaves exactly as it would in production.
    """
    client = boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        config=Config(retries={"max_attempts": max_retries, "mode": "standard"}),
    )
    seen = {"count": 0}

    class _Body:
        def __init__(self, data: bytes):
            self._data = data

        def stream(self, *args, **kwargs):
            yield self._data

        def read(self, *args, **kwargs):
            return self._data

    def simulate_throttle(request, **kwargs):
        seen["count"] += 1
        if seen["count"] > throttled_attempts:
            return None  # let the real request go through
        body = json.dumps({"message": "Too many requests, please wait."}).encode()
        headers = {
            "x-amzn-errortype": "ThrottlingException",
            "content-type": "application/json",
        }
        return AWSResponse(request.url, 429, headers, _Body(body))

    client.meta.events.register("before-send.bedrock-runtime.Converse", simulate_throttle)
    return client


# ============================================================
# 1. SDK retry cap
# ============================================================


def demo_sdk_retry_cap() -> None:
    """Same short outage, two caps: one gives up, one recovers (and costs latency)."""
    print("--- 1. SDK retry cap: the cap trades give-up risk for latency ---")
    print(
        f"  Simulated outage: the first {SIMULATED_THROTTLED_ATTEMPTS} attempts are throttled "
        "(no tokens billed for these).\n"
    )

    rows = []
    for max_retries in SDK_MAX_RETRIES:
        client = make_client(max_retries, SIMULATED_THROTTLED_ATTEMPTS)
        t0 = time.time()
        try:
            resp = client.converse(
                modelId=HAIKU,
                messages=[{"role": "user", "content": [{"text": PROMPT}]}],
                inferenceConfig={"maxTokens": 10},
            )
            attempts = resp["ResponseMetadata"].get("RetryAttempts", 0) + 1
            usage = resp["usage"]
            outcome = "succeeded"
            spend = cost(usage["inputTokens"], usage["outputTokens"])
        except ClientError as err:
            attempts = err.response["ResponseMetadata"].get("RetryAttempts", 0) + 1
            outcome = f"gave up ({err.response['Error']['Code']})"
            spend = 0.0
        elapsed = time.time() - t0
        rows.append((max_retries, attempts, outcome, elapsed, spend))
        RESULTS.append(("SDK retries", f"max retries={max_retries}", attempts, elapsed, spend, outcome))
        print(
            f"  max retries={max_retries}: {outcome} after {attempts} attempt(s) "
            f"in {elapsed:.1f}s, billed ${spend:.6f}"
        )

    print(f"\n{'max retries':>11} {'attempts':>9} {'seconds':>8} {'billed':>10}  outcome")
    for max_retries, attempts, outcome, elapsed, spend in rows:
        print(f"{max_retries:>11} {attempts:>9} {elapsed:>8.1f} ${spend:>9.6f}  {outcome}")
    print(
        "\n  Throttled attempts are not billed, so the cost of retrying is latency and"
        "\n  load, not tokens. A low cap fails fast; a high cap rides out a longer outage"
        "\n  but each extra retry backs off longer. Pick the cap from how long your"
        "\n  transient failures really last, and never leave it unbounded.\n"
    )


# ============================================================
# 2. Application retry cap
# ============================================================


def validate_order_reply(text: str) -> bool:
    """A mis-specified validator: it requires a key the prompt never asked for.

    The prompt asks for "order_id"; this check demands "orderId". The model
    follows the prompt, so EVERY attempt fails - a bug retries cannot fix.
    """
    # Models often wrap JSON in a markdown fence; strip it before parsing.
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return "orderId" in json.loads(cleaned)
    except json.JSONDecodeError:
        return False


def demo_app_retry_cap() -> None:
    """A retry loop around a call that always fails validation: the cap bounds the waste."""
    print("--- 2. Application retry cap: every retry is billed in full ---")
    runtime = boto3.client("bedrock-runtime", region_name=REGION)
    prompt = (
        "Extract the order number from the message below. Reply with only JSON in "
        'the form {"order_id": "<value>"} and nothing else.\n\n'
        f"Message: {ORDER_MESSAGE}"
    )
    print(f"  MAX_APP_RETRIES = {MAX_APP_RETRIES}\n")

    total_in = total_out = 0
    first_attempt_cost = 0.0
    valid = False
    for attempt in range(1, MAX_APP_RETRIES + 1):
        resp = runtime.converse(
            modelId=HAIKU,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 100},
        )
        text = resp["output"]["message"]["content"][0]["text"].strip()
        usage = resp["usage"]
        total_in += usage["inputTokens"]
        total_out += usage["outputTokens"]
        attempt_cost = cost(usage["inputTokens"], usage["outputTokens"])
        if attempt == 1:
            first_attempt_cost = attempt_cost
        valid = validate_order_reply(text)
        print(
            f"  attempt {attempt}: valid={valid}  in={usage['inputTokens']} "
            f"out={usage['outputTokens']}  cost=${attempt_cost:.6f}  reply={text!r}"
        )
        if valid:
            break

    total_cost = cost(total_in, total_out)
    RESULTS.append(
        (
            "App retries",
            f"cap={MAX_APP_RETRIES}",
            attempt,
            None,
            total_cost,
            "valid" if valid else "cap reached, still invalid",
        )
    )
    if valid:
        print("\n  Validation passed.")
    else:
        print(
            f"\n  Cap reached after {MAX_APP_RETRIES} attempts: spent ${total_cost:.6f} "
            f"(~{MAX_APP_RETRIES}x the ${first_attempt_cost:.6f} of one call) and still no valid answer."
        )
        print("  Fallback: log the failure, return a safe default, or route to a person.")
    print(
        "\n  The validator here is mis-specified, so no number of retries would help."
        "\n  Without a cap this loop never ends and the bill grows linearly. Retry only"
        "\n  transient failures, and treat repeated validation failure as a bug to fix.\n"
    )


# ============================================================
# 3. Turn and tool-call caps
# ============================================================

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "check_order_status",
                "description": "Look up the current delivery status of an order.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    }
                },
            }
        }
    ]
}

LOOP_SYSTEM = (
    "You are an order-status assistant. Use check_order_status to look up the "
    "order. Do not answer the customer until the status is DELIVERED; until then, "
    "keep checking."
)


def check_order_status(order_id: str) -> dict:
    """A tool that never reaches DELIVERED - stands in for a flaky or stuck backend."""
    return {"order_id": order_id, "status": "PROCESSING", "note": "Not delivered yet."}


def run_agent_loop(runtime, max_turns: int, max_tool_calls: int) -> dict:
    """Run a tool-use loop bounded by max_turns and max_tool_calls.

    One turn is one model call. The whole conversation is resent each turn, so
    input tokens grow as the history grows.
    """
    messages = [{"role": "user", "content": [{"text": ORDER_MESSAGE}]}]
    turns = tool_calls = total_in = total_out = 0
    per_turn_input = []
    stop = None

    while True:
        # Cap 1: turns. Checked BEFORE calling the model, so a capped loop pays nothing more.
        if turns >= max_turns:
            stop = "max_turns"
            break

        resp = runtime.converse(
            modelId=HAIKU,
            system=[{"text": LOOP_SYSTEM}],
            messages=messages,
            toolConfig=TOOL_CONFIG,
            inferenceConfig={"maxTokens": 300},
        )
        turns += 1
        usage = resp["usage"]
        total_in += usage["inputTokens"]
        total_out += usage["outputTokens"]
        per_turn_input.append(usage["inputTokens"])
        messages.append(resp["output"]["message"])

        # Always inspect stopReason: anything other than tool_use ends the loop.
        if resp["stopReason"] != "tool_use":
            stop = resp["stopReason"]
            break

        uses = [b["toolUse"] for b in resp["output"]["message"]["content"] if "toolUse" in b]

        # Cap 2: tool calls. A turn may request several tools at once.
        if tool_calls + len(uses) > max_tool_calls:
            stop = "max_tool_calls"
            break

        results = []
        for use in uses:
            tool_calls += 1
            result = check_order_status(use["input"].get("order_id", ""))
            results.append(
                {"toolResult": {"toolUseId": use["toolUseId"], "content": [{"json": result}]}}
            )
        messages.append({"role": "user", "content": results})

    return {
        "turns": turns,
        "tool_calls": tool_calls,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "per_turn_input": per_turn_input,
        "stop": stop,
        "cost": cost(total_in, total_out),
    }


def demo_turn_and_tool_caps() -> None:
    """Same stuck agent under tight and loose caps: the cap bounds spend."""
    print("--- 3. Turn and tool-call caps: a stuck agent loop ---")
    print(f"  USER: {ORDER_MESSAGE}")
    print("  The tool always answers PROCESSING, so the agent never gets DELIVERED.\n")

    runtime = boto3.client("bedrock-runtime", region_name=REGION)
    rows = []
    for label, max_turns, max_tool_calls in LOOP_CAPS:
        r = run_agent_loop(runtime, max_turns, max_tool_calls)
        rows.append((label, max_turns, max_tool_calls, r))
        RESULTS.append(
            (
                "Tool loop",
                f"{label} ({max_turns} turns, {max_tool_calls} tools)",
                r["turns"],
                None,
                r["cost"],
                f"stopped: {r['stop']}",
            )
        )
        print(f"  {label} (max_turns={max_turns}, max_tool_calls={max_tool_calls})")
        print(f"    input tokens per turn: {r['per_turn_input']}")
        print(f"    stopped because: {r['stop']}\n")

    print(
        f"{'caps':<12} {'turns':>6} {'tools':>6} {'in tok':>8} {'out tok':>8} {'cost':>10}  stop"
    )
    for label, _, _, r in rows:
        print(
            f"{label:<12} {r['turns']:>6} {r['tool_calls']:>6} {r['input_tokens']:>8} "
            f"{r['output_tokens']:>8} ${r['cost']:>9.6f}  {r['stop']}"
        )
    print(
        "\n  Input tokens per turn climb because each turn resends the whole history, so"
        "\n  total spend grows faster than the turn count. Size the caps from the turns"
        "\n  your successful runs actually need (plus headroom), and give a capped run a"
        "\n  graceful fallback such as escalating to a person.\n"
    )


# ============================================================
# Comparison table
# ============================================================


def print_comparison_table() -> None:
    """Print every cap that was tried, side by side."""
    print("--- Comparison: every cap tried ---")
    print(f"{'Lever':<12} {'Setting':<36} {'Tries/turns':>11} {'Seconds':>8} {'Billed $':>10}  Outcome")
    last = None
    for lever, setting, count, seconds, billed, outcome in RESULTS:
        if last is not None and lever != last:
            print()
        last = lever
        sec_text = f"{seconds:.1f}" if seconds is not None else "-"
        print(f"{lever:<12} {setting:<36} {count:>11} {sec_text:>8} {billed:>10.6f}  {outcome}")
    print(
        "\n  Tries/turns is attempts for the retry rows and model calls for the tool loop."
        "\n  Throttled SDK attempts bill nothing, so that cap trades latency for"
        "\n  resilience. Application retries and loop turns bill on every pass, so the cap"
        "\n  is what bounds the spend. A dash means timing was not measured for that row.\n"
    )


# ============================================================
# Main
# ============================================================


def main():
    print("Retry, turn and stop limits")
    print(f"  Region                  : {REGION}")
    print(f"  Model                   : {HAIKU}")
    print(f"  SDK max retries compare : {SDK_MAX_RETRIES}")
    print(f"  Simulated throttled     : first {SIMULATED_THROTTLED_ATTEMPTS} attempts")
    print(f"  App retry cap           : {MAX_APP_RETRIES}")
    print(f"  Loop caps               : {LOOP_CAPS}\n")

    demo_sdk_retry_cap()
    demo_app_retry_cap()
    demo_turn_and_tool_caps()
    print_comparison_table()

    print("--- Done ---")
    print("  Next steps:")
    print("  1. Set the SDK retry cap from how long transient failures really last")
    print("  2. Retry only transient failures; fix validators and prompts that fail every time")
    print("  3. Cap turns and tool calls from your task's success rate, with a fallback")


if __name__ == "__main__":
    main()
