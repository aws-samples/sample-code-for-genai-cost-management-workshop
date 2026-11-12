"""
High-Effort Optimization - Lever 18: Human Escalation Path

An agent loop that cannot finish a task keeps going. Every extra turn re-sends
the whole growing context, so a stuck conversation is the most expensive kind:
it pays more per turn than a healthy one, and it may never produce an answer.

A human escalation path puts a ceiling on that waste. After N turns without a
resolution, the harness stops the agent and hands the case to a person, with a
short note of what was already tried. Two things make this work:

  - N is set from evidence, not guessed. Run your eval set, look at how many
    turns the tasks the agent DOES resolve actually need, and set N just above
    that. Too low and you escalate tasks the agent could have solved. Too high
    and you pay for turns that never converge.
  - The handoff is built by code. The harness writes the note from its own log
    of tool calls, so escalating costs zero extra model tokens.

This sample runs a small customer-support agent (boto3 Converse tool loop) over
an eval set. The order history is paged, so a lookup takes one turn per page:
orders near the top are found in a couple of turns, orders further down take
more, and orders that are not in the history at all keep the agent paging.

You will learn how to:
- Build a tool loop that stops at a turn limit and hands off instead of failing
- Derive the escalation threshold N from eval results
- Compare turns, tokens, and cost with and without the escalation path

NOTE: A hard turn cap (see the retry and turn limits sample) only stops the
loop. An escalation path also routes the case to someone who can finish it.

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude Haiku 4.5 on Amazon Bedrock
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

# Pricing per 1M tokens (USD). Illustrative workshop values - always verify
# current pricing at https://aws.amazon.com/bedrock/pricing/
PRICING = {HAIKU: {"input": 1.00, "output": 5.00}}

# The generous cap used while measuring the agent on the eval set. Think of it
# as "no real limit": high enough that every task the agent can solve finishes.
EVAL_MAX_TURNS = 8


# ============================================================
# Tools (a support backend with a paged order history)
# ============================================================

# The customer's order history, newest first, 2 orders per page. Pages 1-4
# hold the orders the eval set asks about; the rest is filler history.
_HEAD_ORDERS = [
    ("12345", "Wireless Headphones", "In transit", "2 business days"),
    ("12346", "Notebook Set", "Delivered", "delivered"),
    ("67890", "Mechanical Keyboard", "Delivered", "delivered"),
    ("67891", "Water Bottle", "Delivered", "delivered"),
    ("55555", "Phone Charger", "Shipped", "1 business day"),
    ("55556", "Yoga Mat", "Delivered", "delivered"),
    ("44444", "USB Hub", "Out for delivery", "today"),
    ("44445", "Coffee Grinder", "Delivered", "delivered"),
]
_FILLER_ITEMS = [
    "Garden Hose", "Bike Light", "Cutting Board", "Phone Case", "Throw Blanket",
    "Plant Pot", "Desk Organizer", "Face Towel", "Wall Clock", "Mouse Pad",
    "Candle Set", "Lunch Box", "Picture Frame", "Shower Curtain", "Door Mat",
    "Key Holder", "Tea Kettle", "Pillow Case", "Coaster Set", "Shoe Rack",
]
ORDER_HISTORY = list(_HEAD_ORDERS) + [
    (str(30000 + i), item, "Delivered", "delivered") for i, item in enumerate(_FILLER_ITEMS)
]
PAGE_SIZE = 2
TOTAL_PAGES = len(ORDER_HISTORY) // PAGE_SIZE

# Opaque cursors, so the model can only move forward one page at a time and
# cannot guess page numbers (a real paged API works the same way).
_CURSORS = [None] + [f"cur-{(i * 7919) % 9973:04d}" for i in range(1, TOTAL_PAGES)]

_RETURN_POLICY = {
    "audio": "45-day return window; opened items returnable only if defective.",
    "accessories": "60-day return window; opened items accepted.",
}


def list_orders(cursor: str = "") -> str:
    """Return one page of the order history plus the cursor for the next page."""
    cursor = (cursor or "").strip()
    if cursor and cursor not in _CURSORS:
        return f"Invalid cursor '{cursor}'."
    page = _CURSORS.index(cursor) if cursor else 0
    start = page * PAGE_SIZE
    lines = [
        f"#{oid} {item} - {status} (ETA: {eta})"
        for oid, item, status, eta in ORDER_HISTORY[start:start + PAGE_SIZE]
    ]
    if page + 1 < TOTAL_PAGES:
        return "Orders: " + "; ".join(lines) + f". next_cursor: {_CURSORS[page + 1]}"
    return "Orders: " + "; ".join(lines) + ". No more pages."


def get_return_policy(category: str) -> str:
    return _RETURN_POLICY.get(category.strip().lower(), "30-day standard return window.")


TOOL_IMPLS = {"list_orders": list_orders, "get_return_policy": get_return_policy}

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "list_orders",
                "description": (
                    "List the customer's orders, newest first, 2 per page, with item, "
                    "status, and ETA. Omit cursor for the first page; pass the "
                    "next_cursor from the previous page to get the following page."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"cursor": {"type": "string"}},
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "get_return_policy",
                "description": "Get the return policy for a product category (e.g. audio, accessories).",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"category": {"type": "string"}},
                        "required": ["category"],
                    }
                },
            }
        },
    ]
}

SYSTEM_PROMPT = (
    "You are a concise customer-support agent. Use your tools to look up real "
    "order and policy information before answering. The order history is paged "
    "and can be long: if the order is not on the current page, keep fetching the "
    "next page with next_cursor until you find it or no pages remain. Do not ask "
    "the customer for more details until the whole history has been searched. "
    "When you have the answer, reply directly without calling more tools."
)


# ============================================================
# The eval set
# ============================================================
#
# "expect" is text that only appears in a correct answer. Tasks with expect
# None ask about an order that is not in the history, so the agent can never
# resolve them - the right outcome for those is a handoff to a person.

EVAL_CASES = [
    {"task": "What is the return policy for audio?", "expect": "45"},
    {"task": "What is the return policy for accessories?", "expect": "60"},
    {"task": "When will my Wireless Headphones order arrive?", "expect": "2 business days"},
    {"task": "Has my Mechanical Keyboard order been delivered?", "expect": "Delivered"},
    {"task": "What is the status of my Phone Charger order?", "expect": "Shipped"},
    {"task": "What is the status of my USB Hub order?", "expect": "Out for delivery"},
    {"task": "What is the status of my Desk Lamp order?", "expect": None},
    {"task": "What is the status of my Standing Desk order?", "expect": None},
]


# ============================================================
# The control loop with an escalation path
# ============================================================

def run_agent(user_message: str, max_turns: int) -> dict:
    """Call -> parse -> run tools -> append -> check stop, with a turn limit.

    If the agent is still asking for tools on its last allowed turn, the harness
    does not run them. It stops and marks the task as escalated, and the caller
    builds the handoff from tool_log (no extra model call).
    """
    messages = [{"role": "user", "content": [{"text": user_message}]}]
    total_input = total_output = 0
    tool_log: list[dict] = []

    for turn in range(1, max_turns + 1):
        resp = RUNTIME.converse(
            modelId=HAIKU,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig=TOOL_CONFIG,
            inferenceConfig={"maxTokens": 512, "temperature": 0},
        )
        total_input += resp["usage"]["inputTokens"]
        total_output += resp["usage"]["outputTokens"]

        out_message = resp["output"]["message"]
        messages.append(out_message)  # keep the assistant turn in context

        if resp.get("stopReason") != "tool_use":
            text = next((b["text"] for b in out_message["content"] if "text" in b), "")
            return {"text": text, "turns": turn, "input_tokens": total_input,
                    "output_tokens": total_output, "escalated": False, "tool_log": tool_log}

        if turn == max_turns:
            break  # turn limit reached while the agent still wants tools

        # Run every requested tool and feed the results back as one user turn.
        tool_results = []
        for block in out_message["content"]:
            if "toolUse" not in block:
                continue
            tu = block["toolUse"]
            impl = TOOL_IMPLS.get(tu["name"])
            result = impl(**tu["input"]) if impl else f"Unknown tool: {tu['name']}"
            tool_log.append({"tool": tu["name"], "input": tu["input"], "result": result})
            tool_results.append(
                {"toolResult": {"toolUseId": tu["toolUseId"],
                                "content": [{"text": result}]}}
            )
        messages.append({"role": "user", "content": tool_results})

    return {"text": "", "turns": max_turns, "input_tokens": total_input,
            "output_tokens": total_output, "escalated": True, "tool_log": tool_log}


def build_handoff(task: str, result: dict) -> str:
    """Write the note for the human from the harness log. Pure code, zero tokens."""
    log = result["tool_log"]
    last = log[-1] if log else None
    last_line = f"{last['tool']}({last['input']})" if last else "none"
    return (
        f"Customer asked: {task} | agent used {result['turns']} turns and "
        f"{len(log)} tool calls without an answer | last tool call: {last_line}"
    )


# ============================================================
# Helpers
# ============================================================

def is_resolved(case: dict, result: dict) -> bool:
    """A task is resolved if the agent finished with an answer containing the expected text."""
    if result["escalated"] or case["expect"] is None:
        return False
    return case["expect"].lower() in result["text"].lower()


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    prices = PRICING[HAIKU]
    return (input_tokens / 1_000_000) * prices["input"] + (
        output_tokens / 1_000_000
    ) * prices["output"]


def run_eval_set(max_turns: int, label: str) -> list[dict]:
    """Run every eval case with the given turn limit and score it."""
    results = []
    for i, case in enumerate(EVAL_CASES, start=1):
        print(f"  [{label}] case {i} of {len(EVAL_CASES)}: {case['task']}")
        r = run_agent(case["task"], max_turns)
        r["resolved"] = is_resolved(case, r)
        results.append(r)
    return results


def derive_escalation_threshold(results: list[dict]) -> int:
    """Pick N from eval results: the most turns any RESOLVED task needed."""
    resolved_turns = sorted(r["turns"] for r in results if r["resolved"])
    print(f"  Turns taken by each task the agent resolved: {resolved_turns}")
    if not resolved_turns:
        return EVAL_MAX_TURNS
    return max(resolved_turns)


def print_divider() -> None:
    print("\n" + "-" * 70 + "\n")


def outcome(r: dict, escalation: bool) -> str:
    if r["resolved"]:
        return "resolved"
    if r["escalated"]:
        return "handed to a human" if escalation else "stopped at the hard cap, no answer"
    return "answered without resolving"


def print_comparison_table(baseline: list[dict], escalated: list[dict], n: int) -> None:
    rows = [("No escalation", baseline), (f"Escalate after {n} turns", escalated)]
    base_cost = sum(calculate_cost(r["input_tokens"], r["output_tokens"]) for r in baseline)

    print("--- Comparison: same eval set, with and without an escalation path ---")
    print(f"{'Policy':<22} {'Turns':>6} {'Input tokens':>13} {'Output tokens':>14} "
          f"{'Cost $':>9} {'Relative cost':>14} {'Resolved':>9} {'Hit turn limit':>15}")
    for label, results in rows:
        turns = sum(r["turns"] for r in results)
        in_tok = sum(r["input_tokens"] for r in results)
        out_tok = sum(r["output_tokens"] for r in results)
        cost = calculate_cost(in_tok, out_tok)
        resolved = sum(1 for r in results if r["resolved"])
        stopped = sum(1 for r in results if r["escalated"])
        print(f"{label:<22} {turns:>6} {in_tok:>13} {out_tok:>14} "
              f"{cost:>9.5f} {cost / base_cost:>13.2f}x {resolved:>9} {stopped:>15}")
    print(
        f"\n  Both policies resolve the same tasks. The saving comes from the tasks the"
        f"\n  agent cannot solve: they stop after {n} turns instead of running on, and a"
        f"\n  person gets a handoff note instead of the customer getting an error."
        f"\n  Under the hard cap, 'Hit turn limit' tasks end with no answer; under the"
        f"\n  escalation path they go to a human."
    )


# ============================================================
# Demo
# ============================================================

def main():
    print("--- Human Escalation Path: escalate after N turns, set N from eval results ---")
    print(f"  Model: {HAIKU}")
    print(f"  Eval set: {len(EVAL_CASES)} tasks | hard cap while measuring: {EVAL_MAX_TURNS} turns")

    print_divider()
    print("Step 1 of 4: run the eval set with a generous turn cap\n")
    baseline = run_eval_set(EVAL_MAX_TURNS, "no escalation")

    print_divider()
    print("Step 2 of 4: set N from the eval results\n")
    n = derive_escalation_threshold(baseline)
    print(f"  Escalation threshold N = {n} turns "
          f"(the slowest resolved task needed {n}; anything past that is not converging)")

    print_divider()
    print(f"Step 3 of 4: run the eval set again, escalating to a human after {n} turns\n")
    escalated = run_eval_set(n, f"escalate after {n}")

    print_divider()
    print("Step 4 of 4: compare, task by task\n")
    for case, base, esc in zip(EVAL_CASES, baseline, escalated):
        base_cost = calculate_cost(base["input_tokens"], base["output_tokens"])
        esc_cost = calculate_cost(esc["input_tokens"], esc["output_tokens"])
        print(f"Task: {case['task']}")
        print(f"  No escalation  : {base['turns']} turns | "
              f"{base['input_tokens'] + base['output_tokens']:>5} tokens | "
              f"${base_cost:.5f} | {outcome(base, False)}")
        print(f"  Escalate at {n:<3}: {esc['turns']} turns | "
              f"{esc['input_tokens'] + esc['output_tokens']:>5} tokens | "
              f"${esc_cost:.5f} | {outcome(esc, True)}")
        if esc["escalated"]:
            print(f"  Handoff note   : {build_handoff(case['task'], esc)}")
        print()

    print_divider()
    print_comparison_table(baseline, escalated, n)

    print("\n--- Done ---")
    print("  Next steps:")
    print("  1. Measure turns-to-resolution on your own eval set before choosing N")
    print("  2. Re-measure after prompt, tool, or model changes - N moves with them")
    print("  3. Build the handoff note from your harness log so escalation costs no tokens")
    print("  4. Track the escalation rate: a rising rate means the agent, not N, needs work")


if __name__ == "__main__":
    main()
