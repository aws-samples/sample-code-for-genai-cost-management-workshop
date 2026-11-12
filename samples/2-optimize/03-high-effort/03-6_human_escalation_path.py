"""
High-Effort Optimization - Lever 18: Human Escalation Path

An agent that cannot finish a task keeps looping, and every extra turn re-sends
the whole growing context. A human escalation path puts a ceiling on that waste:
after N turns without an answer, the harness stops the agent and hands the case
to a person.

The key is choosing N from evidence, not guessing:
  1. Run your eval set with a generous turn limit.
  2. Look at how many turns the tasks the agent DOES solve actually need.
  3. Set N to that number. Too low and you escalate tasks the agent could have
     solved. Too high and you pay for turns that never converge.

This sample is a small order-lookup agent. The order list is paged, one order
per call, so finding an order takes one turn per order. Orders near the top are
found quickly, and orders that are not in the list keep the agent paging.

You will learn how to:
- Stop an agent loop at N turns and hand off instead of failing
- Derive N from eval results
- Compare turns, tokens, and cost with and without the escalation path

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
INPUT_PRICE, OUTPUT_PRICE = 1.00, 5.00

# The generous limit used to measure the agent on the eval set.
EVAL_MAX_TURNS = 8


# ============================================================
# The tool: one page of the order list per call
# ============================================================

ORDERS = [
    "Wireless Headphones", "Mechanical Keyboard", "Phone Charger", "USB Hub",
    "Garden Hose", "Bike Light", "Cutting Board", "Phone Case", "Throw Blanket",
    "Plant Pot",
]


def list_orders(cursor: str = "0") -> str:
    """Return one order and the cursor for the next one."""
    i = int(cursor or 0)
    if i >= len(ORDERS):
        return "No more orders."
    return f"Order: {ORDERS[i]} - shipped, tracking TRK{100 + i}. next_cursor: {i + 1}"


TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "list_orders",
                "description": (
                    "Return one order from the customer's order list (item, status, "
                    "tracking code) and a next_cursor. Start with cursor 0, then pass "
                    "the next_cursor from the previous result."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"cursor": {"type": "string"}},
                    }
                },
            }
        }
    ]
}

SYSTEM_PROMPT = (
    "You are a concise customer-support agent. Find the customer's order with the "
    "list_orders tool. The list is long: keep fetching the next order until you "
    "find it or no orders remain, and do not ask the customer for more details "
    "before then. Reply with the tracking code once you find it."
)


# ============================================================
# The eval set
# ============================================================
#
# "expect" is the tracking code a correct answer contains. Tasks with expect
# None ask about an item that is not in the list, so the agent can never solve
# them - the right outcome for those is a handoff to a person.

EVAL_CASES = [
    {"task": "Where is my Wireless Headphones order?", "expect": "TRK100"},
    {"task": "Where is my Mechanical Keyboard order?", "expect": "TRK101"},
    {"task": "Where is my Phone Charger order?", "expect": "TRK102"},
    {"task": "Where is my Desk Lamp order?", "expect": None},
    {"task": "Where is my Standing Desk order?", "expect": None},
]


# ============================================================
# The agent loop, with a turn limit that escalates
# ============================================================

def run_agent(task: str, max_turns: int) -> dict:
    """Call the model, run the tool, repeat. Escalate if max_turns is reached."""
    messages = [{"role": "user", "content": [{"text": task}]}]
    input_tokens = output_tokens = lookups = 0

    for turn in range(1, max_turns + 1):
        resp = RUNTIME.converse(
            modelId=HAIKU,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig=TOOL_CONFIG,
            inferenceConfig={"maxTokens": 512, "temperature": 0},
        )
        input_tokens += resp["usage"]["inputTokens"]
        output_tokens += resp["usage"]["outputTokens"]
        message = resp["output"]["message"]
        messages.append(message)

        result = {"turns": turn, "input_tokens": input_tokens,
                  "output_tokens": output_tokens, "lookups": lookups}

        if resp["stopReason"] != "tool_use":  # the agent answered
            text = next((b["text"] for b in message["content"] if "text" in b), "")
            return {**result, "text": text, "escalated": False}

        if turn == max_turns:  # still asking for tools at the limit: escalate
            return {**result, "text": "", "escalated": True}

        tool_results = []
        for block in message["content"]:
            if "toolUse" in block:
                use = block["toolUse"]
                lookups += 1
                tool_results.append({"toolResult": {
                    "toolUseId": use["toolUseId"],
                    "content": [{"text": list_orders(**use["input"])}],
                }})
        messages.append({"role": "user", "content": tool_results})


# ============================================================
# Helpers
# ============================================================

def cost(result: dict) -> float:
    return (result["input_tokens"] * INPUT_PRICE + result["output_tokens"] * OUTPUT_PRICE) / 1_000_000


def run_eval_set(max_turns: int, escalation: bool) -> list[dict]:
    """Run every eval case with the given turn limit and mark which were resolved."""
    results = []
    for case in EVAL_CASES:
        r = run_agent(case["task"], max_turns)
        r["resolved"] = (
            not r["escalated"] and case["expect"] is not None and case["expect"] in r["text"]
        )
        print(f"  {case['task']}")
        print(f"      -> {r['turns']} turns, {outcome(r, escalation)}")
        results.append(r)
    return results


def outcome(r: dict, escalation: bool) -> str:
    if r["resolved"]:
        return "resolved"
    if r["escalated"]:
        return (
            "still searching, handed to a human"
            if escalation
            else "still searching when the limit stopped it, customer gets no answer"
        )
    return "answered without resolving"


def print_intro() -> None:
    print("--- Human Escalation Path: escalate after N turns, set N from eval results ---")
    print(f"  Model: {HAIKU}")
    print(f"  Eval set: {len(EVAL_CASES)} tasks")
    solvable = sum(c["expect"] is not None for c in EVAL_CASES)
    unsolvable = len(EVAL_CASES) - solvable
    print(f"""
BACKGROUND
  A customer-support agent helps customers find their orders. It has one tool,
  list_orders, which returns ONE order per call (item, status, tracking code)
  plus a cursor for the next one. The agent keeps calling the tool until it
  finds the order the customer asked about, then replies with the tracking code.

  A "turn" is one model call. After each tool call the whole conversation so far
  is sent to the model again, so every turn costs more than the one before it.

THE PROBLEM
  If the order is near the top of the list, the agent finds it in a few turns.
  If the order is NOT in the list (a wrong item name, an order under another
  account), the agent keeps paging until something stops it. It burns tokens on
  every turn and the customer never gets an answer.

THE FIX
  Hand the case to a human once the agent has used N turns without an answer.
  The hard part is choosing N:
    - Too low: you escalate tasks the agent could have solved.
    - Too high: you pay for turns that were never going to find the order.
  So we do not guess. We measure how many turns the solvable tasks really need
  and set N from that.

WHAT THIS DEMO DOES
  Step 1  Run {len(EVAL_CASES)} test tasks ({solvable} the agent can solve, {unsolvable} it cannot) with a
          generous turn limit, and record how many turns each one takes.
  Step 2  Set N to the most turns any SOLVED task needed.
  Step 3  Run the same tasks again, handing off to a human after N turns.
  Then compare turns, tokens, and cost with and without the escalation path.
""")


def print_divider() -> None:
    print("\n" + "-" * 70 + "\n")


def print_comparison_table(baseline: list[dict], escalated: list[dict], n: int) -> None:
    rows = [("No escalation", baseline), (f"Escalate after {n} turns", escalated)]
    base_cost = sum(cost(r) for r in baseline)

    print("--- Comparison: same eval set, with and without an escalation path ---")
    print(f"{'Policy':<24} {'Turns':>6} {'Input tokens':>13} {'Output tokens':>14} "
          f"{'Cost $':>9} {'Relative cost':>14} {'Resolved':>9} {'Hit turn limit':>15}")
    for label, results in rows:
        total_cost = sum(cost(r) for r in results)
        print(f"{label:<24} {sum(r['turns'] for r in results):>6} "
              f"{sum(r['input_tokens'] for r in results):>13} "
              f"{sum(r['output_tokens'] for r in results):>14} "
              f"{total_cost:>9.5f} {total_cost / base_cost:>13.2f}x "
              f"{sum(r['resolved'] for r in results):>9} "
              f"{sum(r['escalated'] for r in results):>15}")
    print(
        "\n  Turns = total model calls across all tasks. Hit turn limit = tasks that were"
        "\n  stopped because the agent was still searching."
        "\n"
        "\n  Both policies resolve the same tasks. The saving comes from the tasks the"
        "\n  agent cannot solve: they stop after N turns instead of running to the"
        "\n  limit, and a person gets the case instead of the customer getting nothing."
    )


# ============================================================
# Demo
# ============================================================

def main():
    print_intro()

    print_divider()
    print(f"Step 1 of 3: run the test tasks with a generous limit ({EVAL_MAX_TURNS} turns)\n")
    print("  The limit is high on purpose, so every task the agent can solve gets the")
    print("  chance to finish. This shows how many turns the solvable tasks really need.\n")
    baseline = run_eval_set(EVAL_MAX_TURNS, escalation=False)
    print("\n  Why did some tasks use all the turns? The Desk Lamp and Standing Desk are not")
    print("  in the customer's order list. The agent cannot know that without checking")
    print("  every order, so it kept paging through the list until the limit stopped it.")
    print("  It never found the order and never gave the customer an answer.")

    print_divider()
    print("Step 2 of 3: set N from the results\n")
    resolved_turns = sorted(r["turns"] for r in baseline if r["resolved"])
    n = max(resolved_turns, default=EVAL_MAX_TURNS)
    print(f"  Turns needed by the tasks the agent solved: {resolved_turns}")
    print(f"  N = {n}, the most turns any solved task needed.")
    print("  A task still going after that many turns is unlikely to be solved, so")
    print("  it is cheaper to hand it to a person than to keep paying for more turns.")

    print_divider()
    print(f"Step 3 of 3: run the same tasks, handing off to a human after {n} turns\n")
    print("  Solved tasks finish exactly as before. Tasks that are still asking for")
    print("  tools at the limit are stopped and handed over.\n")
    escalated = run_eval_set(n, escalation=True)

    print_divider()
    print("Results, task by task\n")
    for case, base, esc in zip(EVAL_CASES, baseline, escalated):
        print(f"Task: {case['task']}")
        print(f"  No escalation : {base['turns']} turns | ${cost(base):.5f} | {outcome(base, False)}")
        print(f"  Escalate at {n} : {esc['turns']} turns | ${cost(esc):.5f} | {outcome(esc, True)}")
        if esc["escalated"]:
            print(f"  Note for human: customer asked '{case['task']}'; the agent made "
                  f"{esc['lookups']} lookups and did not find the order.")
        print()

    print_divider()
    print_comparison_table(baseline, escalated, n)

    print("\n--- Done ---")
    print("  Next steps:")
    print("  1. Measure turns-to-resolution on your own eval set before choosing N")
    print("  2. Re-measure after prompt, tool, or model changes - N moves with them")
    print("  3. Track the escalation rate: a rising rate means the agent, not N, needs work")


if __name__ == "__main__":
    main()
