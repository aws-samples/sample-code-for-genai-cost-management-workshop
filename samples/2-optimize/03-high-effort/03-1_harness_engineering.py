"""
High-Effort Optimization - Lever 13: Harness Engineering

The naive view is "the model is the agent." The current view: the model is one
component; the harness is the deterministic scaffolding around it - tools,
context curation, the control loop, retries, a turn budget, guardrails. Engineer
it and the model's average behavior tracks much closer to its peak.

An agent is just "an LLM using tools in a loop": call the model, parse output,
run tools, append results, check a stop condition, repeat. The loop is where
cost and latency live, because two numbers dominate the bill and p95 latency:

  - Tokens per turn - you re-send the growing context every turn, so lean context
    pays off on turn 1 AND turn 8.
  - Turns per task - each turn is a full round-trip; fewer turns to the answer.

You will learn how to:
- Build a minimal but real agent harness with boto3 Converse (call/parse/tool loop)
- Enforce a turn budget so a runaway loop can't quietly multiply your bill
- See the token cost of a LEAN vs a BLOATED context on the same task

This sample uses a small customer-support tool set. The harness is the point -
the same loop scaffolding applies to any tools.

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude Haiku 4.5 on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import json
import boto3

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

RUNTIME = boto3.client("bedrock-runtime", region_name=REGION)

HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# A turn budget is a harness guardrail: it caps how many model round-trips a
# single task may take, so a loop that fails to converge can't 10x your bill.
MAX_TURNS = 5


# ============================================================
# Tools (the deterministic side of the harness)
# ============================================================
#
# A tool description is a prompt the model reads to decide whether to call the
# tool. Name the verb, list inputs in order, state what comes back. Keep the set
# minimal and non-overlapping - every description is re-sent every turn.

_ORDERS = {
    "12345": {"item": "Wireless Headphones", "status": "In transit", "eta": "2 business days"},
    "67890": {"item": "Mechanical Keyboard", "status": "Delivered", "eta": "delivered"},
}
_RETURN_POLICY = {
    "audio": "45-day return window; opened items returnable only if defective.",
    "accessories": "60-day return window; opened items accepted.",
}


def get_order_status(order_id: str) -> str:
    o = _ORDERS.get(order_id.strip().lstrip("#"))
    if not o:
        return f"No order found for '{order_id}'."
    return f"Order #{order_id}: {o['item']} - {o['status']} (ETA: {o['eta']})."


def get_return_policy(category: str) -> str:
    return _RETURN_POLICY.get(category.strip().lower(), "30-day standard return window.")


TOOL_IMPLS = {"get_order_status": get_order_status, "get_return_policy": get_return_policy}

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_order_status",
                "description": "Look up the status, item, and ETA of an order by its order number.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
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


# ============================================================
# The control loop
# ============================================================

def run_agent(user_message: str, system_prompt: str, max_turns: int = MAX_TURNS) -> dict:
    """A minimal harness: call -> parse -> run tools -> append -> check stop.

    Returns the final text, the number of turns taken, and the cumulative input
    and output tokens (input tokens grow each turn as the context is re-sent).
    """
    messages = [{"role": "user", "content": [{"text": user_message}]}]
    total_input = total_output = 0

    for turn in range(1, max_turns + 1):
        resp = RUNTIME.converse(
            modelId=HAIKU,
            system=[{"text": system_prompt}],
            messages=messages,
            toolConfig=TOOL_CONFIG,
            inferenceConfig={"maxTokens": 512, "temperature": 0},
        )
        total_input += resp["usage"]["inputTokens"]
        total_output += resp["usage"]["outputTokens"]

        out_message = resp["output"]["message"]
        messages.append(out_message)  # keep the assistant turn in context

        if resp.get("stopReason") != "tool_use":
            # No tool requested - the agent is done.
            text = next(
                (b["text"] for b in out_message["content"] if "text" in b), ""
            )
            return {"text": text, "turns": turn, "input_tokens": total_input,
                    "output_tokens": total_output}

        # Run every requested tool and feed the results back as one user turn.
        tool_results = []
        for block in out_message["content"]:
            if "toolUse" not in block:
                continue
            tu = block["toolUse"]
            impl = TOOL_IMPLS.get(tu["name"])
            result = impl(**tu["input"]) if impl else f"Unknown tool: {tu['name']}"
            tool_results.append(
                {"toolResult": {"toolUseId": tu["toolUseId"],
                                "content": [{"text": result}]}}
            )
        messages.append({"role": "user", "content": tool_results})

    # Turn budget exhausted - the harness stops rather than looping forever.
    return {"text": "[stopped: turn budget exhausted]", "turns": max_turns,
            "input_tokens": total_input, "output_tokens": total_output}


# ============================================================
# Demo 1: the loop, with a turn budget
# ============================================================

def demo_agent_loop() -> None:
    print("--- 1. A minimal agent harness (call/parse/tool loop + turn budget) ---\n")

    system = (
        "You are a concise customer-support agent. Use your tools to look up real "
        "order and policy information before answering. When you have the answer, "
        "reply directly without calling more tools."
    )
    task = "Where is order #12345, and can I return the headphones if I open them?"

    result = run_agent(task, system)
    print(f"  Task: {task}")
    print(f"  Answer: {result['text'][:240]}")
    print(f"  Turns: {result['turns']}  |  input tokens: {result['input_tokens']}  "
          f"|  output tokens: {result['output_tokens']}")
    print(
        "\n  The turn budget (MAX_TURNS) is a harness guardrail: a loop that fails to"
        "\n  converge stops instead of quietly multiplying your bill.\n"
    )


# ============================================================
# Demo 2: tokens-per-turn (lean vs bloated context)
# ============================================================

def demo_lean_vs_bloated() -> None:
    """The same task with a lean system prompt vs a bloated one - re-sent every turn."""
    print("--- 2. Tokens per turn: lean vs bloated context ---")
    print("  The system prompt is re-sent on EVERY turn, so bloat is paid repeatedly.\n")

    lean_system = "You are a concise support agent. Use tools to look up orders and policies."

    # A bloated system prompt: boilerplate, hedging, and restated instructions -
    # the kind of thing that accretes in production prompts.
    bloated_system = (
        "You are a highly professional, extremely helpful, and always courteous "
        "customer-support assistant working for our wonderful company. It is very "
        "important that you always try your absolute best to help the customer with "
        "whatever they need. Please be sure to always use your available tools to "
        "look up any and all real order information and any and all relevant return "
        "policy information before you attempt to answer any question, because it is "
        "critically important that you never guess or make up any details whatsoever. "
        "Always remember to be polite, always remember to be concise where possible, "
        "always remember to double-check the order number, and always remember to "
        "thank the customer for their patience and for contacting our support team. "
    ) * 3  # accreted boilerplate, repeated

    task = "Where is order #12345?"

    lean = run_agent(task, lean_system)
    bloated = run_agent(task, bloated_system)

    print(f"  {'context':<10} {'turns':>6} {'input tok':>10} {'output tok':>11}")
    print(f"  {'lean':<10} {lean['turns']:>6} {lean['input_tokens']:>10} {lean['output_tokens']:>11}")
    print(f"  {'bloated':<10} {bloated['turns']:>6} {bloated['input_tokens']:>10} {bloated['output_tokens']:>11}")

    extra = bloated["input_tokens"] - lean["input_tokens"]
    print(
        f"\n  The bloated prompt costs {extra} extra input tokens for the same answer -"
        "\n  and that surcharge is paid on every turn of every task. Trimming the system"
        "\n  prompt is the cheapest harness win there is (pair it with caching, 01-4).\n"
    )


# ============================================================
# Main
# ============================================================

def main():
    demo_agent_loop()
    demo_lean_vs_bloated()

    print("--- Done ---")
    print("  The harness, not the model alone, determines cost and reliability. Audit:")
    print("  - Tools: each description names the verb, inputs, and output format")
    print("  - Context: decide what is cached, retrieved, summarized per turn")
    print("  - Control loop: retries + a turn budget so runaway loops can't 10x the bill")
    print("  - Evals + code-level guardrails for anything irreversible")


if __name__ == "__main__":
    main()
