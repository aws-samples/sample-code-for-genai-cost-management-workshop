"""
High-Effort Optimization - Lever 15: Long-Running Agent State

Some agent tasks cannot finish in one sitting. The agent reaches a point where
it needs a person - an approval, a missing document, a decision - and the
answer may take hours. Nothing runs while it waits, so waiting itself costs no
model tokens. The cost shows up when the agent comes back.

There are two ways to come back:

  - REPLAY: reload the whole conversation so far, every tool result included,
    and continue. The prompt is as large as all the work already done, and that
    whole history is re-sent on every remaining turn.
  - RESUME FROM STATE: during the work, the harness saves a small checkpoint
    outside the context (the decisions made so far). Coming back, the agent
    starts a fresh, short context from that checkpoint and does only the
    remaining work.

Saving state outside the context also means the process does not have to stay
alive during the wait: it can shut down and start again when the answer arrives.

This sample is a refund-review agent. It reviews requests one by one, hits a
request it is not allowed to decide alone, asks the user, and waits. It then
finishes the task both ways and compares the cost.

You will learn how to:
- Pause an agent loop on a question for the user and save a checkpoint
- Resume the same task by replaying history and from the checkpoint
- Compare input tokens, turns, and cost of each way of coming back

NOTE: The wait is simulated - the script does not sleep. The checkpoint is a
local JSON file; in production it would live in a store such as Amazon DynamoDB
or Amazon S3, or in Amazon Bedrock AgentCore Memory.

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude Haiku 4.5 on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import copy
import json
import os
import re
import tempfile
import textwrap

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

# The agent may approve refunds up to this amount by itself.
APPROVAL_LIMIT = 200

# How long the simulated wait for the user lasts (the script does not sleep).
WAIT_HOURS = 3

# The answer the user sends back after the wait.
USER_ANSWER = "Approved. Go ahead and approve the full amount."

CHECKPOINT_PATH = os.path.join(tempfile.gettempdir(), "refund_review_checkpoint.json")


# ============================================================
# The refund requests
# ============================================================

NAMES = ["Alex", "Maria", "Chen", "Priya", "Tom", "Sara", "Omar", "Lena", "Ken", "Ana"]
ITEMS = [
    "desk lamp", "yoga mat", "coffee grinder", "travel mug", "ceramic mug set",
    "notebook set", "water bottle", "backpack", "phone stand", "bike light",
    "cutting board", "throw blanket", "plant pot", "wall clock", "mouse pad",
    "lunch box", "picture frame", "tea kettle", "shoe rack", "door mat",
]
DEFECTS = [
    ("arrived damaged", "The item arrived damaged and cannot be used."),
    ("stopped working", "The item stopped working after two days."),
    ("never arrived", "The package never arrived, and the carrier shows no delivery."),
    ("defective", "The item is defective and will not turn on."),
]
CHANGED_MIND = (
    "changed mind",
    "I simply changed my mind and no longer want it. It works fine and arrived 45 days ago.",
)
PADDING = (
    "Order details: placed online and paid by card. The package was handled by the "
    "standard carrier and tracked through the normal delivery steps. I contacted "
    "support earlier by chat and was asked to submit this form. I kept the original "
    "packaging and have attached photos where relevant. Please let me know the "
    "outcome by email. Additional notes: I have been a customer for several years "
    "and have ordered from this store many times without any problem. The delivery "
    "address on the order is correct and I was home during the delivery window. If "
    "you need more information, such as the order number or a copy of the receipt, "
    "I can send it right away. Thank you for your time and for looking into this "
    "request carefully. "
)
CHANGED_MIND_IDS = {3, 7, 11, 14, 19}  # these should be denied
OVER_LIMIT_ID = 16                     # this one needs the user's approval


def build_requests() -> dict[str, dict]:
    """20 refund requests with long free-text bodies, keyed by id (R01 .. R20)."""
    requests = {}
    for i in range(1, 21):
        rid = f"R{i:02d}"
        amount = 420 if i == OVER_LIMIT_ID else 40 + (i * 17) % 150
        label, sentence = CHANGED_MIND if i in CHANGED_MIND_IDS else DEFECTS[i % len(DEFECTS)]
        requests[rid] = {
            "item": ITEMS[i - 1],
            "amount": amount,
            "label": label,
            "approve": i not in CHANGED_MIND_IDS,
            "text": (
                f"Refund request {rid} from {NAMES[i % len(NAMES)]}. Item: {ITEMS[i - 1]}. "
                f"Amount: ${amount}. {sentence} {PADDING}"
            ),
        }
    return requests


REQUESTS = build_requests()
# Correct total: every defect request is approved, including the one the user approves.
EXPECTED_TOTAL = sum(r["amount"] for r in REQUESTS.values() if r["approve"])


# ============================================================
# The checkpoint (state kept outside the context)
# ============================================================

DECISIONS: dict[str, dict] = {}


def save_checkpoint(pending_question: str = "") -> None:
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"decisions": DECISIONS, "pending_question": pending_question}, f)


def load_checkpoint() -> dict:
    with open(CHECKPOINT_PATH) as f:
        state = json.load(f)
    DECISIONS.clear()
    DECISIONS.update(state["decisions"])
    return state


def restore_checkpoint(state: dict) -> None:
    """Put the checkpoint back exactly as it was when the agent paused."""
    DECISIONS.clear()
    DECISIONS.update(copy.deepcopy(state["decisions"]))
    save_checkpoint(state["pending_question"])


# ============================================================
# Tools
# ============================================================

def get_request(request_id: str) -> str:
    request = REQUESTS.get(request_id.strip().upper())
    return request["text"] if request else f"No request found for '{request_id}'."


def save_progress(request_id: str, decision: str, amount: float = 0) -> str:
    DECISIONS[request_id] = {"decision": decision, "amount": amount}
    save_checkpoint()  # the checkpoint is written after every decision
    print(f"      saved {request_id}: {decision}")
    return f"Saved {request_id}."


def get_total_approved() -> str:
    return str(sum(d["amount"] for d in DECISIONS.values() if d["decision"] == "approve"))


TOOL_IMPLS = {
    "get_request": get_request,
    "save_progress": save_progress,
    "get_total_approved": get_total_approved,
}


def tool_spec(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"toolSpec": {"name": name, "description": description, "inputSchema": {
        "json": {"type": "object", "properties": properties, "required": required}}}}


TOOL_CONFIG = {"tools": [
    tool_spec("get_request", "Read the full text of one refund request by its id (for example R01).",
              {"request_id": {"type": "string"}}, ["request_id"]),
    tool_spec("save_progress", "Save your decision for one request. Use amount 0 when denying.",
              {"request_id": {"type": "string"},
               "decision": {"type": "string", "enum": ["approve", "deny"]},
               "amount": {"type": "number"}},
              ["request_id", "decision", "amount"]),
    tool_spec("ask_user", "Ask the user a question you cannot decide yourself, then wait for the answer.",
              {"question": {"type": "string"}}, ["question"]),
    tool_spec("get_total_approved", "Return the total refund amount approved so far.", {}, []),
]}

SYSTEM_PROMPT = (
    "You are a refund-review agent. Review refund requests one at a time, in order. "
    "For each request: call get_request, decide, then call save_progress with your "
    "decision. Policy: approve a refund of up to "
    f"${APPROVAL_LIMIT} when the customer reports a defective, damaged, or "
    "undelivered item. Deny a refund when the customer only changed their mind. "
    f"A refund over ${APPROVAL_LIMIT} needs the user's approval: call ask_user with a "
    "short question, and do not decide it yourself. When every request is reviewed, "
    "call get_total_approved and reply with only that total as a plain number."
)


# ============================================================
# The agent loop (pauses when the agent asks the user)
# ============================================================

def run_agent(messages: list, max_turns: int = 80) -> dict:
    """Call the model and run tools until it answers or asks the user.

    Returns token usage for this run, and "paused" when the agent called ask_user.
    """
    input_tokens = output_tokens = turns = first_turn_input = 0

    while turns < max_turns:
        resp = RUNTIME.converse(
            modelId=HAIKU,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig=TOOL_CONFIG,
            inferenceConfig={"maxTokens": 512, "temperature": 0},
        )
        turns += 1
        first_turn_input = first_turn_input or resp["usage"]["inputTokens"]
        input_tokens += resp["usage"]["inputTokens"]
        output_tokens += resp["usage"]["outputTokens"]
        message = resp["output"]["message"]
        messages.append(message)

        usage = {"turns": turns, "input_tokens": input_tokens,
                 "output_tokens": output_tokens, "first_turn_input": first_turn_input}

        if resp["stopReason"] != "tool_use":
            text = next((b["text"] for b in message["content"] if "text" in b), "")
            return {**usage, "text": text, "paused": None}

        results, ask = [], None
        for block in message["content"]:
            if "toolUse" not in block:
                continue
            use = block["toolUse"]
            if use["name"] == "ask_user":
                ask = use
            else:
                output = TOOL_IMPLS[use["name"]](**use["input"])
                results.append({"toolResult": {"toolUseId": use["toolUseId"],
                                               "content": [{"text": output}]}})
        if ask:  # stop here and wait for the user
            return {**usage, "text": "", "paused": {
                "question": ask["input"]["question"],
                "tool_use_id": ask["toolUseId"],
                "other_results": results,
            }}
        messages.append({"role": "user", "content": results})

    return {**usage, "text": "", "paused": None}


# ============================================================
# Helpers
# ============================================================

def cost(r: dict) -> float:
    return (r["input_tokens"] * INPUT_PRICE + r["output_tokens"] * OUTPUT_PRICE) / 1_000_000


def approx_tokens(obj) -> int:
    """Rough token estimate (about 4 characters per token)."""
    return len(json.dumps(obj)) // 4


def is_correct(r: dict) -> bool:
    return str(EXPECTED_TOTAL) in re.sub(r"[,$\s]", "", r["text"])


def print_requests() -> None:
    """Show what the agent is about to read: a table of all requests, and one in full."""
    print(f"{'Id':<5} {'Item':<18} {'Amount':>7}  Customer says")
    for rid, r in REQUESTS.items():
        note = f"   <-- over ${APPROVAL_LIMIT}: needs the user" if r["amount"] > APPROVAL_LIMIT else ""
        print(f"{rid:<5} {r['item']:<18} {'$' + str(r['amount']):>7}  {r['label']}{note}")

    first = next(iter(REQUESTS))
    text = REQUESTS[first]["text"]
    print(f"\nEach request is a long free-text form. Here is {first} in full "
          f"(about {approx_tokens(text)} tokens; all {len(REQUESTS)} look like this):\n")
    print(textwrap.indent(textwrap.fill(text, width=68), "    "))


def print_checkpoint_file() -> None:
    """Print the checkpoint exactly as it is stored on disk."""
    with open(CHECKPOINT_PATH) as f:
        state = json.load(f)
    print(f"  Checkpoint file: {CHECKPOINT_PATH}\n")
    print("    {")
    print('      "decisions": {')
    items = list(state["decisions"].items())
    for n, (rid, d) in enumerate(items):
        comma = "," if n < len(items) - 1 else ""
        print(f'        "{rid}": {json.dumps(d)}{comma}')
    print("      },")
    print(f'      "pending_question": {json.dumps(state["pending_question"])}')
    print("    }")


def print_divider() -> None:
    print("\n" + "-" * 70 + "\n")


def print_intro() -> None:
    print("--- Long-Running Agent State: save state outside the context, resume from it ---")
    print(f"  Model: {HAIKU}")
    print(f"""
BACKGROUND
  A refund-review agent works through {len(REQUESTS)} refund requests. For each one it
  reads a long request text, decides, and saves its decision. Refunds over
  ${APPROVAL_LIMIT} need a person's approval. At request {OVER_LIMIT_ID} the agent asks the user,
  and the answer arrives hours later.

THE PROBLEM
  While it waits, nothing runs and no model tokens are used. But when the
  answer arrives, the agent has to come back. If it comes back by replaying the
  whole conversation, the prompt contains every request text it already
  reviewed, and that history is re-sent on every remaining turn.

THE FIX
  Save the agent's state outside the context while it works: the decisions made
  so far. When the answer arrives, start a fresh context from that small
  checkpoint and do only the remaining work. Because the state is saved, the
  process does not need to stay alive during the wait either.

WHAT THIS DEMO DOES
  Step 1  Look at the refund requests the agent will review.
  Step 2  Run the agent until it asks the user, saving a checkpoint as it goes.
  Step 3  Wait for the user, then finish the task two ways: replay the full
          history, and resume from the checkpoint.
  Step 4  Compare input tokens, turns, and cost.
""")


def print_comparison_table(runs: list[tuple[str, dict]]) -> None:
    base_cost = cost(runs[0][1])
    print("--- Comparison: finishing the same task after the wait ---")
    print(f"{'Resume method':<26} {'Turns':>6} {'First-turn input tokens':>24} "
          f"{'Input tokens':>13} {'Output tokens':>14} {'Cost $':>9} {'Relative cost':>14} "
          f"{'Correct total':>14}")
    for label, r in runs:
        print(f"{label:<26} {r['turns']:>6} {r['first_turn_input']:>24} "
              f"{r['input_tokens']:>13} {r['output_tokens']:>14} {cost(r):>9.5f} "
              f"{cost(r) / base_cost:>13.2f}x {'yes' if is_correct(r) else 'NO':>14}")
    print(
        f"\n  Expected total approved: {EXPECTED_TOTAL}. Both methods finish the same"
        "\n  task. Replay starts with the whole history in the prompt and re-sends it on"
        "\n  every remaining turn. Resuming from the checkpoint starts small. The wait"
        "\n  itself cost nothing in either case, and the checkpoint also let the process"
        "\n  shut down while it waited."
    )


# ============================================================
# Demo
# ============================================================

def main():
    print_intro()

    print_divider()
    print("Step 1 of 4: the refund requests the agent will review\n")
    print_requests()

    print_divider()
    print("Step 2 of 4: run the agent until it needs the user\n")
    messages = [{"role": "user", "content": [{"text": f"Review refund requests R01 to R{len(REQUESTS)}."}]}]
    first = run_agent(messages)
    if not first["paused"]:
        print("  The agent finished without asking the user. Run the sample again.")
        return
    paused = first["paused"]
    save_checkpoint(paused["question"])
    paused_state = load_checkpoint()  # the checkpoint as it stands at the pause
    print(f"\n  The agent asked the user: \"{paused['question']}\"")
    print(f"  Work so far: {first['turns']} turns, {first['input_tokens']} input tokens, "
          f"{len(DECISIONS)} decisions saved.")
    print("\n  While it worked, the harness kept this checkpoint outside the context.")
    print("  It is the state a new run needs to continue:\n")
    print_checkpoint_file()
    print(f"\n  Conversation so far: ~{approx_tokens(messages)} tokens.  "
          f"Checkpoint: ~{approx_tokens(DECISIONS)} tokens.")

    print_divider()
    print(f"Step 3 of 4: {WAIT_HOURS} hours pass, then the user answers\n")
    print(f"  (Simulated wait. No model calls and no running agent during these {WAIT_HOURS} hours.)")
    print(f"  User answer: \"{USER_ANSWER}\"")

    saved_history = copy.deepcopy(messages)  # what a replay would reload
    runs = []

    print("\n  Resume A - replay the full conversation, then continue:")
    print(f"      The model receives all {len(saved_history)} saved messages (~{approx_tokens(saved_history)} tokens),")
    print("      then the user's answer.")
    restore_checkpoint(paused_state)
    replay_messages = saved_history + [{"role": "user", "content": paused["other_results"] + [
        {"toolResult": {"toolUseId": paused["tool_use_id"], "content": [{"text": USER_ANSWER}]}}
    ]}]
    runs.append(("Replay full history", run_agent(replay_messages)))

    print("\n  Resume B - fresh context from the checkpoint:")
    restore_checkpoint(paused_state)
    state = load_checkpoint()
    saved_lines = "\n".join(
        f"- {rid}: {d['decision']}" + (f" ${d['amount']}" if d["decision"] == "approve" else "")
        for rid, d in state["decisions"].items()
    )
    remaining = ", ".join(rid for rid in REQUESTS if rid not in state["decisions"])
    resume_prompt = (
        f"A refund review was paused. These requests are already reviewed and saved:\n"
        f"{saved_lines}\n\n"
        f"You asked the user: \"{state['pending_question']}\"\n"
        f"The user answered: \"{USER_ANSWER}\"\n\n"
        f"Continue with the remaining requests, in order: {remaining}."
    )
    print("      The model receives only this message:\n")
    print(textwrap.indent(resume_prompt, "          | "))
    print()
    state_messages = [{"role": "user", "content": [{"text": resume_prompt}]}]
    runs.append(("Resume from checkpoint", run_agent(state_messages)))

    print_divider()
    print("Step 4 of 4: compare\n")
    print_comparison_table(runs)

    os.remove(CHECKPOINT_PATH)

    print("\n--- Done ---")
    print("  Next steps:")
    print("  1. Write the checkpoint after every step, so any pause or restart is safe")
    print("  2. Keep the checkpoint small: decisions and ids, not raw tool output")
    print("  3. Store it somewhere durable (DynamoDB, S3, AgentCore Memory) at real scale")


if __name__ == "__main__":
    main()
