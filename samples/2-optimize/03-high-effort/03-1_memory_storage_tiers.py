"""
High-Effort Optimization - Lever 13: Memory Storage Tiers

An agent with memory remembers facts about its users: preferences, past orders,
earlier problems. The simplest design keeps all of it in the prompt, so every
call pays for every fact the agent has ever stored. Memory only grows, so that
bill grows with it, even though most old facts are irrelevant to the question
being asked.

Tiering fixes this by matching memory to how often it is used:

  - HOT:  recently used facts. Always in the prompt, because they are likely
          to matter.
  - COLD: old facts. Kept in cheaper storage, out of the prompt. The agent
          searches them only when a question needs one and loads just the
          best matches.

This sample is a small local simulation. The memory is a Python list, the tier
policy is an age cutoff, and the cold search is a keyword match. In production
the tiers would be separate stores (for example Amazon Bedrock AgentCore Memory
short-term events and long-term records, or a cache plus a cheaper database).

You will learn how to:
- Split memory into hot and cold tiers with a simple age policy
- Load only hot facts plus the top matching cold facts into the prompt
- Compare input tokens, cost, and correct answers against keeping it all hot

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude 5 Sonnet on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import re

import boto3

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")
RUNTIME = boto3.client("bedrock-runtime", region_name=REGION)
SONNET = "global.anthropic.claude-sonnet-5"

# Pricing per 1M tokens (USD). Illustrative workshop values - always verify
# current pricing at https://aws.amazon.com/bedrock/pricing/
INPUT_PRICE, OUTPUT_PRICE = 3.00, 15.00

# The tier policy: facts used within this many days are hot, the rest are cold.
HOT_DAYS = 7

# How many cold facts to load into the prompt for one question.
TOP_K = 2


# ============================================================
# The memory: facts about one customer, with their age in days
# ============================================================

BASE_FACTS = [
    (1, "Asked about order 88231, a standing desk, and was told it ships on Friday."),
    (2, "Prefers to be contacted by email rather than by phone."),
    (4, "Updated the delivery address to 42 Harbor Street, Portland."),
    (5, "Redeemed a 10 percent loyalty discount on a desk lamp."),
    (7, "Asked for an invoice copy for order 88231."),
    (9, "Subscribed to the monthly newsletter."),
    (12, "Chose forest green as the preferred product color."),
    (15, "Reported a late delivery for a yoga mat and received a 5 dollar credit."),
    (18, "Asked whether the standing desk comes with a cable tray."),
    (21, "Returned a mechanical keyboard because the space bar was sticky."),
    (24, "Joined the loyalty program at the silver level."),
    (28, "Asked for gift wrapping on a ceramic mug set."),
    (31, "Allergic to wool, so avoid recommending wool blankets or sweaters."),
    (35, "Changed the payment method from a debit card to a credit card."),
    (40, "Asked about the warranty length for a coffee grinder."),
    (44, "Reported a cracked screen protector and received a free replacement."),
    (49, "Prefers eco-friendly packaging when available."),
    (53, "Asked for a bulk discount on twenty notebooks for a workshop."),
    (58, "Switched the delivery speed to standard shipping to save money."),
    (63, "Asked how to reset the password for the online account."),
    (70, "Requested a catalog of outdoor furniture."),
    (77, "Opened the account using a work email address."),
    (84, "Placed a first order for a travel mug."),
    (90, "Asked about shipping options to Canada."),
]

# A long-standing customer has a long history. These generated facts stand in for
# it, so the memory grows to a realistic size. They are built to never mention
# the topics the questions below ask about.
HISTORY_FACTS = 150
HISTORY_ITEMS = [
    "desk lamp", "yoga mat", "coffee grinder", "travel mug", "ceramic mug set",
    "notebook", "water bottle", "backpack", "phone stand", "bike light",
    "cutting board", "throw blanket", "plant pot", "wall clock", "mouse pad",
    "lunch box", "picture frame", "tea kettle", "shoe rack", "door mat",
]
HISTORY_TEMPLATES = [
    "Asked about the warranty length for the {item} on order {n}.",
    "Ordered a {item} as part of order {n}.",
    "Asked for an invoice copy for order {n}, which included a {item}.",
    "Reported a delayed delivery for the {item} on order {n} and received a small credit.",
    "Requested gift wrapping for the {item} on order {n}.",
    "Asked about shipping options for the {item} on order {n}.",
]


def build_memory() -> list[tuple[int, str]]:
    """The hand-written facts plus the generated history, newest first."""
    history = [
        (
            8 + (i * 172) // HISTORY_FACTS,
            HISTORY_TEMPLATES[i % len(HISTORY_TEMPLATES)].format(
                item=HISTORY_ITEMS[(i * 7) % len(HISTORY_ITEMS)], n=50000 + i
            ),
        )
        for i in range(HISTORY_FACTS)
    ]
    return sorted(BASE_FACTS + history, key=lambda fact: fact[0])


MEMORY = build_memory()

# "expect" is text that only appears in a correct answer.
QUESTIONS = [
    {"question": "What is the current street address?", "expect": "Harbor"},
    {"question": "Why did this customer return the keyboard?", "expect": "space bar"},
    {"question": "Which material is the customer allergic to?", "expect": "wool"},
]

SYSTEM_PROMPT = (
    "You are a concise customer-support agent. Answer the question using only "
    "the customer memory below. Each fact shows how many days ago it was recorded."
)


# ============================================================
# The tiers
# ============================================================

def apply_tier_policy(memory: list[tuple[int, str]]) -> tuple[list, list]:
    """Split memory into (hot, cold) by age."""
    hot = [fact for fact in memory if fact[0] <= HOT_DAYS]
    cold = [fact for fact in memory if fact[0] > HOT_DAYS]
    return hot, cold


STOPWORDS = {
    "what", "which", "when", "where", "does", "this", "that", "with", "from",
    "have", "their", "customer", "about", "name",
}


def keywords(text: str) -> set[str]:
    """Words of 4+ letters, cut to 5 letters so 'return' matches 'returned'."""
    words = re.findall(r"[a-z]+", text.lower())
    return {w[:5] for w in words if len(w) >= 4 and w not in STOPWORDS}


def age(days: int) -> str:
    return "1 day ago" if days == 1 else f"{days} days ago"


def search_cold(question: str, cold: list, top_k: int = TOP_K) -> list:
    """Return the cold facts that share the most keywords with the question."""
    wanted = keywords(question)
    scored = [(len(wanted & keywords(fact[1])), fact) for fact in cold]
    best = sorted((s for s in scored if s[0] > 0), key=lambda s: -s[0])[:top_k]
    return [fact for _, fact in best]


# ============================================================
# Asking the model
# ============================================================

def ask(question: str, facts: list) -> dict:
    """Put the given facts in the prompt and ask the question."""
    memory_text = "\n".join(f"- ({age(days)}) {text}" for days, text in facts)
    resp = RUNTIME.converse(
        modelId=SONNET,
        system=[{"text": f"{SYSTEM_PROMPT}\n\nCUSTOMER MEMORY:\n{memory_text}"}],
        messages=[{"role": "user", "content": [{"text": question}]}],
        inferenceConfig={"maxTokens": 1024},
    )
    # Claude 5 Sonnet is thinking-capable: the answer is the text block, which is
    # not necessarily the first block. It also does not accept a temperature.
    blocks = resp["output"]["message"]["content"]
    return {
        "text": next((b["text"] for b in blocks if "text" in b), ""),
        "facts": len(facts),
        "input_tokens": resp["usage"]["inputTokens"],
        "output_tokens": resp["usage"]["outputTokens"],
    }


def cost(result: dict) -> float:
    return (result["input_tokens"] * INPUT_PRICE + result["output_tokens"] * OUTPUT_PRICE) / 1_000_000


# ============================================================
# Output
# ============================================================

def print_divider() -> None:
    print("\n" + "-" * 70 + "\n")


def print_intro() -> None:
    print("--- Memory Storage Tiers: keep hot memory in the prompt, search the rest ---")
    print(f"  Model: {SONNET}")
    print(f"""
BACKGROUND
  A customer-support agent remembers {len(MEMORY)} facts about a returning customer,
  up to {max(d for d, _ in MEMORY)} days old: preferences, past orders, and earlier problems.
  The agent puts memory in the prompt so it can answer personal questions.

THE PROBLEM
  If all memory goes in the prompt, every call pays for every fact, even though
  most old facts have nothing to do with the question. Memory only grows, so
  the cost per call grows with it.

THE FIX
  Tier the memory by how recently it was used:
    - Hot  (used in the last {HOT_DAYS} days): always in the prompt.
    - Cold (older): out of the prompt. Search it when a question needs it and
      load only the top {TOP_K} matches.

WHAT THIS DEMO DOES
  Step 1  Split the memory into hot and cold tiers.
  Step 2  Ask {len(QUESTIONS)} questions two ways: all memory in the prompt, and tiered.
  Step 3  Compare input tokens, cost, and correct answers.
""")


def print_comparison_table(results: dict[str, list[dict]]) -> None:
    all_hot_cost = sum(cost(r) for r in results["All in the prompt"])

    print("--- Comparison: same questions, two memory designs ---")
    print(f"{'Design':<20} {'Facts per call':>15} {'Input tokens':>13} {'Output tokens':>14} "
          f"{'Cost $':>9} {'Relative cost':>14} {'Correct':>8}")
    for label, rs in results.items():
        total_cost = sum(cost(r) for r in rs)
        print(f"{label:<20} {sum(r['facts'] for r in rs) / len(rs):>15.1f} "
              f"{sum(r['input_tokens'] for r in rs):>13} "
              f"{sum(r['output_tokens'] for r in rs):>14} "
              f"{total_cost:>9.5f} {total_cost / all_hot_cost:>13.2f}x "
              f"{sum(r['correct'] for r in rs):>5} of {len(rs)}")
    print(
        "\n  Both designs answer the same questions. The tiered design sends only the"
        "\n  facts that are likely to matter, so each call carries far fewer input"
        "\n  tokens. The gap widens as memory grows: the all-in-prompt design pays for"
        "\n  every new fact on every call, while the tiered design pays only for the"
        "\n  hot facts and the few cold matches."
    )


# ============================================================
# Demo
# ============================================================

def main():
    print_intro()

    print_divider()
    print("Step 1 of 3: split the memory into tiers\n")
    hot, cold = apply_tier_policy(MEMORY)
    print(f"  Hot  (last {HOT_DAYS} days): {len(hot)} facts, always in the prompt")
    for days, text in hot:
        print(f"      ({age(days)}) {text}")
    print(f"  Cold (older):        {len(cold)} facts, searched only when needed")

    print_divider()
    print("Step 2 of 3: ask the same questions with each design\n")
    results = {"All in the prompt": [], "Tiered": []}
    for q in QUESTIONS:
        found = search_cold(q["question"], cold)
        all_hot = ask(q["question"], MEMORY)
        tiered = ask(q["question"], hot + found)
        for r in (all_hot, tiered):
            r["correct"] = q["expect"].lower() in r["text"].lower()
        results["All in the prompt"].append(all_hot)
        results["Tiered"].append(tiered)

        print(f"Question: {q['question']}")
        for label, r, detail in [
            ("All in the prompt", all_hot, f"{len(MEMORY)} facts"),
            ("Tiered", tiered, f"{len(hot)} hot + {len(found)} cold facts"),
        ]:
            mark = "correct" if r["correct"] else f"<-- WRONG (expected '{q['expect']}')"
            print(f"  {label:<18}: {detail:<20} | {r['input_tokens']:>4} input tokens "
                  f"| ${cost(r):.5f} | {mark}")
        print()

    print_divider()
    print("Step 3 of 3: compare\n")
    print_comparison_table(results)

    print("\n--- Done ---")
    print("  Next steps:")
    print("  1. Set the hot window from how often your users actually revisit old facts")
    print("  2. Use a managed store or vector search for the cold tier at real scale")
    print("  3. Track how often cold search is needed - it tells you if the cutoff is right")


if __name__ == "__main__":
    main()
