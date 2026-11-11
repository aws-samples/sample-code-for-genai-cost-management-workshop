"""
Medium-Effort Optimization - Lever 10: Remove Extra Model Calls

The cheapest model call is the one you never make. Pipelines grow by adding
"one more prompt" for each step: pull out the order ID, add up the line items,
convert the date, check a policy window. Each of those is a deterministic job
that a few lines of Python do instantly, for free, and without ever being
wrong. A model call for the same step costs tokens, adds latency, and can
quietly get arithmetic or date math wrong - and that error flows into every
step that follows.

The rule: run deterministic steps in code, and use the model only for
reasoning and semantics (understanding what the customer wants, writing the
reply).

This sample processes a small batch of customer support tickets two ways:
- ALL MODEL: five model calls per ticket - extract the order ID, compute the
  total with tax, normalize the date, check refund eligibility, draft a reply.
  Each answer feeds the reply prompt, so a wrong answer reaches the customer.
- HYBRID: the first four steps run in Python (regex, Decimal math, datetime,
  a lookup table); one model call drafts the reply from the verified facts.

You will learn how to:
- Spot steps that do not need a model (parsing, math, dates, lookups, rules)
- Replace them with plain Python and keep the model for the semantic step
- Measure model calls, tokens, latency, cost, and correctness for both designs

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude Haiku 4.5 on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import re
import time
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

import boto3

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

RUNTIME = boto3.client("bedrock-runtime", region_name=REGION)

HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# Illustrative pricing per 1M tokens (USD). Verify current pricing at
# https://aws.amazon.com/bedrock/pricing/
INPUT_PRICE_PER_M = 1.00
OUTPUT_PRICE_PER_M = 5.00

# Business rules for the ticket pipeline. Order dates are set relative to today
# (see days_ago below), so every run has the same mix of eligible, ineligible,
# and borderline orders no matter when you run it.
TODAY = date.today()
REFUND_WINDOW_DAYS = 30
TAX_BY_COUNTRY = {
    "Germany": Decimal("0.19"),
    "France": Decimal("0.20"),
    "Canada": Decimal("0.05"),
    "United States": Decimal("0.07"),
    "Japan": Decimal("0.10"),
}

# Each ticket is a system-generated order block (what a ticketing system
# appends) plus the customer's free-text message. The order date is placed
# days_ago days before today and written in the given format, because the order
# system emits dates in several formats. The 30 and 31 day tickets sit on either
# side of the refund window.
TICKETS = [
    {
        "order": (
            "Order: ORD-482913\n"
            "Date: {date}\n"
            "Items:\n"
            "  2 x Wireless Mouse @ 24.99\n"
            "  1 x USB-C Hub @ 39.50\n"
            "Ship to: Germany"
        ),
        "days_ago": 18,
        "date_format": "%b %d, %Y",
        "message": (
            "Hi, the wireless mouse arrived with a cracked scroll wheel. I'm quite "
            "frustrated, this is the second time I've had a problem. Can I return "
            "it for a refund?"
        ),
    },
    {
        "order": (
            "Order: ORD-517204\n"
            "Date: {date}\n"
            "Items:\n"
            "  1 x Mechanical Keyboard @ 119.00\n"
            "  2 x Keycap Set @ 18.75\n"
            "Ship to: France"
        ),
        "days_ago": 37,
        "date_format": "%d %B %Y",
        "message": (
            "Hello, I'd like to return the keyboard because the keys are too loud "
            "for my open office. Nothing is wrong with it, it is just not what I "
            "expected."
        ),
    },
    {
        "order": (
            "Order: ORD-390557\n"
            "Date: {date}\n"
            "Items:\n"
            "  3 x HDMI Cable @ 12.99\n"
            "  1 x Docking Station @ 149.95\n"
            "Ship to: Canada"
        ),
        "days_ago": 30,
        "date_format": "%Y-%m-%d",
        "message": (
            "My order showed up without the docking station's power adapter. "
            "Could you please send it or credit me?"
        ),
    },
    {
        "order": (
            "Order: ORD-662148\n"
            "Date: {date}\n"
            "Items:\n"
            "  1 x 4K Webcam @ 89.90\n"
            "  1 x Webcam Tripod @ 21.49\n"
            "Ship to: United States"
        ),
        "days_ago": 52,
        "date_format": "%b %d, %Y",
        "message": (
            "Is it possible to get my money back for the webcam? The image quality "
            "is much worse than the listing promised."
        ),
    },
    {
        "order": (
            "Order: ORD-704331\n"
            "Date: {date}\n"
            "Items:\n"
            "  1 x Noise-Cancelling Headphones @ 199.00\n"
            "  2 x Replacement Ear Pads @ 14.25\n"
            "Ship to: Japan"
        ),
        "days_ago": 31,
        "date_format": "%m/%d/%Y",
        "message": (
            "The headphones stopped charging after two weeks. I'd like a refund or "
            "a replacement, whichever is faster."
        ),
    },
]


# Fill in each ticket's order date.
for _t in TICKETS:
    _d = TODAY - timedelta(days=_t["days_ago"])
    _t["order"] = _t["order"].format(date=_d.strftime(_t["date_format"]))


def ticket_text(ticket: dict) -> str:
    """The full ticket as the model would see it."""
    return (
        "--- ORDER (system generated) ---\n"
        f"{ticket['order']}\n"
        "--- CUSTOMER MESSAGE ---\n"
        f"{ticket['message']}"
    )


# ============================================================
# Deterministic steps - plain Python, no model
# ============================================================

ORDER_ID_RE = re.compile(r"^Order:[ \t]*(ORD-\d{6})[ \t]*$", re.MULTILINE)
DATE_RE = re.compile(r"^Date:[ \t]*(.+?)[ \t]*$", re.MULTILINE)
COUNTRY_RE = re.compile(r"^Ship to:[ \t]*(.+?)[ \t]*$", re.MULTILINE)
ITEM_RE = re.compile(r"^[ \t]*(\d+) x .+? @ (\d+\.\d{2})[ \t]*$", re.MULTILINE)

# The order system emits dates in a few formats. Listing them is a lookup, not
# a reasoning task.
DATE_FORMATS = ["%b %d, %Y", "%d %B %Y", "%Y-%m-%d", "%m/%d/%Y"]


def extract_order_id(order: str) -> str:
    return ORDER_ID_RE.search(order).group(1)


def parse_order_date(order: str) -> date:
    raw = DATE_RE.search(order).group(1)
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {raw}")


def order_total(order: str) -> Decimal:
    """Sum the line items, apply the destination country's tax rate."""
    subtotal = sum(
        Decimal(qty) * Decimal(price) for qty, price in ITEM_RE.findall(order)
    )
    rate = TAX_BY_COUNTRY[COUNTRY_RE.search(order).group(1)]
    return (subtotal * (1 + rate)).quantize(Decimal("0.01"), ROUND_HALF_UP)


def refund_eligible(order_date: date) -> bool:
    return (TODAY - order_date).days <= REFUND_WINDOW_DAYS


def compute_facts(order: str) -> dict:
    """All four deterministic steps in one pass - microseconds, zero tokens."""
    order_date = parse_order_date(order)
    return {
        "order_id": extract_order_id(order),
        "total": order_total(order),
        "iso_date": order_date.isoformat(),
        "eligible": refund_eligible(order_date),
    }


# ============================================================
# Model calls
# ============================================================

def call_model(prompt: str, max_tokens: int) -> dict:
    """One Converse call. Returns text, token usage, and wall-clock latency."""
    start = time.perf_counter()
    resp = RUNTIME.converse(
        modelId=HAIKU,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
    )
    return {
        "text": resp["output"]["message"]["content"][0]["text"].strip(),
        "input_tokens": resp["usage"]["inputTokens"],
        "output_tokens": resp["usage"]["outputTokens"],
        "latency": time.perf_counter() - start,
    }


def new_totals() -> dict:
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency": 0.0,
            "checks": 0, "correct": 0}


def tracked_call(totals: dict, prompt: str, max_tokens: int) -> str:
    """call_model plus running totals for the pipeline."""
    r = call_model(prompt, max_tokens)
    totals["calls"] += 1
    totals["input_tokens"] += r["input_tokens"]
    totals["output_tokens"] += r["output_tokens"]
    totals["latency"] += r["latency"]
    return r["text"]


def reply_prompt(context: str, facts: dict) -> str:
    return (
        "You are a customer support agent. Write a short, empathetic reply "
        "(under 80 words) to the customer's message. State the order ID and "
        "order total, and say plainly whether the order is eligible for a "
        "refund. Use exactly these verified facts and do not invent others.\n\n"
        f"Order ID: {facts['order_id']}\n"
        f"Order total (incl. tax): {facts['total']}\n"
        f"Refund eligible: {'YES' if facts['eligible'] else 'NO'} "
        f"({REFUND_WINDOW_DAYS}-day window)\n\n"
        f"{context}"
    )


# ============================================================
# Design 1: every step is a model call
# ============================================================

def run_all_model(ticket: dict, totals: dict) -> dict:
    """Five model calls per ticket. Returns the model's facts and its reply."""
    text = ticket_text(ticket)
    tax_table = ", ".join(
        f"{country} {int(rate * 100)}%" for country, rate in TAX_BY_COUNTRY.items()
    )

    order_id = tracked_call(
        totals,
        f"Extract the order ID from this support ticket. Reply with only the ID.\n\n{text}",
        40,
    )
    total = tracked_call(
        totals,
        "Calculate the order total including tax for this support ticket. "
        f"Tax rates by destination country: {tax_table}. Apply the rate to the "
        "subtotal of all items. Reply with only the number rounded to 2 "
        f"decimals, no currency symbol.\n\n{text}",
        # Haiku often shows its arithmetic despite "reply with only", so give it
        # room to finish; a tight cap would unfairly cut off the answer.
        400,
    )
    iso_date = tracked_call(
        totals,
        "Convert the order date in this support ticket to ISO format "
        f"YYYY-MM-DD. Reply with only the date.\n\n{text}",
        40,
    )
    eligible = tracked_call(
        totals,
        f"Today is {TODAY.isoformat()}. Refunds are allowed within "
        f"{REFUND_WINDOW_DAYS} days of the order date, inclusive. Is the order "
        f"in this support ticket eligible for a refund? Reply with only YES or NO.\n\n{text}",
        200,
    )

    # The reply prompt uses the model's own answers, so any wrong answer above
    # is repeated to the customer. Take the last number / last YES-NO in each
    # answer as the model's final word.
    numbers = re.findall(r"\d+(?:\.\d+)?", total.replace(",", ""))
    verdicts = re.findall(r"\b(YES|NO)\b", eligible.upper())
    facts = {
        "order_id": order_id,
        "total": numbers[-1] if numbers else total,
        "iso_date": iso_date,
        "eligible": bool(verdicts) and verdicts[-1] == "YES",
    }
    reply = tracked_call(totals, reply_prompt(text, facts), 250)
    return {"facts": facts, "reply": reply}


def score_model_facts(model_facts: dict, truth: dict) -> list[tuple[str, bool]]:
    """Compare the model's answers to the Python ground truth."""
    try:
        total_ok = abs(Decimal(model_facts["total"]) - truth["total"]) < Decimal("0.005")
    except Exception:
        total_ok = False
    return [
        ("order id", model_facts["order_id"] == truth["order_id"]),
        ("total", total_ok),
        ("date", model_facts["iso_date"] == truth["iso_date"]),
        ("refund", model_facts["eligible"] == truth["eligible"]),
    ]


# ============================================================
# Design 2: code for the deterministic steps, one model call
# ============================================================

def run_hybrid(ticket: dict, totals: dict) -> dict:
    """Four steps in Python, one model call for the reply."""
    facts = compute_facts(ticket["order"])
    # Only the customer's message goes to the model - it does not need the
    # order block, because the facts are already extracted.
    context = f"CUSTOMER MESSAGE:\n{ticket['message']}"
    reply = tracked_call(totals, reply_prompt(context, facts), 250)
    return {"facts": facts, "reply": reply}


# ============================================================
# Comparison
# ============================================================

def cost_of(t: dict) -> float:
    return (
        t["input_tokens"] / 1_000_000 * INPUT_PRICE_PER_M
        + t["output_tokens"] / 1_000_000 * OUTPUT_PRICE_PER_M
    )


RESULTS: list[tuple[str, dict]] = []


def print_comparison_table() -> None:
    base_cost = cost_of(RESULTS[0][1])
    headers = ["Pipeline", "Model calls", "Input tokens", "Output tokens",
               "Latency (s)", "Cost ($)", "vs. all-model", "Steps correct"]
    rows = []
    for label, t in RESULTS:
        c = cost_of(t)
        rows.append([
            label,
            str(t["calls"]),
            str(t["input_tokens"]),
            str(t["output_tokens"]),
            f"{t['latency']:.1f}",
            f"{c:.5f}",
            f"{c / base_cost:.2f}x",
            t["steps_correct"],
        ])

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    left = {0, 7}  # text columns; the rest are numbers

    def fmt(cells: list[str]) -> str:
        return "  ".join(
            c.ljust(widths[i]) if i in left else c.rjust(widths[i])
            for i, c in enumerate(cells)
        )

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))

    all_model, hybrid = RESULTS[0][1], RESULTS[1][1]
    fewer = (1 - hybrid["calls"] / all_model["calls"]) * 100
    cheaper = (1 - cost_of(hybrid) / base_cost) * 100
    print(
        f"\n  Model calls dropped {all_model['calls']} -> {hybrid['calls']} "
        f"({fewer:.0f}% fewer) and cost fell {cheaper:.0f}%."
        "\n  The four deterministic steps now take microseconds in Python, cost no"
        "\n  tokens, and are exact. The remaining model call does the one job that"
        "\n  needs language understanding: writing the reply.\n"
    )


# ============================================================
# Main
# ============================================================

def facts_line(label: str, facts: dict) -> str:
    """One aligned line of the four values, so two designs read side by side."""
    refund = "YES" if facts["eligible"] else "NO"
    return (
        f"    {label:<10} id {facts['order_id']} | total {str(facts['total']):>7} | "
        f"date {facts['iso_date']} | refund {refund}"
    )


def main():
    print("--- Remove Extra Model Calls: code for deterministic steps ---")
    print(f"  Model: Claude Haiku 4.5, region {REGION}")
    print(f"  Tickets: {len(TICKETS)}, today = {TODAY.isoformat()}, "
          f"refund window = {REFUND_WINDOW_DAYS} days\n")

    # Step 1: run every ticket through both designs and show the two answers
    # next to each other. The Python line is the correct answer: code computes
    # it exactly. The model line is what the model answered when it did each
    # step itself.
    print("Step 1 of 3: run each ticket through both designs")
    print("  Python = code does the four steps (exact).")
    print("  Model  = a separate model call for each of the four steps.\n")

    hybrid = new_totals()
    all_model = new_totals()
    first_reply_hybrid = ""
    first_reply_all = ""
    for i, ticket in enumerate(TICKETS, start=1):
        code_out = run_hybrid(ticket, hybrid)
        model_out = run_all_model(ticket, all_model)

        checks = score_model_facts(model_out["facts"], code_out["facts"])
        all_model["checks"] += len(checks)
        all_model["correct"] += sum(ok for _, ok in checks)
        wrong = [name for name, ok in checks if not ok]
        verdict = f"   <-- WRONG: {', '.join(wrong)}" if wrong else "   (matches)"

        print(f"  Ticket {i}: ordered {ticket['days_ago']} days ago")
        print(facts_line("Python", code_out["facts"]))
        print(facts_line("Model", model_out["facts"]) + verdict + "\n")

        if i == 1:
            first_reply_hybrid = code_out["reply"]
            first_reply_all = model_out["reply"]

    all_model["steps_correct"] = f"{all_model['correct']}/{all_model['checks']}"
    # The facts come from code, so there is nothing for the model to get wrong.
    n_checks = len(TICKETS) * 4
    hybrid["steps_correct"] = f"{n_checks}/{n_checks} (exact)"
    RESULTS.append(("All model (5 calls/ticket)", all_model))
    RESULTS.append(("Hybrid (code + 1 call/ticket)", hybrid))

    # Step 2: the customer-facing replies.
    print("Step 2 of 3: reply to ticket 1 from each design (truncated)")
    print(f"  all model: {' '.join(first_reply_all.split())[:300]}")
    print(f"  hybrid:    {' '.join(first_reply_hybrid.split())[:300]}\n")

    # Step 3: comparison table.
    print("Step 3 of 3: comparison\n")
    print_comparison_table()

    print("--- Done ---")
    print("  To find extra model calls in your own pipeline:")
    print("  1. List every model call and ask: does this step need judgment or")
    print("     language understanding, or is it a fixed rule?")
    print("  2. Move parsing, arithmetic, dates, lookups, validation and routing-by-")
    print("     rule into code; keep summarizing, classifying and writing on the model")
    print("  3. If input is too messy for a parser, fall back to the model only for")
    print("     the records the code cannot handle")
    print("  4. Confirm the pipeline output is unchanged on an eval set (see 01-6)")


if __name__ == "__main__":
    main()
