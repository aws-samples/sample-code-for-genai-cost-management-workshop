"""
Low-Effort Optimization - Lever 02: Prompt Design

The most under-used lever, and the deepest. Most production prompts carry
30-50% wasted tokens (hedging, boilerplate, vague instructions), which costs
money on every call and degrades quality. Three techniques, in order of how
often you reach for them:

  1. Clear, specific instructions - the baseline, always
  2. Few-shot examples - when instructions alone leave room to guess
  3. Structured output - when downstream code consumes the result

You will learn how to:
- Rewrite a vague prompt into a directive, parsable one (fewer tokens, better output)
- Use few-shot examples to pin output format across varied inputs
- Get structured output three ways: prompt-based, tool-use, and native JSON schema
- Run the same prompts on Amazon Nova 2 Lite through the same Converse API, and see
  which structured-output methods it supports (it has no native JSON schema)

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude and Amazon Nova models on Amazon Bedrock
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

# Allowed workshop models (Global cross-region inference profiles)
HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET = "global.anthropic.claude-sonnet-5"
NOVA = "global.amazon.nova-2-lite-v1:0"

# Pricing per 1M tokens (USD). Illustrative workshop values - always verify
# current pricing at https://aws.amazon.com/bedrock/pricing/
PRICING = {
    HAIKU: {"input": 1.00, "output": 5.00},
    SONNET: {"input": 3.00, "output": 15.00},
    NOVA: {"input": 0.30, "output": 2.50},
}

# Readable labels for the output and summary table.
LABELS = {HAIKU: "Haiku 4.5", SONNET: "Sonnet 5", NOVA: "Nova 2 Lite"}

EMAIL = (
    "Hi, the wireless headphones I ordered last week arrived with a cracked case "
    "and the left earbud won't charge. I'd like a replacement before the weekend."
)

CLASSIFY_EMAIL = (
    "My order #A-123 arrived with a cracked screen and the charger is missing. "
    "I want a replacement this week."
)

# Shared schema for the structured-output demos.
SCHEMA = {
    "type": "object",
    "properties": {
        "issue": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "action": {"type": "string"},
    },
    "required": ["issue", "sentiment", "action"],
    "additionalProperties": False,
}


# ============================================================
# Helper
# ============================================================

def build_inference_config(model_id: str, max_tokens: int) -> dict:
    """Build inferenceConfig, omitting temperature for models that reject it.

    Claude 5 models (Sonnet 5, Opus 5) deprecated the temperature parameter and
    return a ValidationException if it is supplied. Haiku 4.5 still accepts it.
    """
    cfg = {"maxTokens": max_tokens}
    if "claude-sonnet-5" not in model_id and "claude-opus-5" not in model_id:
        cfg["temperature"] = 0
    return cfg


# Token usage per call, collected as the demos run and summarised at the end.
# Each entry: (use_case, technique, model_id, input_tokens, output_tokens)
USAGE_LOG: list[tuple[str, str, str, int, int]] = []


def log_usage(use_case: str, technique: str, model_id: str, usage: dict) -> None:
    """Record the token usage of one Converse call for the summary table."""
    USAGE_LOG.append(
        (use_case, technique, model_id, usage["inputTokens"], usage["outputTokens"])
    )


def print_usage_table() -> None:
    """Print a comparison table of average tokens and cost per call for every technique."""
    # Group calls by (use case, technique, model), keeping first-seen order.
    groups: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for use_case, technique, model_id, tok_in, tok_out in USAGE_LOG:
        groups.setdefault((use_case, technique, model_id), []).append((tok_in, tok_out))

    short = LABELS
    rows = []
    for (use_case, technique, model_id), calls in groups.items():
        n = len(calls)
        avg_in = sum(c[0] for c in calls) / n
        avg_out = sum(c[1] for c in calls) / n
        prices = PRICING[model_id]
        avg_cost = (avg_in * prices["input"] + avg_out * prices["output"]) / 1_000_000
        rows.append(
            (
                use_case,
                technique,
                short.get(model_id, model_id),
                str(n),
                str(round(avg_in)),
                str(round(avg_out)),
                f"{avg_cost:.6f}",
            )
        )

    headers = ("Use case", "Technique", "Model", "Calls", "Avg input", "Avg output", "Avg $/call")
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt(cells) -> str:
        # Left-align text columns, right-align the numeric ones.
        parts = [
            c.ljust(widths[i]) if i < 3 else c.rjust(widths[i])
            for i, c in enumerate(cells)
        ]
        return "  " + " | ".join(parts)

    print("--- Comparison: tokens and cost per technique (average per call) ---\n")
    print(fmt(headers))
    print("  " + "-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))
    print("\n  - The VAGUE prompt hit the 200 max_tokens cap, so its real cost is higher.")
    print("  - Few-shot adds input tokens but trims output and pins the format.")
    print("  - Prompt-based and prefill are cheapest but give no schema guarantee.")
    print("  - Native JSON schema guarantees valid output with fewer input tokens than")
    print("    strict tool use, which carries the tool definition on every call.")
    print("  - Nova 2 Lite is the cheapest per token, but has no native JSON schema mode")
    print("    and no strict tool use, so validate its structured output in code.\n")


def converse_simple(
    model_id: str,
    prompt: str,
    max_tokens: int = 200,
    use_case: str = "",
    technique: str = "",
) -> dict:
    """Single-turn Converse call returning text and token usage."""
    resp = RUNTIME.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig=build_inference_config(model_id, max_tokens),
    )
    usage = resp["usage"]
    if use_case:
        log_usage(use_case, technique, model_id, usage)
    return {
        "text": resp["output"]["message"]["content"][0]["text"],
        "input_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],
    }


# ============================================================
# 2.1 Clear, specific instructions
# ============================================================

def demo_clear_instructions() -> None:
    """A vague prompt vs a directive, structured one - fewer tokens, tighter output."""
    print("--- 2.1 Clear, specific instructions ---")
    print("  Lead with the task, constrain the output shape.\n")

    vague = (
        "Could you please help me by reading the customer's email below and then "
        "telling me what you think the main issue might be, and also what their "
        "sentiment seems to be, and what action you would recommend that we take?\n\n"
        f"Customer email: {EMAIL}"
    )
    structured = (
        "Classify the email below.\n\n"
        f"Email: {EMAIL}\n\n"
        'Return JSON: {"issue": string, "sentiment": "positive"|"neutral"|"negative", "action": string}'
    )

    for model_id in [HAIKU, NOVA]:
        for label, prompt in [("VAGUE", vague), ("STRUCTURED", structured)]:
            r = converse_simple(
                model_id,
                prompt,
                use_case="2.1 Clear instructions",
                technique="Vague prompt" if label == "VAGUE" else "Structured prompt",
            )
            print(f"[{label} | {LABELS[model_id]}]  in={r['input_tokens']}  out={r['output_tokens']}")
            print(f"{r['text']}\n")

    print("  The structured prompt runs far fewer input tokens AND returns tighter,")
    print("  parseable output. Cheaper and better are the same edit here.\n")


# ============================================================
# 2.2 Few-shot examples
# ============================================================

def demo_few_shot() -> None:
    """Zero-shot vs few-shot extraction across several inputs to show consistency."""
    print("--- 2.2 Few-shot examples ---")
    print("  Examples pin the output FORMAT better than more prose.\n")

    products = [
        "Apple MacBook Pro 16-inch with M3 chip, 32GB RAM, Space Black - $2,499",
        "Sony WH-1000XM5 wireless noise cancelling headphones in silver for $348",
        "Samsung 65 inch OLED 4K Smart TV (2024 model) priced at $1,799.99",
    ]

    zero_shot = (
        "Extract product info in this exact format:\n"
        "PRODUCT|BRAND|PRICE|CATEGORY\n\n"
        'Product: "{product}"\n'
        "Output:"
    )

    # 3 boundary/diverse, class-balanced, byte-identical examples
    few_shot = (
        "Extract product info in this exact format:\n"
        "PRODUCT|BRAND|PRICE|CATEGORY\n\n"
        "Examples:\n"
        'Product: "Dell XPS 15 laptop with Intel i7, 16GB memory - $1,299"\n'
        "Output: XPS 15|Dell|1299|laptop\n"
        'Product: "Bose QuietComfort earbuds, noise cancelling, $279 retail"\n'
        "Output: QuietComfort Earbuds|Bose|279|audio\n"
        'Product: "LG 55\\" C3 OLED TV (2023) on sale for $1,196.99"\n'
        'Output: C3 OLED 55"|LG|1196.99|tv\n\n'
        "Now extract:\n"
        'Product: "{product}"\n'
        "Output:"
    )

    for p in products:
        print(f"{p[:60]}")
        for model_id in [HAIKU, NOVA]:
            outs = {}
            for label, tmpl in [("zero", zero_shot), ("few", few_shot)]:
                r = converse_simple(
                    model_id,
                    tmpl.format(product=p),
                    max_tokens=60,
                    use_case="2.2 Few-shot examples",
                    technique="Zero-shot" if label == "zero" else "Few-shot (3 examples)",
                )
                outs[label] = r["text"].strip().splitlines()[0]
            print(f"  {LABELS[model_id]}")
            print(f"    {'zero-shot:':<12}{outs['zero']}")
            print(f"    {'few-shot:':<12}{outs['few']}")

    print("\n  Few-shot pins every input to the same PRODUCT|BRAND|PRICE|CATEGORY shape")
    print("  so downstream code parses without special-casing.\n")


# ============================================================
# 2.3 Structured output - three ways
# ============================================================

def parse_json_reply(text: str) -> dict:
    """Parse a JSON reply, tolerating the markdown fence some models add anyway."""
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    return json.loads(cleaned)


def structured_prompt_based(model_id: str = SONNET) -> dict:
    """(a) Prompt-based: ask for JSON. Cheapest, no setup, but can drift."""
    resp = RUNTIME.converse(
        modelId=model_id,
        system=[
            {
                "text": "Classify the email. Reply with ONLY raw JSON, no markdown, no prose: "
                '{"issue": string, "sentiment": "positive"|"neutral"|"negative", "action": string}'
            }
        ],
        messages=[{"role": "user", "content": [{"text": CLASSIFY_EMAIL}]}],
        inferenceConfig=build_inference_config(model_id, 256),
    )
    log_usage("2.3 Structured output", "(a) Prompt-based", model_id, resp["usage"])
    text = resp["output"]["message"]["content"][0]["text"]
    return parse_json_reply(text)  # may still raise if the model added prose


def structured_prefill(model_id: str = HAIKU) -> dict:
    """(a') Prefill trick: end with an assistant '{' to force JSON onset.

    Works on NON-thinking models like Haiku 4.5 and Nova 2 Lite. Thinking-capable
    models (Claude 5 Sonnet / Opus) reject assistant-message prefill.
    """
    resp = RUNTIME.converse(
        modelId=model_id,
        system=[
            {
                "text": "Reply with ONLY raw JSON: "
                '{"issue": string, "sentiment": "positive"|"neutral"|"negative", "action": string}'
            }
        ],
        messages=[
            {"role": "user", "content": [{"text": CLASSIFY_EMAIL}]},
            {"role": "assistant", "content": [{"text": "{"}]},  # prefill forces JSON onset
        ],
        inferenceConfig={"maxTokens": 256, "temperature": 0},
    )
    log_usage("2.3 Structured output", "(a') Prefill", model_id, resp["usage"])
    return json.loads("{" + resp["output"]["message"]["content"][0]["text"])


def structured_tool_use(model_id: str = SONNET) -> tuple[dict, int]:
    """(b) Tool use: schema-enforced via a forced tool call. Schema rides along."""
    tool = {
        "toolSpec": {
            "name": "classify_email",
            "description": "Classify a customer email.",
            "inputSchema": {"json": SCHEMA},  # a dict here, unlike outputConfig
        }
    }
    strict = "claude" in model_id
    if strict:
        # Grammar-enforce the tool input against the schema. Claude only: Nova 2
        # Lite rejects the strict field, so its tool call is schema-guided, not enforced.
        tool["toolSpec"]["strict"] = True
    resp = RUNTIME.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": CLASSIFY_EMAIL}]}],
        toolConfig={
            "tools": [tool],
            "toolChoice": {"tool": {"name": "classify_email"}},  # force the call
        },
        inferenceConfig={"maxTokens": 256},  # Claude 5 Sonnet: no temperature
    )
    technique = "(b) Strict tool use" if strict else "(b) Tool use"
    log_usage("2.3 Structured output", technique, model_id, resp["usage"])
    data = next(
        c["toolUse"]["input"] for c in resp["output"]["message"]["content"] if "toolUse" in c
    )
    return data, resp["usage"]["inputTokens"]


def structured_native_json_schema() -> tuple[dict, int]:
    """(c) Native structured output: constrained decoding guarantees schema-valid text.

    On Converse the schema must be a JSON STRING (json.dumps). Native structured
    output is available on bedrock-runtime. Every object needs
    additionalProperties: false.

    Runs on Claude 5 Sonnet (no temperature). Parse by selecting the text block
    by key, since content[0] is not guaranteed to be text on thinking-capable
    models.
    """
    resp = RUNTIME.converse(
        modelId=SONNET,
        messages=[{"role": "user", "content": [{"text": CLASSIFY_EMAIL}]}],
        inferenceConfig={"maxTokens": 512},  # Claude 5 Sonnet: no temperature
        outputConfig={
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "name": "classify_email",
                        "description": "Classify a customer email.",
                        "schema": json.dumps(SCHEMA),  # a JSON string, not a dict
                    }
                },
            }
        },
    )
    log_usage("2.3 Structured output", "(c) Native JSON schema", SONNET, resp["usage"])
    text = next(b["text"] for b in resp["output"]["message"]["content"] if "text" in b)
    return json.loads(text), resp["usage"]["inputTokens"]


def demo_structured_output() -> None:
    """Compare the three structured-output approaches on the same email."""
    print("--- 2.3 Structured output - three ways ---")
    print("  If downstream code consumes the output, constrain it, don't just ask.\n")

    data = structured_prompt_based()
    print("(a) prompt-based (lowest cost, no guarantee):")
    print(json.dumps(data, indent=2))

    data = structured_prompt_based(NOVA)
    print("\n(a) prompt-based on Nova 2 Lite:")
    print(json.dumps(data, indent=2))

    data = structured_prefill(HAIKU)
    print("\n(a') Haiku prefill (non-thinking models only):")
    print(json.dumps(data, indent=2))

    data = structured_prefill(NOVA)
    print("\n(a') Nova 2 Lite prefill:")
    print(json.dumps(data, indent=2))

    data, in_tokens = structured_tool_use()
    print("\n(b) tool-use (schema-enforced via the call):")
    print(json.dumps(data, indent=2))
    print(f"    input tokens: {in_tokens} (higher - the schema rides along)")

    data, in_tokens = structured_tool_use(NOVA)
    print("\n(b) tool-use on Nova 2 Lite (no strict mode, so schema-guided only):")
    print(json.dumps(data, indent=2))
    print(f"    input tokens: {in_tokens}")

    data, in_tokens = structured_native_json_schema()
    print("\n(c) native JSON schema (API-validated, the default for data extraction):")
    print(json.dumps(data, indent=2))
    print(f"    input tokens: {in_tokens} (lower than tool use - no tool definition)")
    print("    Nova 2 Lite: not available - it rejects outputConfig, so use prompt-based,")
    print("    prefill, or tool use there.")

    print("\n  Use native structured output for data shaping; tool use when the model")
    print("  is actually calling a function. The cost is similar; the difference is")
    print("  whether your pipeline breaks on the one malformed response.\n")


# ============================================================
# Main
# ============================================================

def main():
    demo_clear_instructions()
    demo_few_shot()
    demo_structured_output()
    print_usage_table()

    print("--- Done ---")
    print("  Prompt design cuts wasted input tokens on every call and makes output")
    print("  reliable. Reach for few-shot to pin format, and constrain output when")
    print("  code consumes it.")


if __name__ == "__main__":
    main()
