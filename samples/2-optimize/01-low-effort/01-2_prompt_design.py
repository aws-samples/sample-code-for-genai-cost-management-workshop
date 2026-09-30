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

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude models on Amazon Bedrock
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


def converse_simple(model_id: str, prompt: str, max_tokens: int = 200) -> dict:
    """Single-turn Converse call returning text and token usage."""
    resp = RUNTIME.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig=build_inference_config(model_id, max_tokens),
    )
    usage = resp["usage"]
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

    for label, prompt in [("VAGUE", vague), ("STRUCTURED", structured)]:
        r = converse_simple(HAIKU, prompt)
        print(f"[{label}]  in={r['input_tokens']}  out={r['output_tokens']}")
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
        outs = {}
        for label, tmpl in [("zero", zero_shot), ("few", few_shot)]:
            r = converse_simple(HAIKU, tmpl.format(product=p), max_tokens=60)
            outs[label] = r["text"].strip().splitlines()[0]
        print(f"{p[:60]}")
        print(f"  {'zero-shot:':<12}{outs['zero']}")
        print(f"  {'few-shot:':<12}{outs['few']}")

    print("\n  Few-shot pins every input to the same PRODUCT|BRAND|PRICE|CATEGORY shape")
    print("  so downstream code parses without special-casing.\n")


# ============================================================
# 2.3 Structured output - three ways
# ============================================================

def structured_prompt_based() -> dict:
    """(a) Prompt-based: ask for JSON. Cheapest, no setup, but can drift."""
    resp = RUNTIME.converse(
        modelId=SONNET,
        system=[
            {
                "text": "Classify the email. Reply with ONLY raw JSON, no markdown, no prose: "
                '{"issue": string, "sentiment": "positive"|"neutral"|"negative", "action": string}'
            }
        ],
        messages=[{"role": "user", "content": [{"text": CLASSIFY_EMAIL}]}],
        inferenceConfig={"maxTokens": 256},  # Claude 5 Sonnet: no temperature
    )
    text = resp["output"]["message"]["content"][0]["text"]
    return json.loads(text)  # may raise if the model added prose or fences


def structured_prefill_haiku() -> dict:
    """(a') Prefill trick: end with an assistant '{' to force JSON onset.

    Works on NON-thinking models like Haiku 4.5. Thinking-capable models
    (Claude 5 Sonnet / Opus) reject assistant-message prefill.
    """
    resp = RUNTIME.converse(
        modelId=HAIKU,
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
    return json.loads("{" + resp["output"]["message"]["content"][0]["text"])


def structured_tool_use() -> tuple[dict, int]:
    """(b) Tool use: schema-enforced via a forced tool call. Schema rides along."""
    tool = {
        "toolSpec": {
            "name": "classify_email",
            "description": "Classify a customer email.",
            "inputSchema": {"json": SCHEMA},
        }
    }
    resp = RUNTIME.converse(
        modelId=SONNET,
        messages=[{"role": "user", "content": [{"text": CLASSIFY_EMAIL}]}],
        toolConfig={
            "tools": [tool],
            "toolChoice": {"tool": {"name": "classify_email"}},  # force the call
        },
        inferenceConfig={"maxTokens": 256},  # Claude 5 Sonnet: no temperature
    )
    data = next(
        c["toolUse"]["input"] for c in resp["output"]["message"]["content"] if "toolUse" in c
    )
    return data, resp["usage"]["inputTokens"]


def structured_native_json_schema() -> tuple[dict, int]:
    """(c) Native structured output: constrained decoding guarantees schema-valid text.

    On Converse the schema must be a JSON STRING (json.dumps). Native structured
    output is available on bedrock-runtime (not bedrock-mantle). Every object
    needs additionalProperties: false.

    NOTE: this demo runs on Haiku 4.5. Native structured output (the outputConfig
    path) is not accepted by Claude 5 Sonnet, which returns a ValidationException
    ("output_config.format: Extra inputs are not permitted"). Use tool use (b) on
    the thinking-capable Claude 5 models when you need a schema guarantee there.
    """
    resp = RUNTIME.converse(
        modelId=HAIKU,
        messages=[{"role": "user", "content": [{"text": CLASSIFY_EMAIL}]}],
        inferenceConfig={"maxTokens": 256, "temperature": 0},
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
    data = json.loads(resp["output"]["message"]["content"][0]["text"])
    return data, resp["usage"]["inputTokens"]


def demo_structured_output() -> None:
    """Compare the three structured-output approaches on the same email."""
    print("--- 2.3 Structured output - three ways ---")
    print("  If downstream code consumes the output, constrain it, don't just ask.\n")

    data = structured_prompt_based()
    print("(a) prompt-based (lowest cost, no guarantee):")
    print(json.dumps(data, indent=2))

    data = structured_prefill_haiku()
    print("\n(a') Haiku prefill (non-thinking models only):")
    print(json.dumps(data, indent=2))

    data, in_tokens = structured_tool_use()
    print("\n(b) tool-use (schema-enforced via the call):")
    print(json.dumps(data, indent=2))
    print(f"    input tokens: {in_tokens} (higher - the schema rides along)")

    data, in_tokens = structured_native_json_schema()
    print("\n(c) native JSON schema on Haiku (API-validated, the default for data extraction):")
    print(json.dumps(data, indent=2))
    print(f"    input tokens: {in_tokens} (lower than tool use - no tool definition)")

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

    print("--- Done ---")
    print("  Prompt design cuts wasted input tokens on every call and makes output")
    print("  reliable. Reach for few-shot to pin format, and constrain output when")
    print("  code consumes it.")


if __name__ == "__main__":
    main()
