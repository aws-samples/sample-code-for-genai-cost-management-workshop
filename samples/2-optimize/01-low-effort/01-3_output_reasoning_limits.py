"""
Low-Effort Optimization - Lever 03: Output and Reasoning Limits

Do not default to maximum reasoning or output tokens. Tune both to task
complexity. The defaults favor "won't surprise you," not "fast, cheap,
predictable," and every extra output or reasoning token is billed at the
(higher) output rate and adds latency.

You will learn how to:
- See how max_tokens sets the TPM quota RESERVED up front (affects concurrency)
- Use stop_sequences to halt generation the instant a terminator appears
  (Nova 2 Lite keeps the matched marker in its output and reports end_turn)
- See how temperature controls determinism vs variation
- Tune reasoning effort (low / medium / high) to task complexity and watch
  reasoning (output) tokens and latency scale with it
- Run every lever on Amazon Nova 2 Lite as well, through the same Converse API

Key points:
- max_tokens is the load-bearing one. On Bedrock it reserves quota up front
  (output burns at 5x on Haiku 4.5, 10x on Claude 5 Sonnet, 1x on Nova), refunded
  after. A too-large value throttles concurrency under load. Per-call cost is
  unchanged (you pay for tokens generated); concurrency is not. Right-size it.
- temperature: 0 for classification/extraction/structured output; 0.3-0.7 for
  generation; higher for brainstorming. Deprecated on Claude 5; Haiku 4.5 and
  Nova 2 Lite still accept it.
- stop_sequences: a free latency win when a real terminator exists.
- Adaptive thinking: you set an effort level and the model decides, per
  request, how much to think. Reasoning tokens are billed as OUTPUT tokens, and
  they count against max_tokens. The practical pattern is to step DOWN from the
  default (high) once an eval shows a lower level holds quality for your task.
- Placement matters: effort goes in a SEPARATE output_config object inside
  additionalModelRequestFields, NOT inside thinking. Putting it inside returns
  a ValidationException.
- Model support for adaptive thinking: Claude 5 Opus / Sonnet. Not Haiku 4.5.
- Nova 2 Lite reasons only when enabled, through a different field:
  additionalModelRequestFields reasoningConfig with maxReasoningEffort (low,
  medium, high). At high, maxTokens must be left unset.

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude Haiku 4.5, Claude 5 Sonnet and Amazon Nova 2 Lite on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import time
import boto3

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

RUNTIME = boto3.client("bedrock-runtime", region_name=REGION)

HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# Adaptive thinking requires a thinking-capable model.
SONNET = "global.anthropic.claude-sonnet-5"
NOVA = "global.amazon.nova-2-lite-v1:0"

# Readable labels for the output tables.
LABELS = {HAIKU: "Haiku 4.5", SONNET: "Sonnet 5", NOVA: "Nova 2 Lite"}

# Effort levels to sweep in the reasoning demo (default on the model is high).
EFFORT_LEVELS = ["low", "medium", "high"]

# A simple task: one obvious answer, little to reason about.
SIMPLE_TASK = (
    "Classify the sentiment of this review as positive, neutral, or negative. "
    "Reply with one word.\n\n"
    "'Absolutely love these headphones, the sound is fantastic.'"
)

# A complex task: ambiguous policy trade-off where more thinking can help.
COMPLEX_TASK = (
    "A customer bought a laptop a couple of months ago and now the battery "
    "drains in under an hour. They're out of the 30-day return window but within "
    "the 1-year manufacturer warranty. They're frustrated and asking for a full "
    "refund. What should we do, and how should we phrase the reply? Reason it "
    "through, then give the recommended action and a short customer message."
)

# Output-token burndown rate for the TPM quota: 1 output token consumes this many
# tokens of quota. Claude 4.7 and older (including Haiku 4.5) burn at 5x, the
# newest Claude models (such as Sonnet 5) at 10x, and most other models, Nova
# included, at 1x. Verify at
# https://docs.aws.amazon.com/bedrock/latest/userguide/quotas-token-burndown.html
OUTPUT_BURNDOWN_RATE = {HAIKU: 5, NOVA: 1}

EXAMPLE_TASK = (
    "Summarize this product review in one sentence, then rate its sentiment "
    "as positive, neutral, or negative:\n\n"
    "'Delivery was fast and the packaging was solid, but the device stopped "
    "working after two weeks and support was slow to respond.'"
)

# Results collected as the demos run and printed as one comparison table at the end.
# Each entry: (lever, setting, output_tokens, latency_ms or None, note)
RESULTS: list[tuple[str, str, int, int | None, str]] = []

EMAIL = (
    "Hi, the wireless headphones I ordered last week arrived with a cracked case "
    "and the left earbud won't charge. I'd like a replacement before the weekend."
)


# ============================================================
# 1. max_tokens - TPM reservation
# ============================================================

def demo_max_tokens() -> None:
    """Same output, very different TPM reserved: max_tokens=4096 vs 200."""
    print("--- 1. max_tokens: identical output, different reserved TPM ---")
    print(f"INPUT:\n  {EXAMPLE_TASK}\n")

    rows = []
    for model_id in [HAIKU, NOVA]:
        answer = None
        for mt in [4096, 200]:
            resp = RUNTIME.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": EXAMPLE_TASK}]}],
                inferenceConfig={"maxTokens": mt, "temperature": 0},
            )
            answer = resp["output"]["message"]["content"][0]["text"]
            out_tokens = resp["usage"]["outputTokens"]
            reserved_tpm = resp["usage"]["inputTokens"] + (mt * OUTPUT_BURNDOWN_RATE[model_id])
            rows.append((LABELS[model_id], mt, out_tokens, reserved_tpm))
            RESULTS.append(
                (
                    "max_tokens",
                    f"{LABELS[model_id]} max_tokens={mt}",
                    out_tokens,
                    None,
                    f"reserves {reserved_tpm:,} TPM ({OUTPUT_BURNDOWN_RATE[model_id]}x burndown)",
                )
            )
        print(f"OUTPUT {LABELS[model_id]} (identical regardless of max_tokens):")
        print(f"  {answer}\n")

    print(f"{'model':<12} {'max_tokens':>10}  {'out':>4}  {'reserved TPM':>13}")
    for label, mt, out, tpm in rows:
        print(f"{label:<12} {mt:>10}  {out:>4}  {tpm:>13,}")
    print(
        "\n  Same output, but a large max_tokens reserves far more TPM than a tight one"
        "\n  - that is the concurrency you silently give up under load. The burndown rate"
        "\n  multiplies it: Haiku 4.5 burns output at 5x, Nova 2 Lite at 1x.\n"
    )


# ============================================================
# 2. stop_sequences - early termination
# ============================================================

def demo_stop_sequences() -> None:
    """Generation halts the instant a known terminator appears - fewer tokens, faster."""
    print("--- 2. stop_sequences: halt on a known boundary ---\n")

    prompt = (
        "Categorize the support email into one word (shipping, product, billing, other). "
        "Reply with the category, then the marker END on its own line, then a "
        "two-sentence explanation.\n\n"
        f"Email: {EMAIL}\n\n"
        "Category: "
    )
    # Only the category is needed. The model writes the explanation anyway unless
    # generation halts at the marker.

    for model_id in [HAIKU, NOVA]:
        for use_stop in [False, True]:
            cfg = {"maxTokens": 200, "temperature": 0}
            if use_stop:
                cfg["stopSequences"] = ["END"]
            t0 = time.time()
            resp = RUNTIME.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig=cfg,
            )
            text = resp["output"]["message"]["content"][0]["text"]
            out_tokens = resp["usage"]["outputTokens"]
            ms = int((time.time() - t0) * 1000)
            print(f"{LABELS[model_id]}  stop_sequences={use_stop}")
            print(f"  output: {text!r}")
            print(f"  out_tokens={out_tokens}  ms={ms}  stop_reason={resp['stopReason']}\n")
            RESULTS.append(
                (
                    "stop_sequences",
                    f"{LABELS[model_id]} stop_sequences={'on' if use_stop else 'off'}",
                    out_tokens,
                    ms,
                    f"stop_reason={resp['stopReason']}, output {text.strip()[:30]!r}",
                )
            )


# ============================================================
# 3. temperature - determinism vs variation
# ============================================================

def demo_temperature() -> None:
    """Same creative prompt 3x per setting to show consistency vs variation."""
    print("--- 3. temperature: determinism vs variation ---\n")

    tagline = "Write a one-sentence tagline for a coffee shop."

    for model_id in [HAIKU, NOVA]:
        for temp in [0.0, 0.7, 1.0]:
            print(f"{LABELS[model_id]}  temperature={temp}")
            texts, tokens = [], []
            for run in range(3):
                resp = RUNTIME.converse(
                    modelId=model_id,
                    messages=[{"role": "user", "content": [{"text": tagline}]}],
                    inferenceConfig={"maxTokens": 40, "temperature": temp},
                )
                text = resp["output"]["message"]["content"][0]["text"].strip()
                texts.append(text)
                tokens.append(resp["usage"]["outputTokens"])
                print(f"  run {run + 1}: {text}")
            print()
            RESULTS.append(
                (
                    "temperature",
                    f"{LABELS[model_id]} temperature={temp}",
                    round(sum(tokens) / len(tokens)),
                    None,
                    f"{len(set(texts))} of {len(texts)} runs gave a distinct answer",
                )
            )

    print("  At temperature=0 the runs are near-identical; at 0.7-1.0 they diverge -")
    print("  which is why classification/extraction use 0 and brainstorming uses higher.")
    print("  (Claude 5 Sonnet dropped the parameter; Haiku 4.5 and Nova 2 Lite keep it.)\n")


# ============================================================
# 4. Reasoning effort - reasoning limits
# ============================================================

def converse_with_effort(model_id: str, prompt: str, effort: str, max_tokens: int = 2048) -> dict:
    """Call Converse with reasoning at the given effort level.

    Claude 5 Sonnet: adaptive thinking. effort goes in a SEPARATE output_config
    object inside additionalModelRequestFields - not inside the thinking object.

    Nova 2 Lite: reasoning is OFF unless you enable it with reasoningConfig and a
    maxReasoningEffort. At "high", maxTokens must be left unset, so the sample
    omits it there. Reasoning text comes back redacted, but is still billed.
    """
    inference_config = {"maxTokens": max_tokens}
    if model_id == NOVA:
        if effort == "off":
            fields = None
        else:
            fields = {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": effort}}
            if effort == "high":
                inference_config = {}
    else:
        fields = {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }

    kwargs = {"additionalModelRequestFields": fields} if fields else {}
    if inference_config:
        kwargs["inferenceConfig"] = inference_config

    t0 = time.time()
    resp = RUNTIME.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        **kwargs,
    )

    # Separate the reasoning block from the final answer text. Select blocks by
    # key, not content[0]: reasoning-capable models can lead with a reasoning block.
    reasoning, answer = "", ""
    for block in resp["output"]["message"]["content"]:
        if "reasoningContent" in block:
            reasoning = block["reasoningContent"].get("reasoningText", {}).get("text", "")
        elif "text" in block:
            answer = block["text"]

    return {
        "reasoning": reasoning,
        "answer": answer,
        "output_tokens": resp["usage"]["outputTokens"],  # includes reasoning tokens
        "latency_ms": int((time.time() - t0) * 1000),
        "stop_reason": resp["stopReason"],
    }


def demo_reasoning_effort() -> None:
    """Sweep effort on a simple task and a complex one: match effort to complexity."""
    print("--- 4. Reasoning effort: tune reasoning to task complexity ---\n")

    tasks = [("simple", SIMPLE_TASK), ("complex", COMPLEX_TASK)]
    # Nova 2 Lite reasons only when asked, so add an "off" baseline for it.
    plans = [(SONNET, EFFORT_LEVELS), (NOVA, ["off"] + EFFORT_LEVELS)]
    rows = []
    for name, prompt in tasks:
        print(f"TASK ({name}):\n  {prompt}\n")
        for model_id, levels in plans:
            for effort in levels:
                r = converse_with_effort(model_id, prompt, effort)
                rows.append(
                    (name, LABELS[model_id], effort, r["output_tokens"], r["latency_ms"], r["stop_reason"])
                )
                RESULTS.append(
                    (
                        "effort",
                        f"{LABELS[model_id]} {name} task, effort={effort}",
                        r["output_tokens"],
                        r["latency_ms"],
                        f"stop_reason={r['stop_reason']}",
                    )
                )
                print(
                    f"  {LABELS[model_id]:<12} effort={effort:<6} "
                    f"output_tokens={r['output_tokens']:>5}  ms={r['latency_ms']:>6}"
                )
                if r["reasoning"]:
                    trace = r["reasoning"].strip().replace("\n", " ")
                    print(f"    reasoning trace ({len(r['reasoning'])} chars): {trace[:200]}...")
                print(f"    answer: {r['answer'].strip()[:200]!r}")
        print()

    print("Summary (output tokens include reasoning, billed at the output rate):")
    print(f"{'task':<9} {'model':<12} {'effort':<8} {'out_tokens':>11} {'ms':>8}  stop_reason")
    for name, label, effort, out_tokens, ms, stop in rows:
        print(f"{name:<9} {label:<12} {effort:<8} {out_tokens:>11} {ms:>8}  {stop}")
    print(
        "\n  Effort sets how much the model may think; token counts vary run to run, so"
        "\n  read the trend over many samples, not one call. On Claude 5 Sonnet a simple"
        "\n  task costs about the same at every level, while a complex task spends"
        "\n  hundreds of output tokens and many seconds. On Nova 2 Lite reasoning is off"
        "\n  by default and opt-in: turning it on for the simple task turns a 3-token"
        "\n  answer into hundreds, and at high effort into thousands of billed tokens."
        "\n  Start at the default, then step down to the lowest level that holds quality"
        "\n  on your eval set, and only enable reasoning where the task needs it."
        "\n  Reasoning tokens count against max_tokens - if stop_reason is max_tokens,"
        "\n  the answer was cut off, so size max_tokens for reasoning plus answer.\n"
    )


# ============================================================
# Comparison table
# ============================================================

def print_comparison_table() -> None:
    """Print every setting that was tried, side by side."""
    print("--- Comparison: every setting tried ---")
    print(f"{'Lever':<15} {'Setting':<42} {'Out tokens':>10} {'ms':>7}  Result")
    last = None
    for lever, setting, out_tokens, ms, note in RESULTS:
        if last is not None and lever != last:
            print()
        last = lever
        ms_text = f"{ms}" if ms is not None else "-"
        print(f"{lever:<15} {setting:<42} {out_tokens:>10} {ms_text:>7}  {note}")
    print(
        "\n  Out tokens for temperature is the average of 3 runs. A dash means latency"
        "\n  was not the point of that row. Read each lever on its own rows: the same"
        "\n  answer at max_tokens=200 reserves far less quota than at 4096, a stop"
        "\n  sequence cuts the explanation you did not need, temperature trades"
        "\n  determinism for variety, and reasoning effort only pays on the complex task.\n"
    )


# ============================================================
# Main
# ============================================================

def main():
    print("Output and reasoning limits")
    print(f"  Region         : {REGION}")
    print(f"  Haiku 4.5      : {HAIKU}")
    print(f"  Nova 2 Lite    : {NOVA}")
    print(f"  Claude 5 Sonnet: {SONNET}")
    print(f"  Effort levels  : {', '.join(EFFORT_LEVELS)}\n")

    demo_max_tokens()
    demo_stop_sequences()
    demo_temperature()
    demo_reasoning_effort()
    print_comparison_table()

    print("--- Done ---")
    print("  Right-size max_tokens (set it tight, raise only on truncation), use")
    print("  stop_sequences when a real terminator exists, match temperature to the")
    print("  task, and step reasoning effort down to the lowest level that holds quality.")


if __name__ == "__main__":
    main()
