"""
Low-Effort Optimization - Lever 06: Evaluation (LLM-as-a-Judge)

Every other lever in this tier says "confirm on an eval set". This sample shows
the smallest useful version of that eval: a tiny test set, a scoring rubric, and
a second model (the judge) that grades each answer against the rubric.

You will learn how to:
- Build a small held-out eval set with the inputs your task really sees
- Write a rubric so the judge scores specific criteria, not "overall quality"
- Get machine-readable scores from the judge with native structured output
- Sanity-check the judge with a deliberately bad answer before trusting it
- Use the scores to decide whether the cheaper model is good enough

The rule:
- Score before you switch. Moving from Sonnet to Haiku is only a saving if the
  quality bar still holds - the eval is what proves it.
- Judge on a rubric. Ask for per-criterion scores and a short reason; the reason
  is written BEFORE the scores so the judge reasons first and scores second.
- Calibrate the judge. If it cannot fail an obviously wrong answer, its passes
  mean nothing.
- The judge costs money too. Run it on a sample of traffic or in CI, not on every
  production request, and use the cheapest judge that is still reliable.

This sample hand-rolls the judge so you can see every moving part. In a real
project, consider the managed offerings: Amazon Bedrock Evaluations (model
evaluation jobs that use a judge model) and Amazon Bedrock AgentCore Evaluations
(LLM-as-a-judge scoring of agent traces). See the low-effort README for details.

Known judge biases to keep in mind: judges tend to favor longer answers, and a
model tends to favor its own style (self-preference). Here Claude 5 Sonnet judges
both candidates, so treat the Sonnet-vs-Haiku gap with a little caution and
spot-check a few judgments by hand.

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude models on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import json
import os
import time

import boto3

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

RUNTIME = boto3.client("bedrock-runtime", region_name=REGION)

# Allowed workshop models (Global cross-region inference profiles)
HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET = "global.anthropic.claude-sonnet-5"

# Candidates under test: the cheap model we hope to use, and the incumbent.
CANDIDATES = [HAIKU, SONNET]

# The judge model. Edit this to try a different judge.
JUDGE = SONNET

# A response passes when its average rubric score is at or above this (1-5 scale).
PASS_THRESHOLD = 4.0

# Pricing per 1M tokens (USD). Illustrative workshop values - always verify
# current pricing at https://aws.amazon.com/bedrock/pricing/
PRICING = {
    HAIKU: {"input": 1.00, "output": 5.00},
    SONNET: {"input": 3.00, "output": 15.00},
}

# The task being evaluated. Candidates receive this instruction plus one review.
TASK_PROMPT = (
    "Summarize this product review in one sentence, then rate its sentiment as "
    "positive, neutral, or negative. Format: '<summary> Sentiment: <label>'.\n\n"
    "Review: {review}"
)

# The eval set. In a real project this is 20-100 examples pulled from your own
# traffic, including the hard cases. Five are enough to show the pattern.
EVAL_SET = [
    "Delivery was fast and the packaging was solid, but the device stopped "
    "working after two weeks and support was slow to respond.",
    "Absolutely love this blender. It crushes ice in seconds, it is easy to "
    "clean, and it has been going strong for a year now.",
    "The jacket looks fine and fits as described. Nothing special, nothing "
    "wrong. Does the job for light rain.",
    "Oh great, another charger that 'fast charges' my phone in only six hours. "
    "Really impressive engineering. Would not buy again.",
    "Battery life is excellent and the screen is gorgeous, but the speakers are "
    "tinny and the keyboard flexes when I type. Returning it.",
]

# Calibration case: a deliberately wrong answer (invents a fact, wrong sentiment).
# A trustworthy judge must score this low.
CALIBRATION_REVIEW = EVAL_SET[0]
CALIBRATION_ANSWER = (
    "The customer loved the premium titanium casing and the free replacement "
    "they received. Sentiment: positive"
)

# The rubric. Each criterion is scored 1 (poor) to 5 (excellent).
RUBRIC = """You are a strict evaluator of product-review summaries.

You are given the original REVIEW and a candidate RESPONSE. The response was
asked to give a one-sentence summary followed by 'Sentiment: <positive|neutral|negative>'.

Score each criterion from 1 (poor) to 5 (excellent):
- faithfulness: every claim in the summary is supported by the review; nothing is invented.
- completeness: the summary captures the main points, including both good and bad
  aspects when the review is mixed.
- sentiment: the sentiment label matches the review's overall tone (sarcasm counts
  as its literal opposite).
- format: one summary sentence followed by 'Sentiment: <label>', no extra text.
  Capitalization of the label and a trailing period are acceptable.

Write a rationale of at most 30 words FIRST, then the scores."""

# Native structured output schema for the judge. Every object needs
# additionalProperties: false. The rationale comes first so the judge reasons
# before it scores.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "faithfulness": {"type": "integer"},
        "completeness": {"type": "integer"},
        "sentiment": {"type": "integer"},
        "format": {"type": "integer"},
    },
    "required": ["rationale", "faithfulness", "completeness", "sentiment", "format"],
    "additionalProperties": False,
}

CRITERIA = ["faithfulness", "completeness", "sentiment", "format"]


# ============================================================
# Helper Functions
# ============================================================

def calculate_cost(input_tokens: int, output_tokens: int, model_id: str) -> float:
    """Return the dollar cost of a single request for the given model."""
    prices = PRICING[model_id]
    return (input_tokens / 1_000_000) * prices["input"] + (
        output_tokens / 1_000_000
    ) * prices["output"]


def short_name(model_id: str) -> str:
    """Return a readable model label for tables."""
    return "haiku-4-5" if "haiku" in model_id else "sonnet-5"


def first_text(resp: dict) -> str:
    """Return the first text block, since content[0] may be a reasoning block."""
    return next(b["text"] for b in resp["output"]["message"]["content"] if "text" in b)


def build_inference_config(model_id: str, max_tokens: int) -> dict:
    """Build inferenceConfig, omitting temperature for models that reject it.

    Claude 5 models (Sonnet 5, Opus 5) deprecated the temperature parameter and
    return a ValidationException if it is supplied. Haiku 4.5 still accepts it.
    """
    cfg = {"maxTokens": max_tokens}
    if "claude-sonnet-5" not in model_id and "claude-opus-5" not in model_id:
        cfg["temperature"] = 0
    return cfg


def generate(model_id: str, review: str) -> dict:
    """Run the task on one review and return text, tokens, latency, and cost."""
    t0 = time.time()
    resp = RUNTIME.converse(
        modelId=model_id,
        messages=[
            {"role": "user", "content": [{"text": TASK_PROMPT.format(review=review)}]}
        ],
        inferenceConfig=build_inference_config(model_id, 300),
    )
    usage = resp["usage"]
    return {
        "text": first_text(resp).strip(),
        "latency_ms": int((time.time() - t0) * 1000),
        "cost": calculate_cost(usage["inputTokens"], usage["outputTokens"], model_id),
    }


def judge(review: str, answer: str) -> dict:
    """Grade one answer against the rubric and return scores, average, and cost."""
    prompt = f"REVIEW:\n{review}\n\nRESPONSE:\n{answer}"
    resp = RUNTIME.converse(
        modelId=JUDGE,
        system=[{"text": RUBRIC}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig=build_inference_config(JUDGE, 600),
        outputConfig={
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "name": "review_summary_grade",
                        "description": "Rubric scores for one candidate response.",
                        "schema": json.dumps(JUDGE_SCHEMA),  # a JSON string
                    }
                },
            }
        },
    )
    grade = json.loads(first_text(resp))
    usage = resp["usage"]
    grade["average"] = sum(grade[c] for c in CRITERIA) / len(CRITERIA)
    grade["cost"] = calculate_cost(usage["inputTokens"], usage["outputTokens"], JUDGE)
    return grade


def score_line(grade: dict) -> str:
    """Format the per-criterion scores on one line."""
    parts = " ".join(f"{c[:5]}={grade[c]}" for c in CRITERIA)
    return f"{parts} avg={grade['average']:.2f}"


# ============================================================
# Demo
# ============================================================

def calibrate_judge() -> None:
    """Show the judge failing a deliberately wrong answer before we trust it."""
    print("--- Step 1: calibrate the judge with a known-bad answer ---")
    print(f"  Review : {CALIBRATION_REVIEW}")
    print(f"  Answer : {CALIBRATION_ANSWER}")
    grade = judge(CALIBRATION_REVIEW, CALIBRATION_ANSWER)
    print(f"  Scores : {score_line(grade)}")
    print(f"  Reason : {grade['rationale']}")
    verdict = "PASS" if grade["average"] >= PASS_THRESHOLD else "FAIL"
    print(f"  Verdict: {verdict} (expected FAIL)\n")
    if verdict == "PASS":
        print("  WARNING: the judge passed a wrong answer - fix the rubric first.\n")


def evaluate_candidates() -> dict:
    """Generate and judge every candidate over the eval set; return the results."""
    print("--- Step 2: generate and judge each candidate over the eval set ---")
    results = {
        m: {"scores": [], "passes": 0, "gen_cost": 0.0, "judge_cost": 0.0, "ms": 0}
        for m in CANDIDATES
    }

    for i, review in enumerate(EVAL_SET, start=1):
        print(f"\n[{i}/{len(EVAL_SET)}] {review}")
        for model_id in CANDIDATES:
            out = generate(model_id, review)
            grade = judge(review, out["text"])
            r = results[model_id]
            r["scores"].append(grade["average"])
            r["passes"] += grade["average"] >= PASS_THRESHOLD
            r["gen_cost"] += out["cost"]
            r["judge_cost"] += grade["cost"]
            r["ms"] += out["latency_ms"]
            print(f"  {short_name(model_id):<10} {out['text']}")
            print(f"  {'':<10} {score_line(grade)}")
            print(f"  {'':<10} judge: {grade['rationale']}")
    return results


def print_summary(results: dict) -> None:
    """Print quality vs cost per candidate and the decision the numbers support."""
    n = len(EVAL_SET)
    print("\n--- Step 3: quality vs cost ---")
    print(
        f"{'Model':<12} {'Avg score':>9} {'Pass':>7} {'Gen $':>10} "
        f"{'Judge $':>10} {'Avg ms':>8}"
    )
    avg = {}
    for model_id, r in results.items():
        avg[model_id] = sum(r["scores"]) / n
        print(
            f"{short_name(model_id):<12} {avg[model_id]:>9.2f} "
            f"{r['passes']:>3}/{n:<3} {r['gen_cost']:>10.6f} "
            f"{r['judge_cost']:>10.6f} {r['ms'] // n:>8}"
        )

    judge_total = sum(r["judge_cost"] for r in results.values())
    gen_total = sum(r["gen_cost"] for r in results.values())
    print(f"\n  Judging cost ${judge_total:.6f} vs ${gen_total:.6f} to generate the "
          "answers - the eval itself is not free, so run it on a sample.")

    cheap, incumbent = CANDIDATES[0], CANDIDATES[-1]
    saving = 1 - results[cheap]["gen_cost"] / results[incumbent]["gen_cost"]
    gap = avg[incumbent] - avg[cheap]
    print(f"\n  {short_name(cheap)} costs {saving:.0%} less to generate than "
          f"{short_name(incumbent)}, with a score gap of {gap:+.2f}.")
    if avg[cheap] >= PASS_THRESHOLD and gap <= 0.25:
        print(f"  Decision: quality holds (>= {PASS_THRESHOLD}) - switch to "
              f"{short_name(cheap)} and keep monitoring.")
    else:
        print(f"  Decision: {short_name(cheap)} falls short of the bar - stay on "
              f"{short_name(incumbent)}, or tighten the prompt and re-run.")


# ============================================================
# Main
# ============================================================

def main():
    print("LLM-as-a-Judge evaluation")
    print(f"  Region     : {REGION}")
    print(f"  Candidates : {', '.join(short_name(m) for m in CANDIDATES)}")
    print(f"  Judge      : {short_name(JUDGE)}")
    print(f"  Eval set   : {len(EVAL_SET)} reviews, pass at avg >= {PASS_THRESHOLD}\n")

    calibrate_judge()
    results = evaluate_candidates()
    print_summary(results)

    print("\n--- Done ---")
    print("  Next steps:")
    print("  1. Replace EVAL_SET with 20-100 real examples, including the hard cases")
    print("  2. Hand-check a few judge rationales to confirm they agree with you")
    print("  3. Re-run this eval whenever you change the model, prompt, or parameters")
    print("  4. Run it on a sample of production traffic, not every request")
    print("  5. Move from this hand-rolled judge to a managed AWS offering:")
    print("     - Amazon Bedrock Evaluations (model evaluation job with a judge model):")
    print("       for comparing models or prompts on a JSONL dataset in S3, with built-in")
    print("       and custom metrics, and per-response scores and explanations")
    print("     - Amazon Bedrock AgentCore Evaluations: for agents; scores OpenTelemetry")
    print("       traces with built-in or custom LLM-as-a-judge evaluators, on-demand in")
    print("       CI/CD or online against live traffic")


if __name__ == "__main__":
    main()
