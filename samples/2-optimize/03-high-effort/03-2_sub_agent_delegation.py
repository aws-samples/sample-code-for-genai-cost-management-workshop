"""
High-Effort Optimization - Lever 14: Sub-Agent Delegation

A single agent that does everything carries every tool result, every search hit,
and every intermediate step in one context window. That window grows fast and is
re-sent every turn - by turn 50 the agent spends most of its budget re-reading
its own past work, getting slower and worse.

Sub-agent delegation breaks this with an orchestrator-worker split: a LEAD plans
and answers; a WORKER does the token-heavy subtask (search, extract, summarize)
in its OWN isolated context and returns only a compact result. The worker's raw
material - tens of thousands of tokens of sources, retries, partial drafts -
never enters the lead's window.

The cost lever is the model choice: workers do bounded, well-specified jobs, so
route them to a CHEAP model and keep the STRONG model on the lead, where it sees
only the question and the compressed result.

You will learn how to:
- Implement the orchestrator-worker pattern with plain boto3 Converse
- Run a cheap worker (Haiku) that digests a large document into a short summary
- Keep the strong lead (Claude 5 Sonnet) reasoning over only the compact summary
- Compare the lead's input tokens against a single-agent baseline that ingests
  the whole document itself

NOTE: The source notebook uses the Claude Agent SDK / Strands. This sample
implements the same pattern directly with boto3 so it is self-contained (no
extra dependencies). Mind the orchestration overhead: every delegation is a
separate model call, so the win is the lead's context savings - make sure it
exceeds that overhead.

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude Haiku 4.5 and Claude 5 Sonnet on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import boto3

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

RUNTIME = boto3.client("bedrock-runtime", region_name=REGION)

# Strong lead, cheap worker - the whole point of the split.
LEAD = "global.anthropic.claude-sonnet-5"
WORKER = "global.anthropic.claude-haiku-4-5-20251001-v1:0"


# ============================================================
# A large, token-heavy source document
# ============================================================
#
# Stands in for the raw material a worker would gather (search hits, retrieved
# docs, logs). In the delegated design this never touches the lead's context.

def build_large_document() -> str:
    """A verbose internal report - the token-heavy raw material to be digested."""
    sections = []
    for i in range(1, 41):
        sections.append(
            f"Section {i}. During review period {i}, the platform processed a large "
            f"volume of inference requests across multiple teams. Observed spend was "
            f"driven primarily by oversized max_tokens settings, uncached system "
            f"prompts, and frontier-model calls on tasks a smaller model could handle. "
            f"Recommended actions for area {i}: right-size max_tokens, add prompt "
            f"caching to the stable prefix, and route simple lookups to a cheaper model. "
            f"Estimated savings in area {i}: between 20 and 60 percent of that slice."
        )
    return "INTERNAL COST REVIEW REPORT\n\n" + "\n".join(sections)


# ============================================================
# Helpers
# ============================================================

def supports_temperature(model_id: str) -> bool:
    """Claude 5 models (Sonnet 5, Opus 5) deprecated temperature."""
    return "claude-sonnet-5" not in model_id and "claude-opus-5" not in model_id


def extract_text(message: dict, stop_reason: str = "") -> str:
    """Return the text answer, skipping any reasoning block.

    Thinking-capable models (Claude 5 Sonnet) return a reasoningContent block
    before the text block, so content[0] is not necessarily the answer. If the
    model hit the token budget while still reasoning (stopReason 'max_tokens'),
    it may emit no text block at all - return a clear note so the demo output is
    never blank and the reason is obvious.
    """
    for block in message.get("content", []):
        if "text" in block:
            return block["text"]
    if stop_reason == "max_tokens":
        return "[reasoning hit the token budget before a final answer - raise maxTokens]"
    if any("reasoningContent" in b for b in message.get("content", [])):
        return "[model returned reasoning only - raise maxTokens for the final answer]"
    return ""


def converse(model_id: str, prompt: str, max_tokens: int = 1024) -> dict:
    """Single-turn Converse call returning text and token usage."""
    cfg = {"maxTokens": max_tokens}
    if supports_temperature(model_id):
        cfg["temperature"] = 0
    resp = RUNTIME.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig=cfg,
    )
    return {
        "text": extract_text(resp["output"]["message"], resp.get("stopReason", "")),
        "input_tokens": resp["usage"]["inputTokens"],
        "output_tokens": resp["usage"]["outputTokens"],
    }


# ============================================================
# The worker (cheap, isolated context)
# ============================================================

def research_worker(document: str, question: str) -> dict:
    """Digest the large document into a compact summary. Runs on the CHEAP model.

    The worker sees the whole document; this heavy context stays in the worker's
    own call and never enters the lead's window.
    """
    prompt = (
        "You are a research worker. Read the document below and return ONLY a "
        "5-bullet summary that answers the question - no preamble, no raw quotes.\n\n"
        f"QUESTION: {question}\n\n"
        f"DOCUMENT:\n{document}"
    )
    return converse(WORKER, prompt, max_tokens=400)


# ============================================================
# Delegated design vs single-agent baseline
# ============================================================

def delegated(document: str, question: str) -> dict:
    """Lead delegates the heavy read to the worker, then reasons over the summary."""
    worker_out = research_worker(document, question)

    lead_prompt = (
        "You are a lead analyst. Using ONLY the research summary below, give a brief, "
        "prioritized recommendation.\n\n"
        f"QUESTION: {question}\n\n"
        f"RESEARCH SUMMARY:\n{worker_out['text']}"
    )
    # Claude 5 Sonnet supports adaptive thinking; give room for reasoning + answer.
    lead_out = converse(LEAD, lead_prompt, max_tokens=2048)

    return {
        "answer": lead_out["text"],
        "lead_input_tokens": lead_out["input_tokens"],
        "worker_input_tokens": worker_out["input_tokens"],
    }


def single_agent(document: str, question: str) -> dict:
    """Baseline: the strong (expensive) lead ingests the whole document itself."""
    prompt = (
        "You are a lead analyst. Read the document below and give a brief, "
        "prioritized recommendation.\n\n"
        f"QUESTION: {question}\n\n"
        f"DOCUMENT:\n{document}"
    )
    # Claude 5 Sonnet supports adaptive thinking; reading the whole document
    # takes more reasoning, so allow a large budget for reasoning + answer.
    out = converse(LEAD, prompt, max_tokens=8192)
    return {"answer": out["text"], "lead_input_tokens": out["input_tokens"]}


# ============================================================
# Demo
# ============================================================

def demo_delegation() -> None:
    document = build_large_document()
    approx_tokens = len(document) // 4
    question = "What are the top levers to cut our Bedrock spend, in priority order?"

    print("--- Sub-Agent Delegation: keep the heavy context off the strong model ---")
    print(f"  Source document is ~{approx_tokens} tokens.\n")

    print("BASELINE: single strong agent (Sonnet) reads the whole document")
    base = single_agent(document, question)
    print(f"  lead input tokens: {base['lead_input_tokens']}")
    print(f"  answer: {base['answer'][:200]}\n")

    print("DELEGATED: Haiku worker digests the document, Sonnet lead reads only the summary")
    dele = delegated(document, question)
    print(f"  worker input tokens (cheap model): {dele['worker_input_tokens']}")
    print(f"  lead input tokens (expensive model): {dele['lead_input_tokens']}")
    print(f"  answer: {dele['answer'][:200]}\n")

    saved = base["lead_input_tokens"] - dele["lead_input_tokens"]
    pct = (saved / base["lead_input_tokens"] * 100) if base["lead_input_tokens"] else 0
    print("Comparison - tokens on the EXPENSIVE lead model:")
    print(f"  single-agent lead: {base['lead_input_tokens']:>6}")
    print(f"  delegated lead:    {dele['lead_input_tokens']:>6}")
    print(f"  saved on the lead: {saved:>6}  ({pct:.0f}% fewer tokens on the pricey model)")
    print(
        "\n  The heavy document was read once by the CHEAP worker; the expensive lead"
        "\n  saw only the compact summary. The bigger the raw material - and the more"
        "\n  turns the lead runs - the more this split saves.\n"
    )


# ============================================================
# Main
# ============================================================

def main():
    demo_delegation()

    print("--- Done ---")
    print("  Next steps:")
    print("  1. Delegate token-heavy subtasks (search, extract, summarize) to a cheap worker")
    print("  2. Keep the strong lead reasoning over compact worker outputs only")
    print("  3. Confirm the lead's context savings exceed the extra delegation call's cost")
    print("  4. A worker with bad tool descriptions is just bad, faster - fix the harness first (03-1)")


if __name__ == "__main__":
    main()
