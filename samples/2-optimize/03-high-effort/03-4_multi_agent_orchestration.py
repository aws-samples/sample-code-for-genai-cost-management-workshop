"""
High-Effort Optimization - Lever 16: Multi-Agent Orchestration

Every agent you add is another set of model calls and another set of tokens:
its own instructions, its own context, its own output. Splitting a task across
agents is not free, so split only when it buys you something:

  - ISOLATION: a worker digests token-heavy raw material (search hits, logs,
    documents) in its OWN context window and returns only a compact result, so
    the strong lead model never re-reads it on every turn.
  - PARALLEL WORK: independent pieces of work run at the same time, so the
    task finishes faster.

And once you do split, CAP THE FAN-OUT. A lead that is free to spawn one worker
per item pays the per-agent overhead (instructions plus output) once per item,
and hands the lead a long list of partial results to read.

This sample runs the same question over a large document three ways:

  1. Single agent   - the strong lead (Claude 5 Sonnet) reads the whole document.
  2. Uncapped split - one cheap Haiku worker per section, then the lead.
  3. Capped split   - a small, fixed number of Haiku workers (MAX_WORKERS), then
                      the lead.

You will learn how to:
- Implement the orchestrator-worker pattern with plain boto3 Converse
- Run workers in parallel in isolated contexts, on a cheaper model than the lead
- Enforce a fan-out cap in the harness, not in the prompt
- Compare agents, calls, tokens, latency, and cost for each design

NOTE: This sample uses boto3 directly so it is self-contained (no extra
dependencies). The numbers to watch are the lead's input tokens (the isolation
win) and the total cost and call count across ALL agents (the overhead).

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude Haiku 4.5 and Claude 5 Sonnet on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

# Standard retry mode with a few more attempts, so a burst of parallel worker
# calls rides out brief throttling instead of failing the demo.
RUNTIME = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    config=Config(retries={"mode": "standard", "max_attempts": 8}),
)

# Strong lead, cheap worker - the whole point of the split.
LEAD = "global.anthropic.claude-sonnet-5"
WORKER = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# Pricing per 1M tokens (USD). Illustrative workshop values - always verify
# current pricing at https://aws.amazon.com/bedrock/pricing/
PRICING = {
    LEAD: {"input": 3.00, "output": 15.00},
    WORKER: {"input": 1.00, "output": 5.00},
}

# Number of sections in the source document (the units a naive lead would
# fan out over).
SECTION_COUNT = 40

# The fan-out cap: the most workers the harness will let the lead spawn.
MAX_WORKERS = 4

# How many worker calls may be in flight at once. This is a throttling guard
# for the demo, separate from the fan-out cap above.
MAX_CONCURRENCY = 10

# Each worker returns at most this many bullets.
WORKER_BULLETS = 2

QUESTION = "What are the top levers to cut our Bedrock spend, in priority order?"


# ============================================================
# A large, token-heavy source document
# ============================================================
#
# Stands in for the raw material workers would gather (search hits, retrieved
# docs, logs). In the split designs it never touches the lead's context.

def build_sections() -> list[str]:
    """A verbose internal report, as a list of sections."""
    sections = []
    for i in range(1, SECTION_COUNT + 1):
        sections.append(
            f"Section {i}. During review period {i}, the platform processed a large "
            f"volume of inference requests across multiple teams. Observed spend was "
            f"driven primarily by oversized max_tokens settings, uncached system "
            f"prompts, and frontier-model calls on tasks a smaller model could handle. "
            f"Recommended actions for area {i}: right-size max_tokens, add prompt "
            f"caching to the stable prefix, and route simple lookups to a cheaper model. "
            f"Estimated savings in area {i}: between 20 and 60 percent of that slice."
        )
    return sections


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


def calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Return the dollar cost of the given token counts for the given model."""
    prices = PRICING[model_id]
    return (input_tokens / 1_000_000) * prices["input"] + (
        output_tokens / 1_000_000
    ) * prices["output"]


# ============================================================
# The fan-out cap (a harness rule, not a prompt instruction)
# ============================================================

def cap_fan_out(requested: int) -> int:
    """Return how many workers the harness will actually spawn.

    A prompt that says "use at most 4 workers" is a suggestion. This check is
    code: whatever the lead asks for, the harness never spawns more than
    MAX_WORKERS.
    """
    return min(requested, MAX_WORKERS)


def split_into_chunks(sections: list[str], n_workers: int) -> list[str]:
    """Divide the sections into n_workers near-equal chunks of text."""
    size, extra = divmod(len(sections), n_workers)
    chunks, start = [], 0
    for i in range(n_workers):
        end = start + size + (1 if i < extra else 0)
        chunks.append("\n".join(sections[start:end]))
        start = end
    return chunks


# ============================================================
# The worker (cheap, isolated context)
# ============================================================

def research_worker(chunk: str) -> dict:
    """Digest one chunk of the document into a compact summary. Runs on the CHEAP model.

    The worker sees only its chunk; that heavy context stays in the worker's
    own call and never enters the lead's window.
    """
    prompt = (
        f"You are a research worker. Read the document excerpt below and return "
        f"ONLY a {WORKER_BULLETS}-bullet summary that answers the question - no "
        f"preamble, no raw quotes.\n\n"
        f"QUESTION: {QUESTION}\n\n"
        f"DOCUMENT EXCERPT:\n{chunk}"
    )
    return converse(WORKER, prompt, max_tokens=300)


# ============================================================
# The three designs
# ============================================================

def run_single_agent(sections: list[str]) -> dict:
    """Baseline: the strong (expensive) lead ingests the whole document itself."""
    prompt = (
        "You are a lead analyst. Read the document below and give a brief, "
        "prioritized recommendation.\n\n"
        f"QUESTION: {QUESTION}\n\n"
        "DOCUMENT:\nINTERNAL COST REVIEW REPORT\n\n" + "\n".join(sections)
    )
    t0 = time.time()
    # Claude 5 Sonnet supports adaptive thinking; reading the whole document
    # takes more reasoning, so allow a large budget for reasoning + answer.
    out = converse(LEAD, prompt, max_tokens=8192)
    return {
        "answer": out["text"],
        "agents": 1,
        "calls": 1,
        "lead_in": out["input_tokens"],
        "lead_out": out["output_tokens"],
        "worker_in": 0,
        "worker_out": 0,
        "seconds": time.time() - t0,
    }


def run_split(sections: list[str], n_workers: int) -> dict:
    """Lead delegates to n_workers parallel workers, then reasons over their summaries."""
    chunks = split_into_chunks(sections, n_workers)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        worker_outs = list(pool.map(research_worker, chunks))

    summaries = "\n\n".join(
        f"[Worker {i}]\n{o['text']}" for i, o in enumerate(worker_outs, start=1)
    )
    lead_prompt = (
        "You are a lead analyst. Using ONLY the research summaries below, give a brief, "
        "prioritized recommendation.\n\n"
        f"QUESTION: {QUESTION}\n\n"
        f"RESEARCH SUMMARIES:\n{summaries}"
    )
    # Claude 5 Sonnet supports adaptive thinking; give room for reasoning + answer.
    lead_out = converse(LEAD, lead_prompt, max_tokens=2048)

    return {
        "answer": lead_out["text"],
        "agents": n_workers + 1,
        "calls": n_workers + 1,
        "lead_in": lead_out["input_tokens"],
        "lead_out": lead_out["output_tokens"],
        "worker_in": sum(o["input_tokens"] for o in worker_outs),
        "worker_out": sum(o["output_tokens"] for o in worker_outs),
        "seconds": time.time() - t0,
    }


def total_cost(result: dict) -> float:
    """Cost across ALL agents in a design: the lead plus every worker."""
    return calculate_cost(LEAD, result["lead_in"], result["lead_out"]) + calculate_cost(
        WORKER, result["worker_in"], result["worker_out"]
    )


# ============================================================
# Output
# ============================================================

def print_divider() -> None:
    print("\n" + "-" * 70 + "\n")


def print_design(title: str, result: dict) -> None:
    print(title)
    print(f"  agents: {result['agents']}  |  model calls: {result['calls']}")
    print(f"  lead (Claude 5 Sonnet):  {result['lead_in']} input tokens, "
          f"{result['lead_out']} output tokens")
    if result["worker_in"]:
        print(f"  workers (Haiku, total):  {result['worker_in']} input tokens, "
              f"{result['worker_out']} output tokens")
    print(f"  seconds: {result['seconds']:.1f}  |  cost: ${total_cost(result):.5f}")
    print(f"  answer: {result['answer'][:200].strip()}")


def print_comparison_table(rows: list[tuple[str, dict]]) -> None:
    base_cost = total_cost(rows[0][1])
    print("--- Comparison: same question, three designs ---")
    print(f"{'Design':<22} {'Agents':>6} {'Lead input tokens':>18} "
          f"{'Worker input tokens':>20} {'Worker output tokens':>21} "
          f"{'Seconds':>8} {'Cost $':>9} {'Relative cost':>14}")
    for label, r in rows:
        cost = total_cost(r)
        print(f"{label:<22} {r['agents']:>6} {r['lead_in']:>18} "
              f"{r['worker_in'] or '-':>20} {r['worker_out'] or '-':>21} "
              f"{r['seconds']:>8.1f} {cost:>9.5f} {cost / base_cost:>13.2f}x")
    print(
        "\n  The lead column is the isolation win: the strong model reads far fewer"
        "\n  tokens in both split designs. The Agents, Worker and Cost columns are the"
        "\n  overhead: each extra agent repeats its instructions and adds output, and"
        "\n  an uncapped lead has to read every partial result. Latency from a single"
        "\n  run is indicative only."
    )


# ============================================================
# Demo
# ============================================================

def demo_orchestration() -> None:
    sections = build_sections()
    approx_tokens = sum(len(s) for s in sections) // 4

    print("--- Multi-Agent Orchestration: split for isolation, cap the fan-out ---")
    print(f"  Lead model:   {LEAD}")
    print(f"  Worker model: {WORKER}")
    print(f"  Source document: {len(sections)} sections, ~{approx_tokens} tokens")
    print(f"  Fan-out cap (MAX_WORKERS): {MAX_WORKERS}")

    print_divider()
    single = run_single_agent(sections)
    print_design("1. SINGLE AGENT: Sonnet reads the whole document", single)

    print_divider()
    requested = len(sections)
    uncapped = run_split(sections, requested)
    print_design(f"2. UNCAPPED SPLIT: lead asks for {requested} workers, one per section "
                 f"(no cap)", uncapped)

    print_divider()
    allowed = cap_fan_out(requested)
    capped = run_split(sections, allowed)
    print_design(f"3. CAPPED SPLIT: lead asks for {requested} workers, harness allows "
                 f"{allowed}", capped)

    print_divider()
    print_comparison_table([
        ("Single agent", single),
        (f"Uncapped ({requested} workers)", uncapped),
        (f"Capped ({allowed} workers)", capped),
    ])


# ============================================================
# Main
# ============================================================

def main():
    demo_orchestration()

    print("\n--- Done ---")
    print("  Next steps:")
    print("  1. Add an agent only for isolation (heavy context) or parallel work")
    print("  2. Enforce the fan-out cap in code, and size it from measured results")
    print("  3. Compare TOTAL tokens and cost across all agents, not just the lead's")
    print("  4. A worker with bad tool descriptions is just bad, faster - fix tools and context first")


if __name__ == "__main__":
    main()
