"""
Medium-Effort Optimization - Lever 12: Filter Tool Output

When an agent runs a command, the raw output usually goes straight back to the
model. That output is mostly noise for the question at hand: package install
progress bars, ANSI color codes, hundreds of INFO and DEBUG log lines, the same
deprecation warning repeated for every test, and a row of dots for each passing
test. You pay input tokens for all of it, and in an agent loop you pay again on
every later turn, because the whole history (tool output included) is resent.

The fix is to filter the tool result in code BEFORE it reaches the model: strip
console noise, collapse repeated warnings, drop logs and package output, and
keep what the model needs (the failure, its assertion message, the final
summary line).

This sample feeds the output of a simulated CI step (pip install followed by a
pytest run with one failing test) to Claude Haiku 4.5 three ways:
- RAW: the full tool output, exactly as the command printed it
- PYTHON FILTER: a small hand-written filter tuned to this tool
- RTK: the open-source rtk CLI (https://github.com/rtk-ai/rtk), which ships
  ready-made filters for common developer commands. Optional: if rtk is not
  installed the sample offers to install it (you confirm first), and skips
  this row cleanly if you decline.

You will learn how to:
- Strip ANSI codes, package output, log noise, and repeated warnings
- Filter a tool result in code, or with a ready-made filter such as rtk
- Compare input tokens, cost, and answer quality across the three
- See why filtering must remove noise but keep signal: check the answer, not
  just the token count

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude Haiku 4.5 on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
- OPTIONAL: the rtk CLI from https://github.com/rtk-ai/rtk (a single Rust
  binary). The sample can install it for you with the project's install script:
      curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh
  The script downloads a checksum-verified binary to ~/.local/bin. Install the
  rtk-ai project, not another tool that shares the name.
"""

import os
import re
import shutil
import subprocess
import sys

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

# In an agent loop the tool result stays in the history and is resent on every
# later turn. Used only for the closing estimate.
AGENT_TURNS = 10

# The question the agent is trying to answer with this tool result.
QUESTION = (
    "Which test failed, and what is the most likely root cause? "
    "Answer in two sentences."
)

# The real cause is in a WARNING log line buried in the output, not in the
# assertion itself. The answer check looks for the idea behind it: the bulk
# discount threshold excludes a quantity of exactly 100.
ROOT_CAUSE_RE = re.compile(
    r"exclusive|inclusive|boundary|>=|strictly|greater than|> ?100|off-by-one", re.I
)

# rtk is optional. If it is missing, the sample offers to install it with the
# project's official install script. Set OFFER_RTK_INSTALL = False to never ask.
OFFER_RTK_INSTALL = True
RTK_INSTALL_CMD = (
    "curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh"
)
# Where the install script puts the binary (it may not be on your PATH yet).
RTK_INSTALL_DIR = os.path.expanduser("~/.local/bin")

ESC = "\x1b"


# ============================================================
# The tool: a simulated CI step
# ============================================================

def run_ci_step() -> str:
    """Simulated output of 'pip install' followed by 'pytest'.

    It is built in code so every run is identical and needs no network or test
    project. The shape matches what the real commands print: install progress
    bars with color codes, passing-test dots, repeated deprecation warnings, a
    captured log full of INFO and DEBUG lines, and one failing test.
    """
    out = []

    # pip install: package output
    pkgs = [("boto3", "1.43.126", 139), ("botocore", "1.43.126", 14200),
            ("requests", "2.34.2", 64), ("urllib3", "2.5.0", 129),
            ("certifi", "2026.4.1", 159), ("idna", "3.10", 70),
            ("s3transfer", "0.14.0", 85), ("jmespath", "1.0.1", 20)]
    for name, version, kb in pkgs:
        out.append(f"Collecting {name}=={version}")
        out.append(f"  Downloading {name}-{version}-py3-none-any.whl ({kb} kB)")
        out.append(f"     {ESC}[38;5;237m{'-' * 40}{ESC}[0m {kb}.0/{kb}.0 kB 4.2 MB/s eta 0:00:00")
    for name, version, _ in pkgs:
        out.append(f"Requirement already satisfied: {name} in ./.venv/lib/python3.12/site-packages ({version})")
    out.append("Installing collected packages: " + ", ".join(p[0] for p in pkgs))
    out.append("Successfully installed " + " ".join(f"{p[0]}-{p[1]}" for p in pkgs))
    out.append("")

    # pytest: header, progress dots, one failure
    out.append("============================= test session starts ==============================")
    out.append("platform darwin -- Python 3.12.4, pytest-8.3.2, pluggy-1.5.0")
    out.append("rootdir: /workspace/orders-service")
    out.append("plugins: cov-5.0.0, anyio-4.4.0")
    out.append("collected 60 items")
    out.append("")
    out.append("tests/test_orders.py " + "." * 20)
    out.append("tests/test_pricing.py " + "." * 12 + "F" + "." * 7)
    out.append("tests/test_shipping.py " + "." * 20)
    out.append("")
    out.append("=================================== FAILURES ===================================")
    out.append("______________________ test_discount_applied_to_bulk_order ______________________")
    out.append("")
    out.append("    def test_discount_applied_to_bulk_order():")
    out.append("        order = make_order(items=[Item('SKU-1', qty=100, unit_price=Decimal('9.99'))])")
    out.append(">       assert price_order(order).total == Decimal('899.10')")
    out.append("E       AssertionError: assert Decimal('999.00') == Decimal('899.10')")
    out.append("E        +  where Decimal('999.00') = Priced(total=Decimal('999.00')).total")
    out.append("")
    out.append("tests/test_pricing.py:41: AssertionError")
    out.append("------------------------------ Captured log call -------------------------------")
    for i in range(25):
        out.append(f"INFO     orders.pricing:pricing.py:{20 + i} loading price rule {i} from cache")
    out.append("WARNING  orders.pricing:pricing.py:88 bulk_discount rule 'BULK10' not matched: "
               "min_qty=100 but the threshold check is exclusive (qty > 100)")
    for i in range(10):
        out.append(f"DEBUG    botocore.hooks:hooks.py:238 Event before-parameter-build.s3.GetObject: calling handler {i}")
    out.append("")

    # The same deprecation warning, once per test file location
    out.append("=============================== warnings summary ===============================")
    for where in ["tests/test_orders.py:12", "tests/test_orders.py:57", "tests/test_pricing.py:9",
                  "tests/test_shipping.py:30", "tests/test_shipping.py:44", "tests/test_shipping.py:71"]:
        out.append(where)
        out.append("  /workspace/orders-service/.venv/lib/python3.12/site-packages/botocore/auth.py:419: "
                   "DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for "
                   "removal in a future version. Use timezone-aware objects to represent datetimes "
                   "in UTC: datetime.datetime.now(datetime.UTC).")
        out.append("    datetime.datetime.utcnow()")
        out.append("")
    out.append("-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html")
    out.append("=========================== short test summary info ============================")
    out.append("FAILED tests/test_pricing.py::test_discount_applied_to_bulk_order - "
               "AssertionError: assert Decimal('999.00') == Decimal('899.10')")
    out.append("================== 1 failed, 59 passed, 6 warnings in 3.21s ===================")
    return "\n".join(out)


# ============================================================
# Filter 1: a hand-written Python filter tuned to this tool
# ============================================================

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
PIP_RE = re.compile(
    r"^(Collecting |  Downloading |Requirement already satisfied|"
    r"Installing collected packages|Successfully installed|\s+\S+ \d+(\.\d+)?/\d+(\.\d+)? kB)"
)
LOG_NOISE_RE = re.compile(r"^(INFO|DEBUG)\s")
PYTEST_HEADER_RE = re.compile(r"^(=+ test session starts|platform |rootdir:|plugins:|collected \d+)")
DOTS_RE = re.compile(r"^\S+\.py [.sxFE]+$")


def filter_tool_output(raw: str) -> str:
    """Strip noise from the tool output, keep the failure and the summary."""
    text = ANSI_RE.sub("", raw)

    kept = []
    packages = 0
    warnings: dict[str, int] = {}
    in_warnings = False

    for line in text.splitlines():
        if line.startswith("Collecting "):
            packages += 1
        if PIP_RE.match(line):
            continue

        # Warnings summary: keep each distinct warning once, with a count.
        if "warnings summary" in line:
            in_warnings = True
            continue
        if in_warnings:
            if line.startswith("-- Docs:") or line.startswith("="):
                in_warnings = False
            else:
                if "Warning:" in line:
                    msg = line.strip()
                    warnings[msg] = warnings.get(msg, 0) + 1
                continue

        if LOG_NOISE_RE.match(line) or PYTEST_HEADER_RE.match(line) or DOTS_RE.match(line):
            continue
        if line.startswith("-- Docs:") or not line.strip():
            continue
        kept.append(line)

    summary = []
    if packages:
        summary.append(f"[pip] installed {packages} packages (output removed)")
    for msg, count in warnings.items():
        summary.append(f"[warning x{count}] {msg[:140]}")
    return "\n".join(summary + kept)


# ============================================================
# Filter 2: rtk, a ready-made filter (optional)
# ============================================================

def find_rtk() -> str | None:
    """Path to the rtk binary: on PATH, or where the install script puts it."""
    found = shutil.which("rtk")
    if found:
        return found
    candidate = os.path.join(RTK_INSTALL_DIR, "rtk")
    return candidate if os.access(candidate, os.X_OK) else None


def offer_rtk_install() -> str | None:
    """Ask before running the install script. Returns the rtk path, or None."""
    if not OFFER_RTK_INSTALL:
        return None
    print("  rtk is not installed. It can be installed with the project's install script:")
    print(f"    {RTK_INSTALL_CMD}")
    print("  The script downloads a checksum-verified binary into ~/.local/bin.")
    if not sys.stdin.isatty():
        print("  Not an interactive terminal, so the install is skipped.\n")
        return None
    try:
        answer = input("  Install rtk now? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    print()
    if answer not in ("y", "yes"):
        return None
    # The command is a constant defined above, not built from user input.
    result = subprocess.run(RTK_INSTALL_CMD, shell=True)
    print()
    if result.returncode != 0:
        print("  rtk install failed, skipping the rtk row.\n")
        return None
    return find_rtk()


def filter_with_rtk(rtk_path: str, raw: str) -> str | None:
    """Pipe the output through rtk's pytest filter. None if it printed nothing."""
    result = subprocess.run(
        [rtk_path, "pipe", "-f", "pytest"],
        input=raw, capture_output=True, text=True, timeout=30,
    )
    # rtk prints a hook notice on stderr; only stdout is the filtered result.
    return result.stdout.strip() or None


# ============================================================
# Ask the model
# ============================================================

def ask(tool_output: str) -> dict:
    prompt = (
        "A CI step just finished. Here is its output.\n\n"
        f"<tool_output>\n{tool_output}\n</tool_output>\n\n"
        f"{QUESTION}"
    )
    resp = RUNTIME.converse(
        modelId=HAIKU,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 250, "temperature": 0},
    )
    return {
        "text": resp["output"]["message"]["content"][0]["text"].strip(),
        "input_tokens": resp["usage"]["inputTokens"],
        "output_tokens": resp["usage"]["outputTokens"],
    }


def cost_of(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * INPUT_PRICE_PER_M
        + output_tokens / 1_000_000 * OUTPUT_PRICE_PER_M
    )


# ============================================================
# Comparison
# ============================================================

RESULTS: list[dict] = []


def print_comparison_table() -> None:
    base = RESULTS[0]
    base_cost = cost_of(base["input_tokens"], base["output_tokens"])
    headers = ["Tool output sent", "Characters", "Input tokens", "Output tokens",
               "Cost ($)", "vs. raw", "Found root cause?"]
    rows = []
    for r in RESULTS:
        c = cost_of(r["input_tokens"], r["output_tokens"])
        rows.append([
            r["label"],
            str(r["chars"]),
            str(r["input_tokens"]),
            str(r["output_tokens"]),
            f"{c:.5f}",
            f"{c / base_cost:.2f}x",
            "yes" if r["root_cause"] else "no",
        ])

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    left = {0, 6}

    def fmt(cells: list[str]) -> str:
        return "  ".join(
            c.ljust(widths[i]) if i in left else c.rjust(widths[i])
            for i, c in enumerate(cells)
        )

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))

    # Only count a filter as a saving if the answer was still right.
    good = [r for r in RESULTS[1:] if r["root_cause"]] or RESULTS[1:] or [base]
    best = min(good, key=lambda r: r["input_tokens"])
    print(
        f"\n  Over a {AGENT_TURNS}-turn agent loop the tool output is resent every turn:"
        f"\n  raw {base['input_tokens'] * AGENT_TURNS:,} input tokens vs "
        f"{best['input_tokens'] * AGENT_TURNS:,} with '{best['label']}'.\n"
    )


def show(title: str, text: str, max_lines: int = 14) -> None:
    lines = text.splitlines()
    print(f"  {title} ({len(text)} characters, {len(lines)} lines)")
    for line in lines[:max_lines]:
        print(f"    | {line[:110]}")
    if len(lines) > max_lines:
        print(f"    | ... {len(lines) - max_lines} more lines")
    # Blank lines and a divider so consecutive outputs read as separate blocks.
    print()
    print("  " + "-" * 70)
    print()


# ============================================================
# Main
# ============================================================

def main():
    print("--- Filter Tool Output: strip noise before the model sees it ---")
    print(f"  Model: Claude Haiku 4.5, region {REGION}")
    print(f"  Question for the model: {QUESTION}\n")

    # Step 1: the tool runs and produces its raw output.
    print("Step 1 of 4: the tool returns its raw output")
    raw = run_ci_step()
    show("RAW", raw)

    # Step 2: filter it in code, and with rtk if it is installed.
    print("Step 2 of 4: filter the output before it goes to the model")
    filtered = filter_tool_output(raw)
    show("PYTHON FILTER", filtered, max_lines=20)
    rtk_path = find_rtk() or offer_rtk_install()
    rtk_filtered = filter_with_rtk(rtk_path, raw) if rtk_path else None
    if rtk_filtered is not None:
        show("RTK (rtk pipe -f pytest)", rtk_filtered, max_lines=20)
    else:
        print("  RTK: not available, skipping. See https://github.com/rtk-ai/rtk\n")

    # Step 3: ask the same question with each version.
    print("Step 3 of 4: ask the model the same question with each version")
    candidates = [("Raw", raw), ("Python filter", filtered)]
    if rtk_filtered is not None:
        candidates.append(("rtk pytest filter", rtk_filtered))
    for label, tool_output in candidates:
        r = ask(tool_output)
        r.update(label=label, chars=len(tool_output), root_cause=bool(ROOT_CAUSE_RE.search(r["text"])))
        RESULTS.append(r)
        print(f"  {label}: {' '.join(r['text'].split())[:260]}\n")

    # Step 4: comparison table.
    print("Step 4 of 4: comparison\n")
    print_comparison_table()

    print("--- Done ---")
    print("  1. Never pass raw tool output to the model by default: add a filter between")
    print("     the tool and the model")
    print("  2. Remove noise (progress bars, color codes, INFO/DEBUG logs, package output,")
    print("     repeated warnings); keep signal (errors, assertions, the final summary)")
    print("  3. A general-purpose filter can drop a line your task needs. The last column")
    print("     checks that the answer is still right - run your eval set after changing")
    print("     a filter (see 01-6)")
    print("  4. Keep the full output outside the prompt (a file or log) so the model can")
    print("     ask for more when the filtered view is not enough")


if __name__ == "__main__":
    main()
