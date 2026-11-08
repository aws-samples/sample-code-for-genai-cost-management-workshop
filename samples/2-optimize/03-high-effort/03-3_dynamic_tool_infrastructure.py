"""
High-Effort Optimization - Lever 03: Dynamic Tool Infrastructure

Every tool you give an agent - its name, description, and input schema - is
sent to the model on every call and billed as input tokens. Connect an agent
to a large MCP server and it pays for the whole catalog on every request, even
when the task needs a single tool. An agent loop makes it worse: the tool
definitions ride along on each model call in the loop, not just the first.

You will learn how to:
- Run a Strands agent against a local MCP server that exposes 101 tools
- Run the same agent with an MCP tool filter that loads only the tool it needs
- Compare tools loaded, input/output tokens, latency, and per-call cost side
  by side

The rule:
- One agent, one focused job, a small tool set. Do not load the full tool
  catalog into every prompt.
- Filter MCP tools on the client (tool_filters) instead of loading whole
  servers.
- When the catalog is too large or changes too often to filter by hand, move
  tool selection into the infrastructure - for example AgentCore Gateway
  semantic tool search.

NOTE: The MCP server runs locally as a child process of this script (stdio),
so nothing is deployed. Its tools return fixed demo data.

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials allowed to call Converse/ConverseStream on Amazon Bedrock
- Access to Claude Haiku 4.5 on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import sys
import time

from mcp import StdioServerParameters, stdio_client
from mcp.server.mcpserver import MCPServer
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient


# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

# Allowed workshop models (Global cross-region inference profiles)
HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# Pricing per 1M tokens (USD). Illustrative workshop values - always verify
# current pricing at https://aws.amazon.com/bedrock/pricing/
PRICING = {
    HAIKU: {"input": 1.00, "output": 5.00},
}

# The same user query for both runs, so the comparison is fair.
QUERY = "What is the weather in Paris?"

# Run B loads only the tools this agent actually needs.
ALLOWED_TOOLS = ["get_weather"]

# When the script is launched with this flag, it runs as the local MCP server
# instead of the demo. This keeps the whole sample in a single file.
SERVER_FLAG = "--serve"


# ============================================================
# Local MCP Server (fake tool catalog)
# ============================================================

# log_level="WARNING" keeps the MCP SDK from turning on INFO logs for the demo.
mcp_server = MCPServer("demo-tool-catalog", log_level="WARNING")

# The only tool the agent actually needs to answer QUERY.
@mcp_server.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"25 degrees C and sunny in {city}"


# Building blocks for the fake tools: 20 domains x 5 actions = 100 tool names,
# like a large real-world MCP server (create_invoice, list_order, ...).
DOMAINS = [
    "email", "calendar", "invoice", "customer", "order",
    "ticket", "contract", "employee", "payment", "shipment",
    "product", "report", "meeting", "document", "project",
    "task", "vendor", "budget", "inventory", "campaign",
]
ACTIONS = ["create", "get", "update", "delete", "list"]


def fake_tool(record_id: str) -> str:
    """Placeholder body shared by every fake tool. It is never called."""
    return "ok"


# Register the fake tools. None of them helps answer QUERY, but every
# definition is still sent to the model when the full catalog is loaded.
for domain in DOMAINS:
    for action in ACTIONS:
        mcp_server.add_tool(
            fake_tool,
            name=f"{action}_{domain}",
            description=f"{action.capitalize()} a {domain} record in the company system by its ID.",
        )


# ============================================================
# Helper Functions
# ============================================================

def calculate_cost(input_tokens: int, output_tokens: int, model_id: str) -> float:
    """Return the dollar cost of a single request for the given model."""
    prices = PRICING[model_id]
    return (input_tokens / 1_000_000) * prices["input"] + (
        output_tokens / 1_000_000
    ) * prices["output"]


def run_agent(tool_filters: dict | None = None) -> dict:
    """Run one agent on QUERY against the local MCP server and report usage."""
    # Launch this same file as the MCP server, in the background (stdio).
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.abspath(__file__), SERVER_FLAG],
    )
    mcp_client = MCPClient(
        lambda: stdio_client(server_params),
        tool_filters=tool_filters,
    )

    # The agent starts the MCP server, loads its (filtered) tools, answers
    # QUERY, then shuts the server down when the "with" block ends.
    with Agent(
        model=BedrockModel(model_id=HAIKU, region_name=REGION),
        tools=[mcp_client],
        callback_handler=None,
    ) as agent:
        t0 = time.time()
        result = agent(QUERY)
        latency_ms = int((time.time() - t0) * 1000)
        # Count the tools while the agent is still running.
        tools_loaded = len(agent.tool_names)

    # Token usage summed over every model call the agent made for this query.
    usage = result.metrics.accumulated_usage
    return {
        "text": str(result).strip(),
        "tools_loaded": tools_loaded,
        "input_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],
        "latency_ms": latency_ms,
    }


# ============================================================
# Demo
# ============================================================

def compare_tool_catalogs() -> None:
    """Ask the same query with the full MCP catalog vs only the needed tools."""
    print("--- Dynamic Tool Infrastructure: same query, full vs filtered catalog ---\n")

    full = run_agent()
    filtered = run_agent({"allowed": ALLOWED_TOOLS})
    runs = [("Full catalog", full), ("Filtered catalog", filtered)]

    for label, result in runs:
        print(f"=== {label} (tools loaded: {result['tools_loaded']}) ===")
        print(f"{result['text']}\n")

    print("--- Comparison: same query, two tool catalogs ---")
    print(f"{'Catalog':<18} {'Tools':>6} {'In':>7} {'Out':>6} {'ms':>7} {'$/call':>11}")
    for label, result in runs:
        cost = calculate_cost(result["input_tokens"], result["output_tokens"], HAIKU)
        print(
            f"{label:<18} {result['tools_loaded']:>6} {result['input_tokens']:>7} "
            f"{result['output_tokens']:>6} {result['latency_ms']:>7} {cost:>11.6f}"
        )

    # Savings of the filtered run versus the full catalog, in tokens and dollars.
    full_cost = calculate_cost(full["input_tokens"], full["output_tokens"], HAIKU)
    filtered_cost = calculate_cost(filtered["input_tokens"], filtered["output_tokens"], HAIKU)
    saved_tokens = 1 - filtered["input_tokens"] / full["input_tokens"]
    print(
        f"\n  Filtering the catalog cut input tokens by {saved_tokens:.0%} and made"
        f"\n  each query {full_cost / filtered_cost:.1f}x cheaper "
        f"(${full_cost:.6f} -> ${filtered_cost:.6f}) for the same answer."
    )
    print(
        "\n  Every tool definition rides along on each model call, and an agent makes"
        "\n  at least two per query (pick the tool, then answer), so an unused catalog"
        "\n  is paid for again and again. Input tokens are the stable signal here;"
        "\n  latency from a single run is indicative only."
    )


# ============================================================
# Main
# ============================================================

def main():
    # Child process started by run_agent: act as the MCP server over stdio.
    if SERVER_FLAG in sys.argv:
        mcp_server.run()
        return

    compare_tool_catalogs()

    print("\n--- Done ---")
    print("  Next steps:")
    print("  1. Give each agent a focused job and only the tools that job needs")
    print("  2. Use tool filters on MCP clients instead of loading whole servers")
    print("  3. For large catalogs, look at AgentCore Gateway semantic tool search")


if __name__ == "__main__":
    main()
