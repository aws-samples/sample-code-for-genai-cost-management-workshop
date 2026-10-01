"""
Medium-Effort Optimization - Lever 09: RAG / Indexing (the core concept)

The instinct on a large knowledge source is to stuff it all into the prompt.
That is expensive AND counterproductive: past a certain size accuracy drops
because the model underweights information buried mid-prompt (the "lost in the
middle" effect). Retrieval inverts it - index the corpus once, then per query
pull in only the relevant slice.

This sample teaches the concept WITHOUT a vector database. The whole catalog is
the "corpus"; splitting it into one file per product category is the "index";
and picking the file for the question's category stands in for retrieval. A real
system swaps file-selection for embedding search over chunks, but the cost lever
is identical: send the relevant slice, not the whole corpus.

You will learn how to:
- Build a fake inventory of 10 product categories with many products each
- NAIVE: put the entire catalog in the prompt and measure input tokens
- INDEXED: load only the relevant category file and measure input tokens
- Compare input tokens and cost between the two approaches

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with bedrock-runtime:Converse permission
- Access to Claude Haiku 4.5 on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import boto3

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

RUNTIME = boto3.client("bedrock-runtime", region_name=REGION)

HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# Illustrative input pricing per 1M tokens (USD). Verify current pricing at
# https://aws.amazon.com/bedrock/pricing/
INPUT_PRICE_PER_M = 1.00

# 10 product categories, each with a set of representative products.
CATEGORIES = {
    "laptops": ["UltraBook Pro 14", "UltraBook Air 13", "WorkStation 16", "Budget Notebook 15", "Gaming Rig 17",
                "Convertible 2-in-1 13", "Student Chromebook 14", "Creator Studio 15", "Rugged Field 14", "Thin Client 12"],
    "monitors": ["ViewMax 27 4K", "ViewMax 32 Curved", "Office Display 24", "UltraWide 34", "Portable Screen 15",
                 "Pro Color 27", "Gaming 240Hz 25", "Vertical Pivot 24", "Dual-Stack 21", "Touch Display 27"],
    "keyboards": ["MechType Pro", "MechType TKL", "SlimBoard Wireless", "ErgoSplit 2", "Compact 60%",
                  "LowProfile Wireless", "Numpad Combo", "Backlit Gaming", "Silent Membrane", "Foldable Travel"],
    "mice": ["Precision Wireless", "ErgoVertical", "Gaming Speed X", "TravelMouse Mini", "Trackball Classic",
             "Silent Click Pro", "MMO 12-Button", "Ambidextrous Lite", "Presenter Combo", "Kids Mini Mouse"],
    "headphones": ["SoundPro ANC", "StudioMonitor 2", "Sport Earbuds", "GamerHeadset 7.1", "Budget Buds",
                   "OpenEar Runner", "OverEar Studio Max", "Kids SafeVolume", "Conference Headset", "Bone Conduction Go"],
    "webcams": ["StreamCam 4K", "MeetingCam 1080p", "ClipCam Mini", "PanTilt Conference", "PrivacyCam",
                "AutoFrame 2K", "DualLens Studio", "Ceiling Room Cam", "Document Overhead Cam", "USB Microscope Cam"],
    "storage": ["FastSSD 1TB", "FastSSD 2TB", "RuggedHDD 4TB", "PocketSSD 500GB", "NAS Drive 8TB",
                "NVMe Gen5 2TB", "Encrypted Vault SSD", "microSD 512GB", "External HDD 2TB", "Thunderbolt RAID 16TB"],
    "networking": ["MeshRouter AX", "TravelRouter 6", "Switch 8-Port", "PowerlineKit", "WiFi Extender Pro",
                   "Gaming Router AXE", "Managed Switch 24-Port", "5G Modem Router", "Outdoor AP", "USB WiFi 6 Adapter"],
    "chargers": ["GaN 100W", "GaN 65W", "CarCharger Dual", "WirelessPad 15W", "PowerBank 20000",
                 "Desktop 6-Port Hub", "MagSafe Stand", "Travel Adapter Global", "Solar PowerBank", "AA/AAA Smart Charger"],
    "cables": ["USB-C 2m", "HDMI 2.1 3m", "Thunderbolt 4 1m", "Ethernet Cat6 5m", "DisplayPort 2m",
               "USB-C to Lightning 1m", "Braided USB-A 3m", "Fiber HDMI 10m", "Cat8 Ethernet 3m", "Magnetic Charge 1m"],
}


# ============================================================
# Build the fake inventory
# ============================================================

def product_line(category: str, name: str, i: int) -> str:
    """One verbose catalog line per product - realistic filler that costs tokens."""
    price = 29 + (i * 37) % 900
    sku = f"{category[:3].upper()}-{1000 + i}"
    return (
        f"- [{sku}] {name} ({category}): premium-grade {category} unit with a "
        f"2-year warranty, in-stock and eligible for free next-day shipping. "
        f"List price ${price}.99. Includes standard accessories and a 30-day "
        f"return window. Rated 4.{i % 9 + 1}/5 across verified customer reviews."
    )


def build_full_catalog() -> str:
    """The entire catalog as one big block - the naive 'stuff it all in' corpus."""
    lines = ["FULL PRODUCT CATALOG", ""]
    idx = 0
    for category, products in CATEGORIES.items():
        lines.append(f"== {category.upper()} ==")
        for name in products:
            idx += 1
            lines.append(product_line(category, name, idx))
        lines.append("")
    return "\n".join(lines)


def build_category_index() -> dict:
    """Split the catalog into one 'file' per category - the index we retrieve from."""
    index = {}
    idx = 0
    for category, products in CATEGORIES.items():
        lines = [f"== {category.upper()} =="]
        for name in products:
            idx += 1
            lines.append(product_line(category, name, idx))
        index[category] = "\n".join(lines)
    return index


# ============================================================
# Ask a question with a given context block
# ============================================================

def ask(context: str, question: str) -> dict:
    """Answer the question using only the supplied context block. Returns usage."""
    prompt = (
        "Use only the product information below to answer the question. "
        "Be concise.\n\n"
        f"{context}\n\n"
        f"QUESTION: {question}"
    )
    resp = RUNTIME.converse(
        modelId=HAIKU,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 200, "temperature": 0},
    )
    return {
        "text": resp["output"]["message"]["content"][0]["text"],
        "input_tokens": resp["usage"]["inputTokens"],
        "output_tokens": resp["usage"]["outputTokens"],
    }


def input_cost(tokens: int) -> float:
    return (tokens / 1_000_000) * INPUT_PRICE_PER_M


# ============================================================
# Demo
# ============================================================

def demo_rag_concept() -> None:
    full_catalog = build_full_catalog()
    index = build_category_index()

    # A question about ONE category. In a real system a retriever would find the
    # relevant slice; here we know the question is about headphones.
    question = "Which headphones have active noise cancellation, and what's the price?"
    target_category = "headphones"

    print("--- RAG / Indexing concept: send the relevant slice, not the whole corpus ---")
    print(
        f"  Catalog: {len(CATEGORIES)} categories, "
        f"{sum(len(v) for v in CATEGORIES.values())} products.\n"
    )

    # NAIVE: whole catalog in the prompt.
    print("NAIVE: entire catalog in the prompt")
    naive = ask(full_catalog, question)
    print(f"  input tokens: {naive['input_tokens']}")
    print(f"  answer: {naive['text'][:160]}\n")

    # INDEXED: load only the relevant category 'file'.
    print(f"INDEXED: load only the '{target_category}' file")
    indexed = ask(index[target_category], question)
    print(f"  input tokens: {indexed['input_tokens']}")
    print(f"  answer: {indexed['text'][:160]}\n")

    # Compare.
    saved = naive["input_tokens"] - indexed["input_tokens"]
    pct = (saved / naive["input_tokens"] * 100) if naive["input_tokens"] else 0
    print("Comparison (input tokens):")
    print(f"  naive:    {naive['input_tokens']:>6}  (${input_cost(naive['input_tokens']):.6f})")
    print(f"  indexed:  {indexed['input_tokens']:>6}  (${input_cost(indexed['input_tokens']):.6f})")
    print(f"  saved:    {saved:>6}  ({pct:.0f}% fewer input tokens)\n")

    print(
        "  Same answer, a fraction of the input tokens. The naive prompt pays for 9"
        "\n  irrelevant categories on every call; the indexed approach pays only for"
        "\n  the slice the question needs. On a real corpus (thousands of products,"
        "\n  long documents) this is the difference between a 50K-token dump and a"
        "\n  3-5K-token retrieval - and it improves accuracy too, by avoiding the"
        "\n  'lost in the middle' effect.\n"
    )


# ============================================================
# Main
# ============================================================

def main():
    demo_rag_concept()

    print("--- Done ---")
    print("  This sample uses file-selection as a stand-in for retrieval. In production:")
    print("  1. Chunk and embed your corpus once into a vector store (e.g. Bedrock")
    print("     Knowledge Bases)")
    print("  2. Per query, retrieve only the top-K relevant chunks")
    print("  3. Add metadata filtering and reranking to shrink and sharpen the context")
    print("  4. Keep retrieved chunks AFTER any cachePoint - they are dynamic (see 01-4)")


if __name__ == "__main__":
    main()
