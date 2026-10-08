"""CloudWatch trace usage and AWS Price List helpers for the low-effort samples."""

import json
import os
import re
from datetime import datetime, timedelta, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError


REGION = os.environ.get("AWS_REGION", "us-east-1")
LOG_GROUP_NAME = os.environ.get(
    "AGENTCORE_EVAL_LOG_GROUP",
    "/aws/bedrock-agentcore/runtimes/low-effort-summarizer-local",
)
SPAN_LOG_STREAM = os.environ.get("AGENTCORE_EVAL_SPAN_LOG_STREAM", "spans")
TRACE_LOOKBACK_HOURS = 24
PRICE_LIST_REGION = "us-east-1"
PRICE_LIST_SERVICES = (
    "AmazonBedrockFoundationModels",
    "AmazonBedrock",
    "AmazonBedrockService",
)
MIN_SESSION_ID_LENGTH = 33
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_session_id(session_id):
    if len(session_id) < MIN_SESSION_ID_LENGTH:
        raise ValueError(
            f"Session ID must be at least {MIN_SESSION_ID_LENGTH} characters."
        )
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError(
            "Session ID must start with a letter or digit and contain only "
            "letters, digits, hyphens, or underscores."
        )
    return session_id


def _find_spans(value):
    if isinstance(value, dict):
        if "spanId" in value and isinstance(value.get("attributes"), dict):
            yield value
        for child in value.values():
            yield from _find_spans(child)
    elif isinstance(value, list):
        for child in value:
            yield from _find_spans(child)


def _session_token_counts(logs, session_id):
    start_time = int(
        (datetime.now(timezone.utc) - timedelta(hours=TRACE_LOOKBACK_HOURS))
        .timestamp()
        * 1000
    )
    spans = {}
    pages = logs.get_paginator("filter_log_events").paginate(
        logGroupName=LOG_GROUP_NAME,
        logStreamNames=[SPAN_LOG_STREAM],
        startTime=start_time,
        filterPattern=f'"{session_id}"',
    )
    for page in pages:
        for event in page.get("events", []):
            try:
                payload = json.loads(event["message"])
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            for span in _find_spans(payload):
                if span["attributes"].get("session.id") == session_id:
                    spans[span["spanId"]] = span

    token_spans = []
    for span in spans.values():
        attributes = span["attributes"]
        try:
            input_tokens = int(attributes.get("gen_ai.usage.input_tokens", 0))
            output_tokens = int(attributes.get("gen_ai.usage.output_tokens", 0))
        except (TypeError, ValueError):
            continue
        if input_tokens or output_tokens:
            token_spans.append(
                (
                    attributes.get("gen_ai.operation.name"),
                    attributes.get("gen_ai.request.model"),
                    input_tokens,
                    output_tokens,
                )
            )

    selected = [span for span in token_spans if span[0] == "chat"]
    if not selected:
        selected = [span for span in token_spans if span[0] == "invoke_agent"]
    if not selected:
        selected = token_spans

    usage = {}
    for _, model, input_tokens, output_tokens in selected:
        totals = usage.setdefault(model or "Unknown model", [0, 0])
        totals[0] += input_tokens
        totals[1] += output_tokens
    return usage


def _model_key(name, provider=""):
    name = re.sub(r"\s*\(Amazon Bedrock Edition\)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^(global|us|eu|apac|jp|au|in)\.", "", name.lower())
    if "." in name and " " not in name:
        name = name.split(".", 1)[1]
    name = re.sub(r"-?20\d{6}.*$", "", name)
    name = re.sub(r"[-_]v\d+(?::\d+)?$", "", name)
    key = re.sub(r"[^a-z0-9]", "", name)
    provider_key = re.sub(r"[^a-z0-9]", "", provider.lower())
    if provider_key and key.startswith(provider_key):
        key = key[len(provider_key):]
    return key


def _price_per_million(dimension):
    unit = dimension.get("unit", "").lower().replace(",", "")
    match = re.search(r"([\d.]+)?\s*(million|thousand|m|k)?\s*tokens?", unit)
    price = dimension.get("pricePerUnit", {}).get("USD")
    if not match or price is None:
        return None
    quantity = float(match[1] or 1) * {
        "million": 1_000_000,
        "m": 1_000_000,
        "thousand": 1_000,
        "k": 1_000,
    }.get(match[2], 1)
    return float(price) * 1_000_000 / quantity


def _standard_token_rates(products, model_id):
    rates = {"input": set(), "output": set()}
    prefer_global = model_id.lower().startswith("global.")
    excluded = (
        "batch",
        "cache",
        "flex",
        "priority",
        "provisioned",
        "reserved",
        "long_ctx",
        "long context",
    )

    for product in products:
        attributes = product.get("product", {}).get("attributes", {})
        usage = attributes.get("usagetype", "").lower()
        tier = attributes.get("service_tier", "").lower()
        if tier and tier != "standard":
            continue

        for term in product.get("terms", {}).get("OnDemand", {}).values():
            for dimension in term.get("priceDimensions", {}).values():
                description = dimension.get("description", "").lower()
                label = f"{usage} {description}"
                if any(value in label for value in excluded):
                    continue
                if prefer_global != ("global" in label):
                    continue

                kind = (
                    "input"
                    if "input" in label or "prompt" in label
                    else "output"
                    if "output" in label
                    or "response" in label
                    or "completion" in label
                    else None
                )
                rate = _price_per_million(dimension)
                if kind and rate is not None:
                    rates[kind].add(rate)

    if all(len(value) == 1 for value in rates.values()):
        return {key: next(iter(value)) for key, value in rates.items()}
    return None


def _model_token_rates(pricing, model_id):
    model = model_id.lower()
    model = re.sub(r"^(global|us|eu|apac|jp|au|in)\.", "", model)
    provider = model.split(".", 1)[0] if "." in model else ""
    target = _model_key(model_id)
    providerless_target = _model_key(model_id, provider)

    for service_code in PRICE_LIST_SERVICES:
        try:
            service = pricing.describe_services(ServiceCode=service_code)["Services"][0]
        except (BotoCoreError, ClientError, IndexError):
            continue

        attribute_names = {
            name.lower(): name for name in service.get("AttributeNames", [])
        }
        region_field = attribute_names.get("regioncode")
        if not region_field:
            continue

        for field_key in ("servicename", "model"):
            field = attribute_names.get(field_key)
            if not field:
                continue

            matches = []
            pages = pricing.get_paginator("get_attribute_values").paginate(
                ServiceCode=service_code, AttributeName=field
            )
            for page in pages:
                for item in page.get("AttributeValues", []):
                    value = item.get("Value", "")
                    key = _model_key(value, provider)
                    if key and key in {target, providerless_target}:
                        matches.append(value)

            for value in matches:
                filters = [
                    {"Type": "TERM_MATCH", "Field": field, "Value": value},
                    {
                        "Type": "TERM_MATCH",
                        "Field": region_field,
                        "Value": REGION,
                    },
                ]
                products = []
                pages = pricing.get_paginator("get_products").paginate(
                    ServiceCode=service_code,
                    FormatVersion="aws_v1",
                    Filters=filters,
                )
                for page in pages:
                    products.extend(
                        json.loads(raw) for raw in page.get("PriceList", [])
                    )
                rates = _standard_token_rates(products, model_id)
                if rates:
                    return rates
    return None


def report_session_usage(session_id):
    """Print session token totals and estimated input/output inference cost."""
    logs = boto3.client("logs", region_name=REGION)
    pricing = boto3.client("pricing", region_name=PRICE_LIST_REGION)
    try:
        usage = _session_token_counts(logs, session_id)
    except (BotoCoreError, ClientError) as error:
        print(f"Session token counts unavailable from CloudWatch traces: {error}")
        return
    if not usage:
        print("Session token counts not found in the CloudWatch traces.")
        return

    total_input = sum(counts[0] for counts in usage.values())
    total_output = sum(counts[1] for counts in usage.values())
    print(
        "Agent session tokens: "
        f"input={total_input}, output={total_output}, "
        f"total={total_input + total_output}"
    )
    try:
        cost = 0
        for model, (input_tokens, output_tokens) in usage.items():
            rates = _model_token_rates(pricing, model)
            if not rates:
                raise ValueError("No unambiguous model rates found.")
            cost += (
                input_tokens * rates["input"] + output_tokens * rates["output"]
            ) / 1_000_000
    except (
        BotoCoreError,
        ClientError,
        IndexError,
        KeyError,
        StopIteration,
        TypeError,
        ValueError,
    ):
        print("Estimated cost unavailable from the AWS Price List.")
        return
    print(
        "Estimated input/output inference cost "
        f"(on-demand list price; evaluation excluded): ${cost:.6f}"
    )
