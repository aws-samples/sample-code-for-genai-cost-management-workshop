# Per-Request Metadata Tagging

Sample code for attaching metadata to individual inference calls and querying invocation logs for per-tenant, per-task cost attribution.

## Overview

Per-request metadata tagging gives you the finest-grained cost attribution. By attaching metadata key-value pairs to each API call, you can track costs down to individual tenants, features, or agent steps—without creating additional AWS resources.

## How It Works

1. Enable model invocation logging in Amazon Bedrock
2. Add `requestMetadata` to Converse API calls (or metadata headers to InvokeModel calls)
3. Metadata appears in model invocation logs alongside token usage
4. Query logs with Athena or visualize in QuickSight

## Important Note

This mechanism provides **token counts**, not dollar costs. You convert tokens to cost using the published pricing for each model.

## Best For

- Per-prompt, per-tenant, and per-experiment tracking on `bedrock-runtime`
- Multi-tenant applications needing tenant-level cost allocation
- AgentCore agent step-level attribution
