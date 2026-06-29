# LiteLLM

Sample code for using LiteLLM as a proxy gateway for real-time, multi-provider cost tracking.

## Overview

LiteLLM is an open-source LLM proxy that sits between your applications and model providers. It provides a unified API layer that routes requests to over 100 LLM providers (including Amazon Bedrock), with built-in real-time cost tracking, budget enforcement, and alerting.

## How It Works

1. Deploy LiteLLM as a proxy in front of your model providers
2. Configure routing rules and tagging (user, team, API key)
3. LiteLLM automatically tracks spend for every request in real time
4. Use the built-in dashboard for cost per user, team, API key, and model

## Important Caveat

Placing any proxy in front of Amazon Bedrock collapses the IAM identity—all calls appear under the proxy's IAM role unless the proxy assumes a per-user role or forwards request metadata. LiteLLM's cost figures are estimates based on published pricing; always reconcile against your AWS bill for invoice-accurate numbers.

## Best For

- Real-time cost visibility and budget enforcement
- Multi-provider environments needing a single pane of glass
- Organizations that need immediate spend alerts rather than next-day billing data
