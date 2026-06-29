# Projects

Sample code for creating and tagging projects for cost attribution on the `bedrock-mantle` endpoint (OpenAI-compatible API).

## Overview

Projects provide resource-level tagging for workloads using the OpenAI-compatible API (Chat Completions, Responses) on the `bedrock-mantle` endpoint. Tags applied to projects flow to Cost Explorer and CUR 2.0.

## Tags Used

| Tag Key | Example Value | Purpose |
|---------|---------------|---------|
| `bedrock:projects:Application` | `OrderFulfillmentAgent` | E-commerce order orchestration |
| `bedrock:projects:Environment` | `Staging` | Track by environment |
| `bedrock:projects:Team` | `CommerceEngineering` | Attribute costs to a team |
| `bedrock:projects:CostCenter` | `ECOM-3100` | Map to financial cost center |

These tags use the `bedrock:projects:` prefix and are set when creating or updating the project. They appear in Cost Explorer and CUR 2.0 once activated as cost allocation tags.

## How It Works

1. Create a project in Amazon Bedrock
2. Tag the project with attributes like `bedrock:projects:Application`, `bedrock:projects:Environment`, `bedrock:projects:Team`, `bedrock:projects:CostCenter`
3. Route OpenAI-compatible API calls through the project
4. After ~24 hours, the tags become available for activation in AWS Billing > Cost Allocation Tags
5. Activate the cost allocation tags
6. Make additional API calls through the project
7. After ~24 hours, costs appear in Cost Explorer and CUR 2.0, grouped by project tags

## Best For

- Teams using the OpenAI SDK through the bedrock-mantle endpoint
- Applications built on the OpenAI-compatible Chat Completions or Responses API
