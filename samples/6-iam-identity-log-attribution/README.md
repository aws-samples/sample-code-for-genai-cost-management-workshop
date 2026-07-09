# IAM Identity Log Attribution

Sample code for using Amazon Bedrock model invocation logging with IAM caller identity to track per-user and per-role token consumption in near real-time via CloudWatch Logs Insights.

## Overview

While IAM principal cost allocation (sample 1) provides billed-dollar visibility in CUR and Cost Explorer with a ~24 hour delay, model invocation logging gives you **near real-time, per-request token-level visibility** by caller identity. Every Bedrock inference call is logged with the full IAM ARN of the caller, input/output token counts, model ID, and timestamps — queryable within seconds via CloudWatch Logs Insights.

This is complementary to CUR-based attribution: use invocation logs for real-time monitoring and per-request detail, and CUR for invoice-reconciled dollar amounts.

## How It Works

1. Enable model invocation logging in Amazon Bedrock (sends logs to CloudWatch Logs)
2. Each log entry automatically includes `identity.arn` — the full IAM ARN of the caller
3. Query CloudWatch Logs Insights to aggregate token usage by role, user, model, or time period
4. Build dashboards and alarms for usage anomalies

## Best For

- Real-time token usage monitoring per developer or application
- Detecting usage spikes before they hit your bill
- Per-request attribution when CUR's daily aggregation isn't granular enough
- Debugging which role/session is driving unexpected costs

## Prerequisites

- Python 3.12+
- Model invocation logging enabled in Amazon Bedrock (CloudWatch Logs destination)
- IAM credentials with permissions for `logs:StartQuery`, `logs:GetQueryResults`, `bedrock:GetModelInvocationLoggingConfiguration`
- Dependencies installed via `pip install -r requirements.txt` from the repository root

## Example CloudWatch Logs Insights Queries

These queries use the IAM roles created by `1-iam-principal-attribution` (Alice, Bob, Carol) to demonstrate per-developer token tracking via invocation logs.

### Token usage by role

```
fields @timestamp, identity.arn, modelId, input.inputTokenCount, output.outputTokenCount
| filter identity.arn like /bedrock-workshop-developer-(alice|bob|carol)/
| parse identity.arn "assumed-role/*/" as role_name
| stats sum(input.inputTokenCount) as total_input_tokens,
        sum(output.outputTokenCount) as total_output_tokens,
        count(*) as request_count
  by role_name, modelId
| sort total_output_tokens desc
```

### Token usage by role (last 7 days, daily breakdown)

Set the time range to **Last 7 days** manually in the CloudWatch Logs Insights console.

```
fields @timestamp, identity.arn, modelId, input.inputTokenCount, output.outputTokenCount
| filter identity.arn like /bedrock-workshop-developer-(alice|bob|carol)/
| parse identity.arn "assumed-role/*/" as role_name
| stats sum(input.inputTokenCount) as total_input_tokens,
        sum(output.outputTokenCount) as total_output_tokens,
        count(*) as request_count
  by bin(1d) as day, role_name, modelId
| sort day desc, total_output_tokens desc
```
