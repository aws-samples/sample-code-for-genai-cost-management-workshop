# Sample Code

This folder contains code samples for each cost attribution method covered in the workshop.

## Samples

| # | Method | Folder | Description |
|---|--------|--------|-------------|
| 1 | IAM Principal Attribution | [1-iam-principal-attribution](1-iam-principal-attribution/) | Tag IAM users/roles to track per-developer spend |
| 2 | Application Inference Profiles | [2-application-inference-profiles](2-application-inference-profiles/) | Create tagged profiles for per-application cost isolation |
| 3 | Workspaces | [3-workspaces](3-workspaces/) | Workspaces for Anthropic Messages API on bedrock-mantle |
| 4 | Projects | [4-projects](4-projects/) | Projects for OpenAI-compatible API workloads on bedrock-mantle |
| 5 | Per-Request Metadata Tagging | [5-per-request-metadata-tagging](5-per-request-metadata-tagging/) | Per-request metadata for tenant/task-level attribution |
| 6 | IAM Identity Log Attribution | [6-iam-identity-log-attribution](6-iam-identity-log-attribution/) | Model invocation logging with IAM caller identity for near real-time token tracking |
| 7 | LiteLLM | [7-litellm](7-litellm/) | Third-party proxy for real-time multi-provider cost tracking |

## Tag Naming Convention

All samples follow the `bedrock:<method>:<tag name>` pattern:

| Method | Tag Prefix | Example |
|--------|-----------|---------|
| IAM Principal Attribution | `bedrock:iam-principal:` | `bedrock:iam-principal:Team` |
| Application Inference Profiles | `bedrock:inference-profiles:` | `bedrock:inference-profiles:Team` |
| Workspaces | `bedrock:workspaces:` | `bedrock:workspaces:Team` |
| Projects | `bedrock:projects:` | `bedrock:projects:Team` |

## Setup

See the [main README](../README.md) for environment setup, virtual environment creation, and dependency installation.
