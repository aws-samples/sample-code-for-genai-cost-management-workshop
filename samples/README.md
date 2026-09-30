# Sample Code

This folder contains code samples for the workshop, organized into two parts:

- **[1-track/](1-track/)** — cost attribution: know *who is spending what, and where*
- **[2-optimize/](2-optimize/)** — cost reduction: spend less without sacrificing quality (coming soon)

## Track

Attribution methods that show you where your generative AI spend goes.

| # | Method | Folder | Description |
|---|--------|--------|-------------|
| 1 | IAM Principal Attribution | [1-track/1-iam-principal-attribution](1-track/1-iam-principal-attribution/) | Tag IAM users/roles to track per-developer spend |
| 2 | Application Inference Profiles | [1-track/2-application-inference-profiles](1-track/2-application-inference-profiles/) | Create tagged profiles for per-application cost isolation |
| 3 | Workspaces | [1-track/3-workspaces](1-track/3-workspaces/) | Workspaces for Anthropic Messages API on bedrock-mantle |
| 4 | Projects | [1-track/4-projects](1-track/4-projects/) | Projects for OpenAI-compatible API workloads on bedrock-mantle |
| 5 | Per-Request Metadata Tagging | [1-track/5-per-request-metadata-tagging](1-track/5-per-request-metadata-tagging/) | Per-request metadata for tenant/task-level attribution |
| 6 | IAM Identity Log Attribution | [1-track/6-iam-identity-log-attribution](1-track/6-iam-identity-log-attribution/) | Model invocation logging with IAM caller identity for near real-time token tracking |
| 7 | LiteLLM | [1-track/7-litellm](1-track/7-litellm/) | Third-party proxy for real-time multi-provider cost tracking |

## Optimize

Cost-reduction levers that lower your bill once you can see where it goes. See [2-optimize/](2-optimize/) — samples coming soon.

## Tag Naming Convention

All track samples follow the `bedrock:<method>:<tag name>` pattern:

| Method | Tag Prefix | Example |
|--------|-----------|---------|
| IAM Principal Attribution | `bedrock:iam-principal:` | `bedrock:iam-principal:Team` |
| Application Inference Profiles | `bedrock:inference-profiles:` | `bedrock:inference-profiles:Team` |
| Workspaces | `bedrock:workspaces:` | `bedrock:workspaces:Team` |
| Projects | `bedrock:projects:` | `bedrock:projects:Team` |

## Setup

See the [main README](../README.md) for environment setup, virtual environment creation, and dependency installation.
