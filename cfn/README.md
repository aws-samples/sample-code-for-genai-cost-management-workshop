# Workshop Infrastructure

The [`workshop-stack.yaml`](workshop-stack.yaml) CloudFormation template provisions the full workshop environment.

## What It Creates

| Resource Group | What It Creates |
|---------------|-----------------|
| Networking | VPC, public subnet, Internet Gateway, route table |
| Compute | EC2 instance (Graviton/AMD) with Code Editor IDE served via CloudFront |
| Security | Security group (CloudFront-only ingress), Secrets Manager secret for editor auth |
| Bootstrap (SSM) | SSM Document that installs packages, Python venv, Docker, Node.js, AWS CDK, uv, Claude Code CLI, and clones the workshop repo |
| Bedrock Logging | CloudWatch log group + IAM role; enables model invocation logging (text only) |
| Athena | Lambda-backed custom resource that sets the primary workgroup output location |
| LiteLLM | Docker Compose service started during bootstrap, proxied via Nginx at `/litellm/` |

## Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `InstanceType` | `t4g.large` | EC2 instance size |
| `InstanceOperatingSystem` | `AmazonLinux-2023` | OS (AL2023, Ubuntu 22/24) |
| `InstanceVolumeSize` | `40` GB | Root volume size |
| `RepoUrl` | (empty) | Git repo to clone into the home folder |
| `HomeFolder` | `/workshop` | Working directory inside the instance |
| `DevServerPort` | `8081` | Port Nginx proxies for a workshop app |

## Prerequisites

- The stack must be deployed in a region listed in the `AWSRegionsPrefixListID` mapping: `eu-west-1`, `us-east-1`, `us-east-2`, or `us-west-2`.
- Your IAM principal needs permissions to create IAM roles, EC2 instances, Lambda functions, CloudFront distributions, and Secrets Manager secrets.
- Bedrock model access must be enabled for the models used in the workshop (Nova 2 Lite, Claude Sonnet 4.6, Claude Haiku 4.5).

## Deploying the Stack

The template exceeds 51,200 bytes, so it must be uploaded to S3 before deployment. Create a bucket (or use an existing one), then deploy:

```bash
# **Pick a unique name** for the S3 bucket that will hold the template
export CFN_BUCKET=my-cfn-templates-bucket

# Create the bucket (one-time setup)
aws s3 mb s3://$CFN_BUCKET

# Deploy the stack
aws cloudformation deploy \
  --template-file cfn/workshop-stack.yaml \
  --stack-name genai-cost-workshop \
  --s3-bucket $CFN_BUCKET \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    InstanceType=t4g.large \
    HomeFolder=/workshop \
    RepoUrl=https://github.com/aws-samples/sample-code-for-genai-cost-management-workshop
```

The bucket must be in the same region you are deploying to.

The stack takes approximately 10-15 minutes to complete. Once deployed, retrieve the Code Editor URL from the stack outputs:

```bash
aws cloudformation describe-stacks \
  --stack-name genai-cost-workshop \
  --query "Stacks[0].Outputs" \
  --output table
```

## Outputs

| Output | Description |
|--------|-------------|
| `URL` | Full Code Editor URL with auth token |
| `CodeEditorURL` | Base URL for the Code Editor |
| `LiteLLMURL` | LiteLLM proxy dashboard |

## Cleanup

To delete all resources created by the stack and the S3 bucket:

```bash
# Delete the CloudFormation stack
aws cloudformation delete-stack --stack-name genai-cost-workshop

# Wait for deletion to complete
aws cloudformation wait stack-delete-complete --stack-name genai-cost-workshop

# Remove the template bucket and its contents
aws s3 rb s3://$CFN_BUCKET --force
```
