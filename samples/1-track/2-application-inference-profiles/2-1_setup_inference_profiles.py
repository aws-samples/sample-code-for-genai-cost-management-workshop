"""
Amazon Bedrock Application Inference Profiles - Profile Setup & Tagging

This script handles the setup for per-application cost attribution:
- Creates application inference profiles wrapping foundation models
- Tags profiles with cost allocation attributes
- Verifies tags are set correctly
- Creates multiple profiles for different applications and models

Tags used: bedrock:inference-profiles:Application, bedrock:inference-profiles:Environment,
           bedrock:inference-profiles:Team, bedrock:inference-profiles:CostCenter

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with permissions for bedrock:CreateInferenceProfile,
  bedrock:ListInferenceProfiles, bedrock:ListTagsForResource,
  bedrock:DeleteInferenceProfile
- Access to Claude or Nova models on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt

After running this script, use 2-2_invoke_models.py to make inference calls
through the profiles.
"""

import os
from typing import Optional
import boto3

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

# Boto3 client for Bedrock management
bedrock_client = boto3.client("bedrock", region_name=REGION)

# The source model to wrap in an inference profile.
# Use a system-defined inference profile ID as the model source.
# This demo uses Global cross-region inference profiles for maximum throughput:
#   - global.amazon.nova-2-lite-v1:0 (Amazon Nova 2 Lite — fast, cost-effective)
#   - global.anthropic.claude-sonnet-4-6 (Claude Sonnet 4.6 — balanced)
#   - global.anthropic.claude-haiku-4-5-20251001-v1:0 (Claude Haiku 4.5 — low latency)

MODELS = {
    "nova-2-lite": "arn:aws:bedrock:us-east-1::inference-profile/global.amazon.nova-2-lite-v1:0",
    "claude-sonnet-4-6": "arn:aws:bedrock:us-east-1::inference-profile/global.anthropic.claude-sonnet-4-6",
    "claude-haiku-4-5": "arn:aws:bedrock:us-east-1::inference-profile/global.anthropic.claude-haiku-4-5-20251001-v1:0",
}

# Default model for the main demo
SOURCE_MODEL_ARN = MODELS["claude-haiku-4-5"]


# ============================================================
# Inference Profile Management Functions
# ============================================================

def create_inference_profile(name: str, tags: dict, description: str = None, model_source: str = None) -> dict:
    """
    Create an application inference profile with cost allocation tags.
    Returns the profile ARN and status.
    """
    # Check if profile already exists
    existing = find_profile_by_name(name)
    if existing:
        print(f"  Profile '{name}' already exists: {existing['inferenceProfileArn']}")
        return existing

    # Build the tags array
    tag_list = [{"key": k, "value": v} for k, v in tags.items()]

    params = {
        "inferenceProfileName": name,
        "modelSource": {"copyFrom": model_source or SOURCE_MODEL_ARN},
        "tags": tag_list,
    }
    if description:
        params["description"] = description

    response = bedrock_client.create_inference_profile(**params)
    print(f"  Created profile '{name}': {response['inferenceProfileArn']}")
    return response


def find_profile_by_name(name: str) -> Optional[dict]:
    """Find an existing application inference profile by name."""
    paginator = bedrock_client.get_paginator("list_inference_profiles")
    for page in paginator.paginate(typeEquals="APPLICATION"):
        for profile in page.get("inferenceProfileSummaries", []):
            if profile["inferenceProfileName"] == name:
                return profile
    return None


def list_inference_profiles() -> list:
    """List all application inference profiles in the account."""
    profiles = []
    paginator = bedrock_client.get_paginator("list_inference_profiles")
    for page in paginator.paginate(typeEquals="APPLICATION"):
        for profile in page.get("inferenceProfileSummaries", []):
            profiles.append(profile)
            print(f"  {profile['inferenceProfileName']} — {profile['inferenceProfileArn']}")
    return profiles


def get_profile_tags(profile_arn: str) -> list:
    """Get the tags for an inference profile."""
    response = bedrock_client.list_tags_for_resource(resourceARN=profile_arn)
    return response.get("tags", [])


def delete_inference_profile(profile_arn: str) -> None:
    """Delete an application inference profile."""
    bedrock_client.delete_inference_profile(inferenceProfileIdentifier=profile_arn)
    print(f"  Deleted profile: {profile_arn}")


# ============================================================
# Main
# ============================================================

def main():
    # Optional: Set to True to delete all existing inference profiles before running
    CLEAN_START = False

    if CLEAN_START:
        print("--- Deleting All Existing Inference Profiles ---")
        profiles = list_inference_profiles()
        for profile in profiles:
            delete_inference_profile(profile["inferenceProfileArn"])
        if not profiles:
            print("  No profiles to delete.")
        print()

    # Step 1: List existing application inference profiles
    print("--- Step 1: Listing Existing Inference Profiles ---")
    profiles = list_inference_profiles()
    if not profiles:
        print("  No application inference profiles found.")
    print()

    # Step 2: Create the main inference profile with cost allocation tags
    print("--- Step 2: Create Primary Inference Profile ---")
    profile = create_inference_profile(
        name="ClaimsProcessingAgent_Production",
        description="Production inference profile for the insurance claims processing agent",
        tags={
            "bedrock:inference-profiles:Application": "ClaimsProcessingAgent",
            "bedrock:inference-profiles:Environment": "Production",
            "bedrock:inference-profiles:Team": "InsurancePlatform",
            "bedrock:inference-profiles:CostCenter": "INS-4400",
        },
    )
    profile_arn = profile["inferenceProfileArn"]
    print(f"  Profile ARN: {profile_arn}\n")

    # Step 3: Verify tags on the profile
    print("--- Step 3: Verify Profile Tags ---")
    tags = get_profile_tags(profile_arn)
    for tag in tags:
        print(f"  {tag['key']} = {tag['value']}")
    print()

    # Step 4: Create additional profiles for different applications and models
    print("--- Step 4: Create Additional Profiles (Different Applications & Models) ---")
    applications = [
        {
            "name": "ClaimsProcessingAgent_Staging",
            "description": "Staging environment for claims agent testing",
            "model": MODELS["claude-haiku-4-5"],
            "tags": {
                "bedrock:inference-profiles:Application": "ClaimsProcessingAgent",
                "bedrock:inference-profiles:Environment": "Staging",
                "bedrock:inference-profiles:Team": "InsurancePlatform",
                "bedrock:inference-profiles:CostCenter": "INS-4401",
            },
        },
        {
            "name": "PolicyRecommendation_Production",
            "description": "Recommends policy options to customers based on their needs",
            "model": MODELS["claude-sonnet-4-6"],
            "tags": {
                "bedrock:inference-profiles:Application": "PolicyRecommendation",
                "bedrock:inference-profiles:Environment": "Production",
                "bedrock:inference-profiles:Team": "InsurancePlatform",
                "bedrock:inference-profiles:CostCenter": "INS-4410",
            },
        },
        {
            "name": "FraudDetectionAgent_Production",
            "description": "Analyzes claims for potential fraud indicators",
            "model": MODELS["nova-2-lite"],
            "tags": {
                "bedrock:inference-profiles:Application": "FraudDetectionAgent",
                "bedrock:inference-profiles:Environment": "Production",
                "bedrock:inference-profiles:Team": "RiskManagement",
                "bedrock:inference-profiles:CostCenter": "INS-4500",
            },
        },
    ]

    for app in applications:
        create_inference_profile(
            name=app["name"],
            description=app["description"],
            tags=app["tags"],
            model_source=app["model"],
        )
    print()

    print("--- Setup Complete ---")
    print("  Inference profiles created and tagged.")
    print("  You can now run 2-2_invoke_models.py to make inference calls through the profiles.")
    print()
    print("  Next steps:")
    print("  1. Run 2-2_invoke_models.py to invoke models through the profiles")
    print("  2. Wait ~24 hours for tags to appear in AWS Billing > Cost Allocation Tags")
    print("  3. Activate the bedrock:inference-profiles:* tags")
    print("  4. View per-application costs in Cost Explorer")


if __name__ == "__main__":
    main()
