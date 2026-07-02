"""
Amazon Bedrock Application Inference Profiles - Cost Attribution with bedrock-runtime

This sample demonstrates how to use Application Inference Profiles for
per-application cost attribution on the bedrock-runtime endpoint.

You will learn how to:
- Create application inference profiles wrapping foundation models
- Tag profiles with cost allocation attributes
- Route Converse API calls through profiles (using profile ARN instead of model ID)
- Track costs across multiple applications using separate profiles

Tags used: bedrock:inference-profiles:Application, bedrock:inference-profiles:Environment,
           bedrock:inference-profiles:Team, bedrock:inference-profiles:CostCenter

Prerequisites:
- An AWS account with Amazon Bedrock access
- IAM credentials with permissions for bedrock and bedrock-runtime
- Access to Claude models on Amazon Bedrock
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import json
from typing import Optional
import boto3

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

# Boto3 clients for Bedrock management and runtime inference
bedrock_client = boto3.client("bedrock", region_name=REGION)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)

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
# Inference Functions
# ============================================================

def invoke_with_profile(profile_arn: str, user_message: str, system_prompt: str = None) -> str:
    """
    Send a Converse API request through an inference profile.
    Uses the profile ARN as the modelId — this is what routes the call
    through the profile and associates costs with its tags.
    """
    messages = [{"role": "user", "content": [{"text": user_message}]}]

    params = {
        "modelId": profile_arn,
        "messages": messages,
        "inferenceConfig": {"maxTokens": 1024},
    }
    if system_prompt:
        params["system"] = [{"text": system_prompt}]

    response = bedrock_runtime.converse(**params)
    return response["output"]["message"]["content"][0]["text"]


def claims_processing_agent(profile_arn: str) -> None:
    """
    Simulate an insurance claims processing agent with multi-step reasoning.
    All calls are attributed to the same inference profile for cost tracking.
    """
    system_prompt = (
        "You are an insurance claims processing agent. You help assess claims, "
        "verify policy coverage, estimate damages, and recommend next steps. "
        "Be concise, professional, and cite relevant policy sections when applicable."
    )

    # Step 1: Initial claim assessment
    print("  Step 1: Initial claim assessment...")
    result = invoke_with_profile(
        profile_arn=profile_arn,
        system_prompt=system_prompt,
        user_message=(
            "New claim submitted: CLM-2024-78432. "
            "Type: Auto collision. Date: June 15, 2026. "
            "Policyholder: Michael Chen, Policy #AUT-889-2341. "
            "Description: Rear-ended at intersection, airbags deployed, "
            "vehicle towed from scene. No injuries reported. "
            "Perform initial assessment and flag any concerns."
        ),
    )
    print(f"  {result}\n")

    # Step 2: Coverage verification
    print("  Step 2: Verifying coverage...")
    result = invoke_with_profile(
        profile_arn=profile_arn,
        system_prompt=system_prompt,
        user_message=(
            "Policy #AUT-889-2341 details: "
            "Comprehensive + Collision coverage, $500 deductible, "
            "$50,000 per-accident limit. Policy active since 2022, "
            "no lapsed payments. Last claim: 18 months ago (windshield replacement). "
            "Confirm coverage applies to claim CLM-2024-78432."
        ),
    )
    print(f"  {result}\n")

    # Step 3: Damage estimate and recommendation
    print("  Step 3: Damage estimate and recommendation...")
    result = invoke_with_profile(
        profile_arn=profile_arn,
        system_prompt=system_prompt,
        user_message=(
            "Adjuster report for CLM-2024-78432: "
            "Front bumper replacement: $1,200. Hood repair: $800. "
            "Radiator replacement: $650. Airbag replacement (2x): $2,400. "
            "Labor: $1,500. Rental car (7 days): $490. "
            "Total estimate: $7,040. "
            "Vehicle value: $22,000. "
            "Provide recommendation: approve, deny, or escalate. Include payout calculation."
        ),
    )
    print(f"  {result}\n")


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

    # Step 2: Create an inference profile with cost allocation tags
    print("--- Step 2: Create Inference Profile ---")
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

    # Step 4: Invoke through the profile
    print("--- Step 4: Inference Through Profile (Converse API) ---")
    result = invoke_with_profile(
        profile_arn=profile_arn,
        system_prompt="You are a helpful assistant. Be concise.",
        user_message="What are the key steps in processing an insurance claim from submission to payout?",
    )
    print(result)
    print()

    # Step 5: Multi-step claims processing agent
    print("--- Step 5: Claims Processing Agent (Multi-step) ---")
    claims_processing_agent(profile_arn)

    # Step 6: Multiple profiles for different applications (each using a different model)
    print("--- Step 6: Multiple Profiles (Different Applications & Models) ---")
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

    queries = [
        "A claim was filed for hail damage but the weather report shows no hail that day. What should I check?",
        "A 35-year-old homeowner in Seattle wants comprehensive coverage. What policy tier would you recommend?",
        "Flag potential red flags: CLM-2024-91001 filed 2 days after policy activation, total loss claim on a 15-year-old vehicle valued at $3,200.",
    ]

    for app, query in zip(applications, queries):
        prof = create_inference_profile(
            name=app["name"],
            description=app["description"],
            tags=app["tags"],
            model_source=app["model"],
        )
        prof_arn = prof["inferenceProfileArn"]

        result = invoke_with_profile(
            profile_arn=prof_arn,
            user_message=query,
        )
        print(f"  [{app['name']}]")
        print(f"  Query: {query}")
        print(f"  Response: {result}\n")


if __name__ == "__main__":
    main()
