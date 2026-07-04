"""
Amazon Bedrock Application Inference Profiles - Invoke Models Through Profiles

This script invokes Bedrock models through application inference profiles
created by 2-1_setup_inference_profiles.py. Costs are attributed to each
profile's tags in Cost Explorer.

You will learn how to:
- Route Converse API calls through profiles (using profile ARN as model ID)
- Run a multi-step claims processing agent
- Invoke different models through different profiles

Prerequisites:
- Run 2-1_setup_inference_profiles.py first to create the inference profiles
- IAM credentials with permissions for bedrock-runtime:Converse
- Dependencies installed via: pip install -r requirements.txt
"""

import os
from typing import Optional
import boto3

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

# Boto3 clients
bedrock_client = boto3.client("bedrock", region_name=REGION)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)

# Profile names created by 2-1_setup_inference_profiles.py
PROFILE_NAMES = [
    "ClaimsProcessingAgent_Production",
    "ClaimsProcessingAgent_Staging",
    "PolicyRecommendation_Production",
    "FraudDetectionAgent_Production",
]


# ============================================================
# Helper Functions
# ============================================================

def find_profile_by_name(name: str) -> Optional[dict]:
    """Find an existing application inference profile by name."""
    paginator = bedrock_client.get_paginator("list_inference_profiles")
    for page in paginator.paginate(typeEquals="APPLICATION"):
        for profile in page.get("inferenceProfileSummaries", []):
            if profile["inferenceProfileName"] == name:
                return profile
    return None


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
    # Resolve profile ARNs by name
    print("--- Resolving Inference Profiles ---")
    profile_arns = {}
    for name in PROFILE_NAMES:
        profile = find_profile_by_name(name)
        if profile:
            profile_arns[name] = profile["inferenceProfileArn"]
            print(f"  Found: {name}")
        else:
            print(f"  NOT FOUND: {name} — run 2-1_setup_inference_profiles.py first")

    if not profile_arns:
        print("\n  No profiles found. Please run 2-1_setup_inference_profiles.py first.")
        return
    print()

    # Step 1: Simple inference through the primary profile
    primary_arn = profile_arns.get("ClaimsProcessingAgent_Production")
    if primary_arn:
        print("--- Step 1: Inference Through Profile (Converse API) ---")
        result = invoke_with_profile(
            profile_arn=primary_arn,
            system_prompt="You are a helpful assistant. Be concise.",
            user_message="What are the key steps in processing an insurance claim from submission to payout?",
        )
        print(result)
        print()

        # Step 2: Multi-step claims processing agent
        print("--- Step 2: Claims Processing Agent (Multi-step) ---")
        claims_processing_agent(primary_arn)

    # Step 3: Invoke through different profiles
    print("--- Step 3: Invoke Through Multiple Profiles ---")
    queries = {
        "FraudDetectionAgent_Production": "A claim was filed for hail damage but the weather report shows no hail that day. What should I check?",
        "PolicyRecommendation_Production": "A 35-year-old homeowner in Seattle wants comprehensive coverage. What policy tier would you recommend?",
        "ClaimsProcessingAgent_Staging": "Flag potential red flags: CLM-2024-91001 filed 2 days after policy activation, total loss claim on a 15-year-old vehicle valued at $3,200.",
    }

    for name, query in queries.items():
        arn = profile_arns.get(name)
        if not arn:
            print(f"  [{name}] Skipped — profile not found\n")
            continue

        result = invoke_with_profile(profile_arn=arn, user_message=query)
        print(f"  [{name}]")
        print(f"  Query: {query}")
        print(f"  Response: {result}\n")

    print("--- Done ---")
    print("  Next steps:")
    print("  1. Wait ~24 hours for tags to appear in AWS Billing > Cost Allocation Tags")
    print("  2. Activate the bedrock:inference-profiles:* tags")
    print("  3. View per-application costs in Cost Explorer")


if __name__ == "__main__":
    main()
