"""
Medium-Effort Optimization - Lever 12: Batch Inference

Amazon Bedrock batch inference processes large volumes of data asynchronously at
50% of the on-demand price. It is ideal for workloads that are not latency
sensitive - embeddings, entity extraction, LLM-as-judge evaluations, and
text categorization or summarization for reporting. For eligible traffic this is
the lowest-effort, highest-confidence optimization: same model, same output,
half the price.

You will learn how to:
- Build a JSONL input file in the Converse batch format (recordId + modelInput)
- Upload it to S3 and submit a batch job with create_model_invocation_job
- Poll job status and find the output location

How it works:
1. Write a .jsonl file where each line is one request: a recordId and a
   modelInput in Converse request format.
2. Upload it to an S3 input location.
3. Submit the job with an IAM service role Bedrock can assume to read the input
   and write the output. Jobs move Submitted -> InProgress -> Completed/Failed.
4. Read the .out file from the S3 output location when the job completes.

NOTE: A live batch job requires an S3 bucket and an IAM service role, runs
asynchronously (minutes to hours), and has a minimum record count (commonly 100
records per job). This script ALWAYS builds and shows the JSONL locally. It only
uploads and submits a real job when you set both environment variables:

    export BEDROCK_BATCH_BUCKET=your-s3-bucket
    export BEDROCK_BATCH_ROLE_ARN=arn:aws:iam::<account>:role/<batch-service-role>

Without them, it prints the sample records and the exact API call it would make,
then exits cleanly - so you can inspect the format without provisioning anything.

Prerequisites:
- An AWS account with Amazon Bedrock access
- For a live job: an S3 bucket and a Bedrock batch service role (see
  https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-permissions.html)
- IAM credentials with bedrock:CreateModelInvocationJob and s3 read/write
- Dependencies installed via: pip install -r requirements.txt
"""

import os
import json
import time
import uuid
import boto3

# ============================================================
# Configuration
# ============================================================

REGION = os.environ.get("AWS_REGION", "us-east-1")

# Batch jobs support cross-region inference profiles. Haiku 4.5 is a good,
# cheap fit for high-volume categorization/summarization work.
MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# Live-submission gating: both must be set to actually upload + submit a job.
BUCKET = os.environ.get("BEDROCK_BATCH_BUCKET")
ROLE_ARN = os.environ.get("BEDROCK_BATCH_ROLE_ARN")

# Bedrock batch jobs require a minimum number of records (commonly 100).
NUM_RECORDS = 100


# ============================================================
# Build the JSONL input in Converse batch format
# ============================================================

def build_batch_records(n: int) -> list[dict]:
    """Build n batch records. Each is a recordId + a Converse-format modelInput.

    Here the offline task is short product-review sentiment classification - a
    classic non-latency-sensitive workload well suited to batch.
    """
    reviews = [
        "Delivery was fast and the item works great. Very happy with it.",
        "Stopped working after a week and support never replied. Avoid.",
        "It's fine. Does the job but nothing special for the price.",
        "Exceeded my expectations - the build quality is excellent.",
        "Arrived damaged and the return process was a nightmare.",
    ]
    records = []
    for i in range(n):
        review = reviews[i % len(reviews)]
        records.append(
            {
                "recordId": f"REVIEW{i + 1:07d}",
                "modelInput": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": "Classify the sentiment of this product review as "
                                    "positive, neutral, or negative. Reply with one word.\n\n"
                                    f"Review: {review}"
                                }
                            ],
                        }
                    ],
                    "inferenceConfig": {"maxTokens": 5},
                },
            }
        )
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ============================================================
# Submit + monitor a live batch job
# ============================================================

def submit_batch_job(local_jsonl: str) -> str:
    """Upload the JSONL to S3 and submit a batch job. Returns the job ARN."""
    s3 = boto3.client("s3", region_name=REGION)
    bedrock = boto3.client("bedrock", region_name=REGION)

    suffix = str(uuid.uuid4())[:8]
    input_key = f"batch-inference/input/reviews-{suffix}.jsonl"
    output_prefix = f"batch-inference/output/{suffix}/"

    print(f"  Uploading {local_jsonl} -> s3://{BUCKET}/{input_key}")
    s3.upload_file(local_jsonl, BUCKET, input_key)

    response = bedrock.create_model_invocation_job(
        roleArn=ROLE_ARN,
        modelId=MODEL_ID,
        jobName=f"review-sentiment-batch-{suffix}",
        inputDataConfig={"s3InputDataConfig": {"s3Uri": f"s3://{BUCKET}/{input_key}"}},
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{BUCKET}/{output_prefix}"}},
    )
    job_arn = response["jobArn"]
    print(f"  Submitted batch job: {job_arn}")
    return job_arn


def poll_once(job_arn: str) -> str:
    """Fetch and print the current job status a single time."""
    bedrock = boto3.client("bedrock", region_name=REGION)
    status = bedrock.get_model_invocation_job(jobIdentifier=job_arn)["status"]
    print(f"  Job status: {status}")
    return status


# ============================================================
# Main
# ============================================================

def main():
    print("--- Batch Inference: half-price offline processing ---\n")

    records = build_batch_records(NUM_RECORDS)
    local_jsonl = "batch_input.jsonl"
    write_jsonl(records, local_jsonl)
    print(f"Built {len(records)} records in {local_jsonl} (Converse batch format).")
    print("First record:")
    print(json.dumps(records[0], indent=2))
    print()

    if not (BUCKET and ROLE_ARN):
        print("BEDROCK_BATCH_BUCKET / BEDROCK_BATCH_ROLE_ARN not set - skipping live submission.")
        print("The job that WOULD be submitted:")
        print(f"  modelId={MODEL_ID}")
        print(f"  inputDataConfig=s3://<bucket>/batch-inference/input/reviews-<id>.jsonl")
        print(f"  outputDataConfig=s3://<bucket>/batch-inference/output/<id>/")
        print(f"  roleArn=<your Bedrock batch service role>")
        print("\n  Set both env vars to upload the JSONL and submit a real job.")
        print("  Batch inference bills at 50% of on-demand for the same model and output.")
        os.remove(local_jsonl)
        return

    try:
        job_arn = submit_batch_job(local_jsonl)
        # A single status check - a real job runs asynchronously for minutes to hours.
        time.sleep(2)
        poll_once(job_arn)
        print("\n  Poll get_model_invocation_job until Completed, then read the .out")
        print("  file from the S3 output location. Batch bills at 50% of on-demand.")
    finally:
        if os.path.exists(local_jsonl):
            os.remove(local_jsonl)

    print("\n--- Done ---")


if __name__ == "__main__":
    main()
