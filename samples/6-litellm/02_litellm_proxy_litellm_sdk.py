"""
LiteLLM Proxy - LiteLLM SDK Client

Demonstrates calling the LiteLLM proxy using the LiteLLM Python SDK.
The SDK adds cost-management features on top of the standard OpenAI-compatible interface.

Advantages over OpenAI SDK:
- Native metadata parameter for tagging (user, team, project)
- Token counting before sending (budget pre-checks)
- Client-side fallbacks between models
- Switch providers by changing a single model string

Prerequisites:
- LiteLLM proxy running on localhost:4000 (see docker-compose.yml)
"""

import litellm

# Point LiteLLM SDK at the proxy
litellm.api_base = "http://localhost:4000/litellm"
litellm.api_key = "sk-1234"

# --- Simple completion ---
print("=" * 60)
print("LiteLLM SDK → LiteLLM Proxy → Amazon Bedrock")
print("=" * 60)

response = litellm.completion(
    model="claude-haiku-4-5",
    messages=[{"role": "user", "content": "What is generative AI? Answer in one sentence."}],
    max_tokens=100
)

print(f"\nModel: {response.model}")
print(f"Response: {response.choices[0].message.content}")
print(f"Tokens - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}")

# Cost tracking - unique to LiteLLM SDK
cost = litellm.completion_cost(completion_response=response)
print(f"Estimated Cost: ${cost:.6f}")

# --- With request tags for spend attribution ---
print("\n" + "=" * 60)
print("With request tags for spend attribution")
print("=" * 60)

# Tags are LiteLLM's mechanism for custom spend tracking.
# They appear in the request_tags field of spend logs and can have budgets set against them.
# Pass tags via the x-litellm-tags header (comma-separated).

response = litellm.completion(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "What is Amazon Bedrock? Answer in one sentence."}],
    max_tokens=100,
    extra_headers={
        "x-litellm-tags": "cost-workshop,team-alpha,project-genai,environment-dev,litellm-sdk"
    }
)

print(f"\nModel: {response.model}")
print(f"Response: {response.choices[0].message.content}")
print(f"Tokens - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}")
print(f"Tags sent: cost-workshop, team-alpha, project-genai, environment-dev, litellm-sdk")

cost = litellm.completion_cost(completion_response=response)
print(f"Estimated Cost: ${cost:.6f}")

# --- Token counting before sending (budget pre-check) ---
print("\n" + "=" * 60)
print("Token counting (pre-send budget check)")
print("=" * 60)

messages = [
    {"role": "system", "content": "You are a helpful assistant that explains cloud concepts."},
    {"role": "user", "content": "Explain the shared responsibility model in AWS in detail."}
]

token_count = litellm.token_counter(model="claude-haiku-4-5", messages=messages)
print(f"\nEstimated input tokens: {token_count}")
print("→ Use this to check against budget limits before sending requests")

# --- Comparing models with tags ---
print("\n" + "=" * 60)
print("Cost comparison across models (tagged)")
print("=" * 60)

models = ["claude-haiku-4-5", "claude-sonnet-4-6"]
prompt = [{"role": "user", "content": "What is cloud computing? One sentence."}]

for model in models:
    response = litellm.completion(
        model=model,
        messages=prompt,
        max_tokens=50,
        extra_headers={
            "x-litellm-tags": "cost-comparison-test,cost-workshop,litellm-sdk"
        }
    )
    print(f"\n  {model}:")
    print(f"    Response: {response.choices[0].message.content}")
    print(f"    Tokens - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}")
    cost = litellm.completion_cost(completion_response=response)
    print(f"    Estimated Cost: ${cost:.6f}")

print("\n→ Check aggregated spend by tags at http://localhost:4000/litellm/ui")
print("→ Tags appear in the 'Tags' section of each request trace")
