"""
LiteLLM Proxy - OpenAI SDK Client

Demonstrates calling the LiteLLM proxy using the standard OpenAI Python SDK.
The proxy exposes an OpenAI-compatible API, so any OpenAI SDK client works out of the box.

Advantages:
- No extra dependencies beyond the widely-used OpenAI SDK
- Familiar interface for developers already using OpenAI
- Works with any OpenAI-compatible proxy

Prerequisites:
- LiteLLM proxy running on localhost:4000 (see docker-compose.yml)
"""

from openai import OpenAI

# Point the OpenAI client at the LiteLLM proxy
client = OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000/litellm/v1"
)

# --- Simple completion ---
print("=" * 60)
print("OpenAI SDK → LiteLLM Proxy → Amazon Bedrock")
print("=" * 60)

response = client.chat.completions.create(
    model="claude-haiku-4-5",
    messages=[{"role": "user", "content": "What is generative AI? Answer in one sentence."}],
    max_tokens=100,
    extra_headers={
        "x-litellm-tags": "openai-sdk"
    }
)

print(f"\nModel: {response.model}")
print(f"Response: {response.choices[0].message.content}")
print(f"Tokens - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}")

# --- With request tags for spend attribution ---
print("\n" + "=" * 60)
print("With request tags for spend attribution")
print("=" * 60)

# Use x-litellm-tags header to pass tags for spend tracking.
# Tags appear in the 'Tags' section of each request trace in the UI.
response = client.chat.completions.create(
    model="nova-2-lite",
    messages=[{"role": "user", "content": "What is Amazon Bedrock? Answer in one sentence."}],
    max_tokens=100,
    user="workshop-user-1",
    extra_headers={
        "x-litellm-tags": "cost-workshop,team-alpha,project-genai,environment-dev,openai-sdk"
    }
)

print(f"\nModel: {response.model}")
print(f"Response: {response.choices[0].message.content}")
print(f"Tokens - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}")
print(f"Tags sent via header: cost-workshop, team-alpha, project-genai, environment-dev")
print("\n→ Check spend by tags at http://localhost:4000/litellm/ui")
print("→ Tags appear in the 'Tags' section of each request trace")
