"""
LiteLLM configuration helper.
Reads from .env and configures the LLM provider properly.
Supports DeepSeek, Claude, and OpenAI-compatible endpoints.
"""
import os
import litellm

# —— Load from env ——
API_KEY = os.getenv("LITELLM_API_KEY", "")
API_BASE = os.getenv("LITELLM_API_BASE", "")
MODEL = os.getenv("LITELLM_MODEL", "deepseek/deepseek-chat")

# —— Provider-specific setup ——
# When using deepseek/ prefix, LiteLLM looks for DEEPSEEK_API_KEY.
# Map our generic LITELLM_API_KEY → provider key.
if MODEL.startswith("deepseek/"):
    if API_KEY and not os.getenv("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = API_KEY
    if API_BASE and not os.getenv("DEEPSEEK_API_BASE"):
        os.environ["DEEPSEEK_API_BASE"] = API_BASE

elif MODEL.startswith("claude"):
    if API_KEY and not os.getenv("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = API_KEY

elif MODEL.startswith("openai/"):
    if API_KEY and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = API_KEY


def get_model() -> str:
    """Return the configured model string for LiteLLM calls."""
    return MODEL


def get_llm_kwargs(overrides: dict = None) -> dict:
    """Return a base kwargs dict for litellm.completion calls."""
    kwargs = {
        "model": MODEL,
        "temperature": 0.1,
        "max_tokens": 1200,
    }
    if overrides:
        kwargs.update(overrides)
    return kwargs


# Print config on import (helpful for debugging)
if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print(f"API Base: {API_BASE}")
    print(f"Key set: {'yes' if API_KEY else 'no'}")
