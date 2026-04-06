import asyncio
import os
from typing import Any

import anthropic
import openai

from config import ModelConfig


def create_client(model_cfg: ModelConfig) -> Any:
    """Build the right async API client for the model's backend."""
    if model_cfg.api_backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY environment variable")
        return openai.AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY environment variable")
    return anthropic.AsyncAnthropic(api_key=api_key)


def is_rate_limit_error(e: Exception) -> bool:
    if isinstance(e, (anthropic.RateLimitError, openai.RateLimitError)):
        return True
    err_str = str(e).lower()
    return "429" in str(e) or "rate limit" in err_str


async def call_with_retry(coro_fn, *args, **kwargs):
    """Call an async function with one rate-limit retry (60 s back-off)."""
    for attempt in range(2):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            if attempt == 0 and is_rate_limit_error(e):
                await asyncio.sleep(60)
            else:
                raise
    return None
