from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    max_tokens: int
    api_backend: str  # "anthropic" or "openrouter"


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "claude": ModelConfig(
        model_id="claude-sonnet-4-6",
        max_tokens=2048,
        api_backend="anthropic",
    ),
    "qwen3vl": ModelConfig(
        model_id="qwen/qwen3-vl-235b-a22b-instruct",
        max_tokens=2048,
        api_backend="openrouter",
    ),
    "qwen3.5vl": ModelConfig(
        model_id="qwen/qwen3.5-122b-a10b",
        max_tokens=8192,
        api_backend="openrouter",
    ),
}

DEFAULT_MODEL_MODE = "qwen3.5vl"
