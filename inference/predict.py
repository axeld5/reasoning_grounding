import asyncio
from typing import Any

from PIL import Image

from config import CLICK_PATTERN, DEFAULT_TEMPERATURE, SYSTEM_PROMPT, ModelConfig
from inference.image import encode_image


def parse_click(response_text: str) -> tuple[float, float] | None:
    """Extract the last CLICK(x, y) from the model response (normalised [0,1])."""
    matches = CLICK_PATTERN.findall(response_text)
    if not matches:
        return None
    x, y = matches[-1]
    return float(x), float(y)


async def _predict_anthropic(
    client: Any, semaphore: asyncio.Semaphore,
    b64: str, media_type: str, instruction: str,
    model_cfg: ModelConfig, temperature: float,
) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
            {"type": "text", "text": f"Instruction: {instruction}"},
        ],
    }]
    async with semaphore:
        response = await client.messages.create(
            model=model_cfg.model_id, max_tokens=model_cfg.max_tokens,
            system=SYSTEM_PROMPT, messages=messages, temperature=temperature,
        )
    return response.content[0].text


async def _predict_openrouter(
    client: Any, semaphore: asyncio.Semaphore,
    b64: str, media_type: str, instruction: str,
    model_cfg: ModelConfig, temperature: float,
) -> str:
    data_url = f"data:{media_type};base64,{b64}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": f"Instruction: {instruction}"},
            ],
        },
    ]
    async with semaphore:
        response = await client.chat.completions.create(
            model=model_cfg.model_id, max_tokens=model_cfg.max_tokens,
            messages=messages, temperature=temperature,
        )
    return response.choices[0].message.content or ""


async def predict_click(
    client: Any,
    semaphore: asyncio.Semaphore,
    pil_image: Image.Image,
    instruction: str,
    model_cfg: ModelConfig,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[str, tuple[float, float] | None]:
    """Send screenshot + instruction to the model. Returns (raw_text, coords|None)."""
    b64, media_type = encode_image(pil_image)

    if model_cfg.api_backend == "openrouter":
        text = await _predict_openrouter(
            client, semaphore, b64, media_type, instruction, model_cfg, temperature,
        )
    else:
        text = await _predict_anthropic(
            client, semaphore, b64, media_type, instruction, model_cfg, temperature,
        )

    return text, parse_click(text)
