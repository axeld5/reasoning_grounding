"""Evaluate Qwen3.5-4B on ScreenSpot locally (transformers) or via OpenRouter.

Usage (local – base model):
    python eval_local_qwen.py -i split_eval.jsonl -o eval_4b_results.jsonl --backend local

Usage (local – fine-tuned LoRA):
    python eval_local_qwen.py -i split_eval.jsonl -o eval_4b_lora_results.jsonl \
        --backend local --adapter-path ./qwen35_4b_screenspot_lora

Usage (OpenRouter – concurrent):
    python eval_local_qwen.py -i split_eval.jsonl -o eval_4b_results.jsonl \
        --backend openrouter -c 10

The model runs in **thinking mode** by default: it produces a reasoning
trace inside <think>...</think> tags before the final CLICK(x, y).
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from config import CLICK_PATTERN, DATASET_CONFIGS, SYSTEM_PROMPT
from metrics import point_in_bbox, print_report


def load_eval_samples(jsonl_path: str) -> list[dict]:
    with open(jsonl_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_images(samples: list[dict], mode: str) -> dict[int, Image.Image]:
    cfg = DATASET_CONFIGS[mode]
    indices = sorted({s["idx"] for s in samples})
    print(f"Loading {len(indices)} images from {cfg.hf_name} (split={cfg.split}) ...")
    ds = load_dataset(cfg.hf_name, split=cfg.split, streaming=False)
    images = {}
    for idx in tqdm(indices, desc="Fetching images"):
        images[idx] = ds[idx]["image"]
    return images


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_and_score(response: str, sample: dict) -> dict:
    """Parse CLICK from response and check hit against bbox."""
    response = strip_thinking(response)
    coords = None
    matches = CLICK_PATTERN.findall(response)
    if matches:
        x, y = matches[-1]
        coords = (float(x), float(y))

    img_w, img_h = sample["img_size"]
    pred_px = None
    hit = False
    if coords:
        nx, ny = coords
        pred_px = [nx * img_w, ny * img_h]
        hit = point_in_bbox(pred_px[0], pred_px[1], sample["bbox"], img_w, img_h)

    return {
        "idx": sample["idx"],
        "rollout_id": 0,
        "file_name": sample["file_name"],
        "instruction": sample["instruction"],
        "bbox": sample["bbox"],
        "img_size": sample["img_size"],
        "data_type": sample["data_type"],
        "data_source": sample["data_source"],
        "predicted_coords_norm": list(coords) if coords else None,
        "predicted_coords_px": pred_px,
        "hit": hit,
        "raw_response": response,
    }


def checkpoint(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


# ── Local (transformers) backend ──────────────────────────────────────────

def evaluate_local(samples, images, args):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading model: {args.model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter on top of the base model when provided
    if args.adapter_path:
        from peft import PeftModel

        print(f"Loading LoRA adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    processor_path = args.adapter_path or args.model_id
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    model.eval()

    results: list[dict] = []
    correct = total = 0
    out_path = Path(args.output)

    for sample in tqdm(samples, desc="Evaluating (local)"):
        img = images[sample["idx"]]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Instruction: {sample['instruction']}"},
                ],
            },
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text], images=[img], return_tensors="pt", padding=True,
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        response = processor.decode(
            output_ids[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        record = _parse_and_score(response, sample)
        results.append(record)

        total += 1
        if record["hit"]:
            correct += 1

        if total % 10 == 0:
            acc = correct / total
            tqdm.write(f"  [{total}] running accuracy: {acc:.1%} ({correct}/{total})")
        if total % 50 == 0:
            checkpoint(results, out_path)

    return results


# ── OpenRouter backend (async + concurrent) ──────────────────────────────

def evaluate_openrouter(samples, images, args):
    """Run evaluation via OpenRouter with async concurrency.

    Make sure OPENROUTER_API_KEY is set (in .env or environment) or pass --api-key.
    The model ID on OpenRouter is typically 'qwen/qwen3.5-4b'.
    Check https://openrouter.ai/models for the latest availability.
    """
    return asyncio.run(_evaluate_openrouter_async(samples, images, args))


async def _evaluate_openrouter_async(samples, images, args):
    import base64
    import io
    import os

    import openai

    from tqdm.asyncio import tqdm as atqdm

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY or pass --api-key")

    client = openai.AsyncOpenAI(
        api_key=api_key, base_url="https://openrouter.ai/api/v1",
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    def encode_image(pil_img: Image.Image) -> str:
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=85)
        b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    async def _call(sample: dict) -> dict:
        img = images[sample["idx"]]
        data_url = encode_image(img)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": f"Instruction: {sample['instruction']}"},
                ],
            },
        ]

        for attempt in range(2):
            try:
                async with semaphore:
                    resp = await client.chat.completions.create(
                        model=args.model_id,
                        messages=messages,
                        max_tokens=args.max_new_tokens,
                        temperature=0.0,
                    )
                text = resp.choices[0].message.content or ""
                break
            except (openai.RateLimitError, openai.APIStatusError) as e:
                if attempt == 0 and "429" in str(e):
                    await asyncio.sleep(60)
                else:
                    atqdm.write(f"API error for idx={sample['idx']}: {e}")
                    text = ""
                    break
            except Exception as e:
                atqdm.write(f"API error for idx={sample['idx']}: {e}")
                text = ""
                break

        return _parse_and_score(text, sample)

    coros = [_call(s) for s in samples]
    results: list[dict] = []
    correct = total = 0
    out_path = Path(args.output)

    async for coro in atqdm(
        asyncio.as_completed(coros), total=len(coros), desc="Evaluating (OpenRouter)",
    ):
        record = await coro
        results.append(record)

        total += 1
        if record["hit"]:
            correct += 1

        if total % 10 == 0:
            acc = correct / total
            atqdm.write(f"  [{total}] running accuracy: {acc:.1%} ({correct}/{total})")
        if total % 50 == 0:
            checkpoint(results, out_path)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate Qwen3.5-4B on ScreenSpot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-i", "--input", default="split_eval.jsonl",
        help="Input JSONL with evaluation samples (from holdout split)",
    )
    p.add_argument("-o", "--output", default="eval_4b_results.jsonl")
    p.add_argument(
        "-m", "--mode", choices=["normal", "pro"], default="pro",
        help="ScreenSpot variant (determines which HF dataset to load images from)",
    )
    p.add_argument("-n", "--num-samples", type=int, default=None)
    p.add_argument(
        "--backend", choices=["local", "openrouter"], default="local",
        help="'local' uses transformers on GPU; 'openrouter' calls the API",
    )
    p.add_argument(
        "--model-id", default=None,
        help="Base HF model ID for local, or OpenRouter model ID. "
             "Defaults: Qwen/Qwen3.5-4B (local), qwen/qwen3.5-9b (openrouter)",
    )
    p.add_argument(
        "--adapter-path", default=None,
        help="Path to a LoRA adapter directory (local backend only). "
             "Loaded on top of --model-id.",
    )
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument(
        "-c", "--concurrency", type=int, default=10,
        help="Max concurrent API requests (OpenRouter backend only)",
    )
    p.add_argument("--api-key", default=None, help="OpenRouter API key")

    args = p.parse_args()

    if args.model_id is None:
        args.model_id = (
            "Qwen/Qwen3.5-4B" if args.backend == "local" else "qwen/qwen3.5-9b"
        )

    samples = load_eval_samples(args.input)
    if args.num_samples:
        samples = samples[: args.num_samples]
    print(f"Evaluating {len(samples)} samples with {args.model_id} ({args.backend})")

    images = load_images(samples, args.mode)

    if args.backend == "local":
        results = evaluate_local(samples, images, args)
    else:
        results = evaluate_openrouter(samples, images, args)

    out_path = Path(args.output)
    checkpoint(results, out_path)
    print_report(results, errors=0, output_path=str(out_path.resolve()), num_rollouts=1)


if __name__ == "__main__":
    main()
