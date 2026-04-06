"""SFT fine-tuning of Qwen3.5-4B on ScreenSpot grounding data with LoRA.

Trains on correct samples from the 122B model evaluation (hit=true in
train_results.jsonl).  Uses TRL's SFTTrainer with a LoRA adapter and
2-GPU DDP via accelerate.

Usage:
    # 2-GPU DDP training (recommended)
    accelerate launch --config_file accelerate_config.yaml train_sft_qwen.py

    # Single-GPU (for testing)
    python train_sft_qwen.py --batch-size 2 --grad-accum 8

All hyperparameters have sensible defaults for 2×H100.  The script:
  1. Filters train_results.jsonl → keeps only hit=true samples
  2. Loads the corresponding images from HuggingFace
  3. Builds a conversational SFT dataset with reasoning traces:
     system + user[image] → assistant(<think>reasoning</think> CLICK)
  4. Trains a LoRA adapter on Qwen3.5-4B
"""

import argparse
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv

load_dotenv()
from datasets import Dataset, load_dataset
from peft import LoraConfig
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer

from config import CLICK_PATTERN, DATASET_CONFIGS, SYSTEM_PROMPT


# ── Data helpers ──────────────────────────────────────────────────────────

def load_correct_samples(
    results_path: str, deduplicate: bool = False,
) -> list[dict]:
    """Load results JSONL and keep only correct predictions (hit=true)."""
    with open(results_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    total = len(records)
    records = [r for r in records if r.get("hit", False)]
    hits = len(records)
    print(f"Loaded {total} results → {hits} correct ({hits / total:.1%})")

    if deduplicate:
        seen: OrderedDict[int, dict] = OrderedDict()
        for r in records:
            if r["idx"] not in seen:
                seen[r["idx"]] = r
        records = list(seen.values())
        print(f"Deduplicated to {len(records)} unique samples")

    return records


def _split_reasoning(raw_response: str, coords: list[float]) -> tuple[str, str]:
    """Split a 122B model response into (reasoning, click_text).

    The reasoning (everything before the final CLICK) becomes the thinking
    trace; the CLICK itself is the visible answer.
    """
    match = list(CLICK_PATTERN.finditer(raw_response))
    if match:
        last = match[-1]
        reasoning = raw_response[: last.start()].strip()
        click_text = last.group(0)
    else:
        reasoning = ""
        click_text = f"CLICK({coords[0]:.4f}, {coords[1]:.4f})"
    return reasoning, click_text


def build_sft_dataset(records: list[dict], mode: str) -> Dataset:
    """Build an HF Dataset with ``messages`` + ``images`` columns.

    The 122B model's raw reasoning is placed into the assistant message's
    ``reasoning_content`` field so the Qwen3.5 chat template renders it
    inside ``<think>...</think>`` tags.  The model thus learns to reason
    before answering.
    """
    cfg = DATASET_CONFIGS[mode]
    print(f"Loading images from {cfg.hf_name} (split={cfg.split}) ...")
    hf_ds = load_dataset(cfg.hf_name, split=cfg.split, streaming=False)

    unique_indices = sorted({r["idx"] for r in records})
    image_cache: dict[int, Image.Image] = {}
    for idx in tqdm(unique_indices, desc="Fetching images"):
        image_cache[idx] = hf_ds[idx]["image"]

    messages_list: list[list[dict]] = []
    images_list: list[list[Image.Image]] = []
    n_with_reasoning = 0

    for r in records:
        coords = r.get("predicted_coords_norm")
        if coords is None:
            continue

        reasoning, click_text = _split_reasoning(
            r.get("raw_response", ""), coords,
        )
        if reasoning:
            n_with_reasoning += 1

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Instruction: {r['instruction']}"},
                ],
            },
            {
                "role": "assistant",
                "content": click_text,
                "reasoning_content": reasoning,
            },
        ]

        messages_list.append(messages)
        images_list.append([image_cache[r["idx"]]])

    print(
        f"Built dataset with {len(messages_list)} training examples "
        f"({n_with_reasoning} with reasoning traces)"
    )
    return Dataset.from_dict({"messages": messages_list, "images": images_list})


# ── Collate function ──────────────────────────────────────────────────────

def make_collate_fn(processor: AutoProcessor, max_length: int = 4096):
    """Return a collate function that tokenises VLM conversations.

    Loss is computed on the **assistant response only** (reasoning + CLICK).
    The prompt (system + user + ``<|im_start|>assistant\\n<think>\\n``) is
    masked out so the model learns to *generate* reasoning, not to parrot
    the prompt.
    """

    def collate_fn(examples: list[dict]) -> dict:
        full_texts: list[str] = []
        prompt_texts: list[str] = []
        all_images: list[list[Image.Image]] = []

        for ex in examples:
            msgs = ex["messages"]

            full_text = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
            )
            # Prompt = system + user turns with the thinking preamble.
            # Default enable_thinking=True → prompt ends with
            #   <|im_start|>assistant\n<think>\n
            # so the model learns to produce reasoning + </think> + CLICK.
            prompt_msgs = msgs[:-1]
            prompt_text = processor.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_texts.append(full_text)
            prompt_texts.append(prompt_text)
            imgs = [img.convert("RGB") for img in ex["images"]]
            all_images.append(imgs)

        batch = processor(
            text=full_texts,
            images=all_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        labels = batch["input_ids"].clone()

        # Mask everything before the assistant response (per sample)
        for i, (ptxt, imgs) in enumerate(zip(prompt_texts, all_images)):
            prompt_enc = processor(
                text=[ptxt], images=imgs, return_tensors="pt",
            )
            prompt_len = prompt_enc["input_ids"].shape[1]
            labels[i, :prompt_len] = -100

        # Mask padding
        labels[labels == processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="SFT + LoRA fine-tuning of Qwen3.5-4B on ScreenSpot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--results", default="train_results.jsonl",
                    help="Results JSONL from 122B model evaluation")
    p.add_argument("-m", "--mode", choices=["normal", "pro"], default="pro",
                    help="ScreenSpot variant")
    p.add_argument("--deduplicate", action="store_true",
                    help="Keep only one sample per unique image idx")

    # Model
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument("--output-dir", default="qwen35_4b_screenspot_lora")

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4,
                    help="Per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=4,
                    help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=4096,
                    help="Max sequence length (incl. image + reasoning tokens)")

    # Weights & Biases
    p.add_argument("--wandb-project", default="screenspot-sft",
                    help="W&B project name")
    p.add_argument("--wandb-run-name", default=None,
                    help="W&B run name (auto-generated if omitted)")
    p.add_argument("--wandb-entity", default=None,
                    help="W&B team/user entity")
    p.add_argument("--no-wandb", action="store_true",
                    help="Disable W&B logging (TensorBoard only)")

    args = p.parse_args()

    # ── Data ──────────────────────────────────────────────────────────────
    records = load_correct_samples(args.results, deduplicate=args.deduplicate)
    dataset = build_sft_dataset(records, args.mode)

    # ── Model + processor ─────────────────────────────────────────────────
    print(f"Loading model: {args.model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_id, trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "right"

    # ── LoRA ──────────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # ── Weights & Biases ────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    report_to = ["tensorboard", "wandb"] if use_wandb else ["tensorboard"]

    if use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "model_id": args.model_id,
                "mode": args.mode,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "max_length": args.max_length,
                "train_samples": len(dataset),
                "deduplicate": args.deduplicate,
            },
        )

    # ── SFT config ────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=report_to,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )

    collate_fn = make_collate_fn(processor, max_length=args.max_length)

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        processing_class=processor,
        peft_config=peft_config,
    )

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    trainer.train()

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\nAdapter + processor saved to {Path(args.output_dir).resolve()}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
