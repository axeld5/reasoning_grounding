"""Run model evaluation on a holdout / validation JSONL."""

import argparse
import asyncio

from dotenv import load_dotenv

load_dotenv()

from config import CROP_SIZE_RATIO, DATASET_CONFIGS, DEFAULT_MODEL_MODE, DEFAULT_ROLLOUTS, DEFAULT_TEMPERATURE, MAX_CONCURRENCY, MODEL_CONFIGS
from eval.runner import run_validation


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate model on holdout validation set")
    p.add_argument("-i", "--input", required=True,
                   help="Path to holdout JSONL (produced by --holdout-output)")
    p.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default=DEFAULT_MODEL_MODE,
                   help=f"Model to use (default: {DEFAULT_MODEL_MODE})")
    p.add_argument("-m", "--mode", choices=list(DATASET_CONFIGS.keys()), default="normal",
                   help="Dataset mode matching the holdout (default: normal)")
    p.add_argument("-n", "--num-samples", type=int, default=None,
                   help="Number of holdout samples to evaluate (default: all)")
    p.add_argument("-o", "--output", default="val_results.jsonl",
                   help="Output JSONL path (default: val_results.jsonl)")
    p.add_argument("-c", "--concurrency", type=int, default=MAX_CONCURRENCY,
                   help=f"Max concurrent API requests (default: {MAX_CONCURRENCY})")
    p.add_argument("-r", "--rollouts", type=int, default=DEFAULT_ROLLOUTS,
                   help=f"Rollouts per prompt (default: {DEFAULT_ROLLOUTS})")
    p.add_argument("-t", "--temperature", type=float, default=DEFAULT_TEMPERATURE,
                   help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})")
    p.add_argument("--zoom-in", action="store_true",
                   help="Enable 2-pass zoom-in strategy")
    p.add_argument("--crop-ratio", type=float, default=CROP_SIZE_RATIO,
                   help=f"Zoom-in crop size ratio (default: {CROP_SIZE_RATIO})")

    args = p.parse_args()
    asyncio.run(run_validation(
        holdout_path=args.input, dataset_mode=args.mode,
        model_mode=args.model, num_samples=args.num_samples,
        output_path=args.output,
        concurrency=args.concurrency, num_rollouts=args.rollouts,
        temperature=args.temperature, zoom_in=args.zoom_in,
        crop_ratio=args.crop_ratio,
    ))


if __name__ == "__main__":
    main()
