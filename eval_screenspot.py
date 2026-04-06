"""ScreenSpot GUI-grounding evaluation CLI."""

import argparse
import asyncio

from dotenv import load_dotenv

load_dotenv()

from config import CROP_SIZE_RATIO, DATASET_CONFIGS, DEFAULT_MODEL_MODE, DEFAULT_ROLLOUTS, DEFAULT_TEMPERATURE, MAX_CONCURRENCY, MODEL_CONFIGS
from eval import run_evaluation


def main() -> None:
    p = argparse.ArgumentParser(description="ScreenSpot evaluation")
    p.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default=DEFAULT_MODEL_MODE,
                   help=f"Model to use (default: {DEFAULT_MODEL_MODE})")
    p.add_argument("-m", "--mode", choices=list(DATASET_CONFIGS.keys()), default="normal",
                   help="Dataset mode: 'normal' or 'pro'")
    p.add_argument("-n", "--num-samples", type=int, default=None,
                   help="Number of samples to evaluate (default: all)")
    p.add_argument("--random-sample", action="store_true",
                   help="Draw samples uniformly at random")
    p.add_argument("--sample-seed", type=int, default=None,
                   help="RNG seed for --random-sample")
    p.add_argument("-o", "--output", default="results.jsonl",
                   help="Output JSONL path")
    p.add_argument("--holdout-output", default=None,
                   help="Write non-evaluated samples to this JSONL for validation")
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
    asyncio.run(run_evaluation(
        dataset_mode=args.mode, model_mode=args.model,
        num_samples=args.num_samples, output_path=args.output,
        holdout_path=args.holdout_output, concurrency=args.concurrency,
        num_rollouts=args.rollouts, temperature=args.temperature,
        random_sample=args.random_sample, sample_seed=args.sample_seed,
        zoom_in=args.zoom_in, crop_ratio=args.crop_ratio,
    ))


if __name__ == "__main__":
    main()
