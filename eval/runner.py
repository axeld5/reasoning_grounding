import asyncio
import json
from pathlib import Path

from config import CROP_SIZE_RATIO, DEFAULT_ROLLOUTS, DEFAULT_TEMPERATURE, MAX_CONCURRENCY, MODEL_CONFIGS
from data import load_screenspot
from data.loader import load_from_holdout, save_holdout
from eval.client import create_client
from eval.passes import run_pass, run_zoomin_pass
from metrics import print_report


async def run_evaluation(
    dataset_mode: str = "normal",
    model_mode: str = "claude",
    num_samples: int | None = None,
    output_path: str = "results.jsonl",
    holdout_path: str | None = None,
    concurrency: int = MAX_CONCURRENCY,
    num_rollouts: int = DEFAULT_ROLLOUTS,
    temperature: float = DEFAULT_TEMPERATURE,
    random_sample: bool = False,
    sample_seed: int | None = None,
    zoom_in: bool = False,
    crop_ratio: float = CROP_SIZE_RATIO,
) -> None:
    model_cfg = MODEL_CONFIGS[model_mode]
    client = create_client(model_cfg)
    semaphore = asyncio.Semaphore(concurrency)

    samples = load_screenspot(
        dataset_mode, num_samples, random_sample=random_sample, seed=sample_seed,
    )

    n_passes = 2 if zoom_in else 1
    total_calls = len(samples) * num_rollouts * n_passes
    print(
        f"Evaluating {len(samples)} samples \u00d7 {num_rollouts} rollouts "
        f"\u00d7 {n_passes} pass{'es' if n_passes > 1 else ''} "
        f"= {total_calls} calls with {model_cfg.model_id} "
        f"(concurrency={concurrency}, temperature={temperature}) ..."
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    pass1 = await run_pass(
        client, semaphore, samples, model_cfg, num_rollouts, temperature,
        desc="Pass 1 (full image)", checkpoint_path=out_file,
    )

    results = pass1
    if zoom_in:
        results = await run_zoomin_pass(
            client, semaphore, samples, pass1, model_cfg, num_rollouts, temperature, crop_ratio,
            checkpoint_path=out_file,
        )

    # Final clean write (overwrites checkpoint with complete, ordered results)
    errors = len(samples) * num_rollouts - len(results)
    with open(out_file, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    print_report(results, errors, str(out_file.resolve()), num_rollouts)

    # Holdout / validation split (progressive save)
    if holdout_path:
        eval_indices = {s.idx for s in samples}
        save_holdout(dataset_mode, eval_indices, Path(holdout_path))


async def run_validation(
    holdout_path: str,
    dataset_mode: str = "normal",
    model_mode: str = "claude",
    num_samples: int | None = None,
    output_path: str = "val_results.jsonl",
    concurrency: int = MAX_CONCURRENCY,
    num_rollouts: int = DEFAULT_ROLLOUTS,
    temperature: float = DEFAULT_TEMPERATURE,
    zoom_in: bool = False,
    crop_ratio: float = CROP_SIZE_RATIO,
) -> None:
    """Run evaluation on a holdout JSONL (images reloaded from HF by idx)."""
    model_cfg = MODEL_CONFIGS[model_mode]
    client = create_client(model_cfg)
    semaphore = asyncio.Semaphore(concurrency)

    samples = load_from_holdout(holdout_path, mode=dataset_mode, num_samples=num_samples)

    n_passes = 2 if zoom_in else 1
    total_calls = len(samples) * num_rollouts * n_passes
    print(
        f"Validating {len(samples)} holdout samples \u00d7 {num_rollouts} rollouts "
        f"\u00d7 {n_passes} pass{'es' if n_passes > 1 else ''} "
        f"= {total_calls} calls with {model_cfg.model_id} "
        f"(concurrency={concurrency}, temperature={temperature}) ..."
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    pass1 = await run_pass(
        client, semaphore, samples, model_cfg, num_rollouts, temperature,
        desc="Val pass 1 (full image)", checkpoint_path=out_file,
    )

    results = pass1
    if zoom_in:
        results = await run_zoomin_pass(
            client, semaphore, samples, pass1, model_cfg, num_rollouts, temperature, crop_ratio,
            checkpoint_path=out_file,
        )

    errors = len(samples) * num_rollouts - len(results)
    with open(out_file, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    print_report(results, errors, str(out_file.resolve()), num_rollouts)
