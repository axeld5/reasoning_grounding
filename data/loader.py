import json
import random
from collections.abc import Iterator
from pathlib import Path

from datasets import load_dataset
from PIL import Image

from config import DATASET_CONFIGS, SAVE_EVERY, DatasetConfig
from data.sample import PARSERS, ScreenSpotSample

_FIELD_KEYS = {
    "normal": ("file_name", "data_type", "data_source"),
    "pro": ("img_filename", "ui_type", "application"),
}


def load_screenspot(
    mode: str = "normal",
    num_samples: int | None = None,
    random_sample: bool = False,
    seed: int | None = None,
) -> list[ScreenSpotSample]:
    """Load a ScreenSpot variant and return typed sample objects."""
    cfg: DatasetConfig = DATASET_CONFIGS[mode]
    parse_row = PARSERS[mode]
    rng = random.Random(seed)

    use_streaming = cfg.streaming and not random_sample
    print(f"Loading dataset: {cfg.hf_name} (split={cfg.split}, streaming={use_streaming}) ...")
    ds = load_dataset(cfg.hf_name, split=cfg.split, streaming=use_streaming)

    samples: list[ScreenSpotSample] = []

    if use_streaming:
        for idx, row in enumerate(ds):
            if num_samples is not None and idx >= num_samples:
                break
            samples.append(parse_row(idx, row))
    else:
        n = len(ds)
        if random_sample:
            if num_samples is not None:
                indices = rng.sample(range(n), min(num_samples, n))
            else:
                indices = list(range(n))
                rng.shuffle(indices)
        elif num_samples is not None:
            indices = list(range(min(num_samples, n)))
        else:
            indices = list(range(n))

        for ds_idx in indices:
            samples.append(parse_row(ds_idx, ds[ds_idx]))

    print(f"Loaded {len(samples)} samples" + (" (random)." if random_sample else "."))
    return samples


def _iter_holdout(
    mode: str,
    exclude_indices: set[int],
) -> Iterator[dict]:
    """Yield holdout metadata dicts for every sample not in *exclude_indices*."""
    cfg: DatasetConfig = DATASET_CONFIGS[mode]
    fname_key, type_key, src_key = _FIELD_KEYS[mode]

    ds = load_dataset(cfg.hf_name, split=cfg.split, streaming=cfg.streaming)
    for idx, row in enumerate(ds):
        if idx in exclude_indices:
            continue
        img: Image.Image = row["image"]
        w, h = img.size
        yield {
            "idx": idx,
            "file_name": row.get(fname_key, f"sample_{idx}"),
            "instruction": row["instruction"],
            "bbox": row["bbox"],
            "img_size": [w, h],
            "data_type": row.get(type_key, "unknown"),
            "data_source": row.get(src_key, row.get("data_souce", "unknown")),
        }


def save_holdout(
    mode: str,
    exclude_indices: set[int],
    output_path: Path,
    save_every: int = SAVE_EVERY,
) -> int:
    """Stream holdout records to *output_path*, flushing every *save_every* rows."""
    print(f"Generating holdout (excluding {len(exclude_indices)} eval indices) ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    last_saved = 0

    for record in _iter_holdout(mode, exclude_indices):
        records.append(record)
        if len(records) - last_saved >= save_every:
            with open(output_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            last_saved = len(records)

    # Final flush
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Holdout: {len(records)} samples -> {output_path.resolve()}")
    return len(records)


def load_from_holdout(
    holdout_path: str | Path,
    mode: str = "normal",
    num_samples: int | None = None,
) -> list[ScreenSpotSample]:
    """Reload ScreenSpotSamples from a holdout JSONL by fetching images from HF."""
    holdout_path = Path(holdout_path)
    with open(holdout_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if num_samples is not None:
        records = records[:num_samples]

    indices = sorted(r["idx"] for r in records)
    print(f"Loading {len(indices)} images from HF for holdout validation ...")

    cfg: DatasetConfig = DATASET_CONFIGS[mode]
    parse_row = PARSERS[mode]

    # Always non-streaming so we can random-access by index
    ds = load_dataset(cfg.hf_name, split=cfg.split, streaming=False)

    samples: list[ScreenSpotSample] = []
    for ds_idx in indices:
        samples.append(parse_row(ds_idx, ds[ds_idx]))

    print(f"Loaded {len(samples)} validation samples.")
    return samples
