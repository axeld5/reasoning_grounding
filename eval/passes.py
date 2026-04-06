import asyncio
import json
from pathlib import Path
from typing import Any

from tqdm.asyncio import tqdm as atqdm

from config import SAVE_EVERY
from data import ScreenSpotSample
from eval.client import call_with_retry
from inference import crop_around_point, map_zoomin_prediction, predict_click
from metrics import point_in_bbox


def _checkpoint(results: list[dict], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


async def run_pass(
    client: Any,
    semaphore: asyncio.Semaphore,
    samples: list[ScreenSpotSample],
    model_cfg,
    num_rollouts: int,
    temperature: float,
    desc: str,
    checkpoint_path: Path | None = None,
    save_every: int = SAVE_EVERY,
) -> list[dict]:
    """Run a single inference pass over all samples x rollouts."""
    tasks = {(s.idx, r): (s, r) for s in samples for r in range(num_rollouts)}

    async def _do(sample: ScreenSpotSample, rollout_id: int):
        raw, coords = await call_with_retry(
            predict_click, client, semaphore, sample.image, sample.instruction,
            model_cfg=model_cfg, temperature=temperature,
        )
        return sample, rollout_id, raw, coords

    coros = [_do(s, r) for s, r in tasks.values()]
    results: list[dict] = []
    correct = total = last_saved = 0

    async for coro in atqdm(asyncio.as_completed(coros), total=len(coros), desc=desc):
        try:
            sample, rollout_id, raw, coords = await coro
        except Exception as e:
            print(f"API error: {e}")
            continue

        img_w, img_h = sample.img_size
        pred_px = None
        hit = False
        if coords is not None:
            nx, ny = coords
            pred_px = [nx * img_w, ny * img_h]
            hit = point_in_bbox(pred_px[0], pred_px[1], sample.bbox, img_w, img_h)

        total += 1
        if hit:
            correct += 1

        results.append({
            "idx": sample.idx, "rollout_id": rollout_id,
            "file_name": sample.file_name, "instruction": sample.instruction,
            "bbox": sample.bbox, "img_size": [img_w, img_h],
            "data_type": sample.data_type, "data_source": sample.data_source,
            "predicted_coords_norm": list(coords) if coords else None,
            "predicted_coords_px": pred_px, "hit": hit, "raw_response": raw,
        })

        if total % 10 == 0:
            acc = correct / total if total else 0
            atqdm.write(f"  [{total}] running accuracy: {acc:.1%} ({correct}/{total})")

        if total - last_saved >= save_every:
            _checkpoint(results, checkpoint_path)
            last_saved = total

    if total > last_saved:
        _checkpoint(results, checkpoint_path)

    return results


async def run_zoomin_pass(
    client: Any,
    semaphore: asyncio.Semaphore,
    samples: list[ScreenSpotSample],
    pass1_results: list[dict],
    model_cfg,
    num_rollouts: int,
    temperature: float,
    crop_ratio: float,
    checkpoint_path: Path | None = None,
    save_every: int = SAVE_EVERY,
) -> list[dict]:
    """Run the zoom-in second pass: crop around pass-1 predictions and re-predict."""
    sample_by_idx = {s.idx: s for s in samples}

    async def _do_zoom(p1: dict):
        sample = sample_by_idx[p1["idx"]]
        coords_norm = p1["predicted_coords_norm"]
        if coords_norm is None:
            return p1

        nx, ny = coords_norm
        crop_img, crop_bbox = crop_around_point(
            sample.image, nx, ny, crop_ratio=crop_ratio, resize_to_original=True,
        )
        raw, coords2 = await call_with_retry(
            predict_click, client, semaphore, crop_img, sample.instruction,
            model_cfg=model_cfg, temperature=temperature,
        )

        img_w, img_h = sample.img_size
        pred_px = None
        hit = False
        mapped_norm = None

        if coords2 is not None:
            mx, my = map_zoomin_prediction(coords2[0], coords2[1], crop_bbox, img_w, img_h)
            mapped_norm = [mx, my]
            pred_px = [mx * img_w, my * img_h]
            hit = point_in_bbox(pred_px[0], pred_px[1], sample.bbox, img_w, img_h)

        return {
            **p1,
            "pass1_coords_norm": p1["predicted_coords_norm"], "pass1_hit": p1["hit"],
            "predicted_coords_norm": mapped_norm, "predicted_coords_px": pred_px,
            "hit": hit, "zoom_crop_bbox": list(crop_bbox), "raw_response_zoom": raw,
        }

    coros = [_do_zoom(p1) for p1 in pass1_results]
    results: list[dict] = []
    correct = total = last_saved = 0

    async for coro in atqdm(asyncio.as_completed(coros), total=len(coros), desc="Pass 2 (zoom-in)"):
        try:
            record = await coro
        except Exception as e:
            print(f"API error (zoom): {e}")
            continue

        total += 1
        if record["hit"]:
            correct += 1
        results.append(record)

        if total % 10 == 0:
            acc = correct / total if total else 0
            atqdm.write(f"  [zoom {total}] running accuracy: {acc:.1%} ({correct}/{total})")

        if total - last_saved >= save_every:
            _checkpoint(results, checkpoint_path)
            last_saved = total

    if total > last_saved:
        _checkpoint(results, checkpoint_path)

    return results
