from collections import Counter, defaultdict


def point_in_bbox(px: float, py: float, bbox: list[float], img_w: int, img_h: int) -> bool:
    """Check if a pixel coordinate (px, py) falls within the bounding box.

    bbox may be normalised [0,1] or absolute pixels -- detected automatically.
    """
    x1, y1, x2, y2 = bbox

    if all(0 <= v <= 1 for v in bbox):
        x1, x2 = x1 * img_w, x2 * img_w
        y1, y2 = y1 * img_h, y2 * img_h

    return x1 <= px <= x2 and y1 <= py <= y2


def _group_by_sample(results: list[dict]) -> dict[int, list[dict]]:
    """Group result records by sample idx."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["idx"]].append(r)
    return dict(groups)


def compute_pass_at_k(results: list[dict], k: int | None = None) -> float:
    """Fraction of unique samples where at least one of the first k rollouts hit."""
    groups = _group_by_sample(results)
    if not groups:
        return 0.0
    hits = 0
    for rollouts in groups.values():
        sorted_rollouts = sorted(rollouts, key=lambda r: r["rollout_id"])
        subset = sorted_rollouts[:k] if k is not None else sorted_rollouts
        if any(r["hit"] for r in subset):
            hits += 1
    return hits / len(groups)


def compute_majority_vote(results: list[dict]) -> float:
    """Accuracy when taking the majority-vote hit/miss across rollouts per sample."""
    groups = _group_by_sample(results)
    if not groups:
        return 0.0
    correct = 0
    for rollouts in groups.values():
        votes = Counter(r["hit"] for r in rollouts)
        if votes[True] >= votes[False]:
            correct += 1
    return correct / len(groups)


def compute_breakdown(results: list[dict]) -> dict[str, dict]:
    """Group results by data_source/data_type and compute per-group accuracy."""
    breakdown: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        key = f"{r['data_source']}/{r['data_type']}"
        breakdown[key]["total"] += 1
        if r["hit"]:
            breakdown[key]["correct"] += 1

    return dict(breakdown)


def print_report(results: list[dict], errors: int, output_path: str, num_rollouts: int = 1) -> None:
    correct = sum(1 for r in results if r["hit"])
    total = len(results)
    no_parse = sum(1 for r in results if r["predicted_coords_norm"] is None)
    acc = correct / total if total else 0
    num_samples = len(_group_by_sample(results))

    print(f"\n{'=' * 60}")
    print(f"Samples: {num_samples}  |  Rollouts/sample: {num_rollouts}  |  Total calls: {total}")
    print(f"Per-rollout accuracy: {acc:.1%}  ({correct}/{total})")

    if num_rollouts > 1:
        pass_at_n = compute_pass_at_k(results)
        majority = compute_majority_vote(results)
        print(f"Pass@{num_rollouts} (any rollout hits): {pass_at_n:.1%}")
        print(f"Majority-vote accuracy:  {majority:.1%}")
        if num_rollouts > 2:
            pass_at_1 = compute_pass_at_k(results, k=1)
            print(f"Pass@1 (first rollout):   {pass_at_1:.1%}")

    print(f"Parse failures (no CLICK found): {no_parse}")
    print(f"API errors: {errors}")
    print(f"Results written to: {output_path}")
    print(f"{'=' * 60}")

    breakdown = compute_breakdown(results)
    print("\nBreakdown by source/type (per-rollout):")
    for key in sorted(breakdown):
        b = breakdown[key]
        b_acc = b["correct"] / b["total"] if b["total"] else 0
        print(f"  {key:30s}  {b_acc:.1%}  ({b['correct']}/{b['total']})")
