"""Measure whether a trained YOLO CG-DMF model departs from equal fusion.

The tool runs deterministic inference on evenly spaced split samples and reads
the two learned modality scores through forward hooks. It writes per-image
weights and condition-group summaries; it does not train or download weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _condition_value(value: torch.Tensor) -> int | float:
    """Convert a one-element condition tensor to a JSON scalar."""
    if value.numel() != 1:
        raise ValueError("Expected one condition scalar per sample.")
    return int(value.item()) if value.dtype == torch.int64 else float(value.item())


def _entropy01(weight: torch.Tensor) -> torch.Tensor:
    """Return normalized binary entropy, where 1 means exactly equal fusion."""
    weight = weight.float().clamp(1e-6, 1 - 1e-6)
    entropy = -(weight * weight.log2() + (1 - weight) * (1 - weight).log2())
    return entropy


def _summarise(values: Sequence[float]) -> Dict[str, Any]:
    """Return compact descriptive statistics for finite scalar values."""
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Weight summary requires nonempty finite values.")
    return {
        "count": int(len(array)), "mean": float(array.mean()), "std": float(array.std()),
        "median": float(np.median(array)), "min": float(array.min()), "max": float(array.max()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Run fusion-weight diagnostics and write a new output directory.

    Args:
        argv: Optional command-line argument sequence.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=256,
                        help="Evenly spaced diagnostic subset; 0 means the full split.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.max_images < 0:
        raise ValueError("batch-size must be positive and max-images nonnegative.")
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new diagnostic output directory: {args.output_dir}")

    from yolo_rgbt.augment import PairedAugmentConfig
    from yolo_rgbt.config import load_config
    from yolo_rgbt.data import ATRUMODYOLODataset, move_batch
    from yolo_rgbt.model import load_local_checkpoint

    config = load_config(args.config)
    if config.variant != "condition" or not config.dynamic_fusion:
        raise ValueError("Use the dynamic condition configuration; equal fusion has no learned scores to inspect.")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu explicitly.")
    model = load_local_checkpoint(args.weights, config).to(device).float().eval()
    dataset = ATRUMODYOLODataset(
        config.data_root, args.split, config.input_size, config.min_gt_side_px,
        PairedAugmentConfig(enabled=False), config.seed,
    )
    if args.max_images:
        count = min(args.max_images, len(dataset))
        indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64).tolist()
        selected = Subset(dataset, indices)
    else:
        selected = dataset
    loader = DataLoader(
        selected, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=dataset.collate_fn, pin_memory=device.type == "cuda",
    )

    captures: Dict[tuple[int, str], torch.Tensor] = {}
    handles = []
    for scale, block in zip(model.fusion_indices, model.fusions):
        for modality, module in (("rgb", block.rgb_score), ("ir", block.ir_score)):
            def capture(_module: Any, _inputs: Any, output: torch.Tensor,
                        key: tuple[int, str] = (int(scale), modality)) -> None:
                captures[key] = output.detach()
            handles.append(module.register_forward_hook(capture))

    rows: List[Dict[str, Any]] = []
    weight_fields = [f"rgb_weight_s{scale}" for scale in model.fusion_indices]
    deviation_fields = [f"mean_abs_dev_s{scale}" for scale in model.fusion_indices]
    entropy_fields = [f"entropy_s{scale}" for scale in model.fusion_indices]
    try:
        with torch.inference_mode():
            for raw in loader:
                captures.clear()
                batch = move_batch(raw, device)
                model.predict(batch["img"], conditions=batch["conditions"])
                per_scale = {}
                for scale in model.fusion_indices:
                    rgb_score = captures.get((int(scale), "rgb"))
                    ir_score = captures.get((int(scale), "ir"))
                    if rgb_score is None or ir_score is None:
                        raise RuntimeError(f"Fusion hooks did not capture scale {scale}.")
                    per_scale[int(scale)] = (rgb_score - ir_score).sigmoid().float()
                for batch_index, image_id in enumerate(raw["image_id"]):
                    row: Dict[str, Any] = {
                        "image_id": image_id,
                        "weather": _condition_value(raw["conditions"]["weather"][batch_index]),
                        "illumination": _condition_value(raw["conditions"]["illumination"][batch_index]),
                        "location": _condition_value(raw["conditions"]["location"][batch_index]),
                    }
                    for scale, weight in per_scale.items():
                        sample = weight[batch_index:batch_index + 1]
                        row[f"rgb_weight_s{scale}"] = float(sample.mean().cpu())
                        row[f"mean_abs_dev_s{scale}"] = float((sample - 0.5).abs().mean().cpu())
                        row[f"entropy_s{scale}"] = float(_entropy01(sample).mean().cpu())
                    rows.append(row)
    finally:
        for handle in handles:
            handle.remove()

    if not rows:
        raise RuntimeError("No samples were processed.")
    aggregate = {
        name: {
            "rgb_weight": _summarise([float(row[f"rgb_weight_s{scale}"]) for row in rows]),
            "mean_abs_deviation_from_equal": _summarise(
                [float(row[f"mean_abs_dev_s{scale}"]) for row in rows]),
            "normalized_entropy": _summarise([float(row[f"entropy_s{scale}"]) for row in rows]),
        }
        for scale, name in zip(model.fusion_indices, ("P3_stride8", "P4_stride16", "P5_stride32"))
    }
    by_condition: Dict[str, Dict[str, Any]] = {}
    for field in ("weather", "illumination", "location"):
        groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[int(row[field])].append(row)
        by_condition[field] = {}
        for value, members in sorted(groups.items()):
            by_condition[field][str(value)] = {
                "count": len(members),
                **{f"rgb_weight_s{scale}": _summarise(
                    [float(row[f"rgb_weight_s{scale}"]) for row in members])
                   for scale in model.fusion_indices},
            }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fields = ["image_id", "weather", "illumination", "location", *weight_fields,
              *deviation_fields, *entropy_fields]
    with (args.output_dir / "fusion_weights.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "scope": "deterministic fusion-weight diagnostics; no training",
        "checkpoint": str(args.weights.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "config": str(args.config.resolve()),
        "split": args.split, "images": len(rows), "scales": list(model.fusion_indices),
        "aggregate": aggregate, "by_condition": by_condition,
        "interpretation": {
            "equal_fusion_reference": 0.5,
            "normalized_entropy_reference": 1.0,
            "collapse_warning": "A mean absolute deviation near zero and entropy near one indicate bypass-like weights.",
        },
    }
    (args.output_dir / "fusion_weight_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    compact = {scale: {"rgb_weight_mean": value["rgb_weight"]["mean"],
                       "mean_abs_deviation": value["mean_abs_deviation_from_equal"]["mean"],
                       "normalized_entropy": value["normalized_entropy"]["mean"]}
               for scale, value in aggregate.items()}
    print(json.dumps({"images": len(rows), "aggregate": compact,
                      "report": str(args.output_dir / "fusion_weight_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
