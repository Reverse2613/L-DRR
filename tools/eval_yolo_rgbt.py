"""User-operated original-GT evaluation, including prediction-cache reuse."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Export or reuse paired predictions and compute unified geometry metrics.

    Args:
        argv: Optional CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=Path)
    source.add_argument("--predictions", type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--ap50-only", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("batch-size must be positive; no implicit default for an invalid value.")
    import torch
    from torch.utils.data import DataLoader
    from yolo_rgbt.config import load_config
    from yolo_rgbt.data import ATRUMODYOLODataset
    from yolo_rgbt.evaluation import collect_predictions, evaluate_payload, prediction_payload
    from yolo_rgbt.model import load_local_checkpoint

    config = load_config(args.config)
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new evaluation output directory: {args.output_dir}")
    dataset = ATRUMODYOLODataset(config.data_root, args.split, config.input_size, config.min_gt_side_px)
    if args.predictions is not None:
        payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    else:
        device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; explicitly select --device cpu to run CPU inference.")
        model = load_local_checkpoint(args.weights, config)
        loader = DataLoader(
            dataset, batch_size=args.batch_size or config.val_batch, shuffle=False,
            num_workers=config.workers, collate_fn=dataset.collate_fn, pin_memory=device.type == "cuda",
        )
        predictions = collect_predictions(model, loader, config, device)
        payload = prediction_payload(dataset, predictions, config, model.checkpoint_sha256)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "predictions.json").write_text(json.dumps(payload), encoding="utf-8")
    # Save expensive inference first; a failed metric calculation can be
    # resumed from this cache without another pass over the images.
    result = evaluate_payload(dataset, payload, full_metrics=not args.ap50_only)
    (args.output_dir / "geometric_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"Geometric IoU/VOC07 results: {args.output_dir / 'geometric_metrics.json'}")


if __name__ == "__main__":
    main()
