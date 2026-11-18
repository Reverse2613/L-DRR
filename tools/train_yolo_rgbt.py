"""User-operated paired YOLO11s-OBB training; no automatic weight downloads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse an explicit experiment and start its native training loop.

    Args:
        argv: Optional CLI argument sequence.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--data-root", type=Path, help="Server-specific ATR-UMOD root.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--val-batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--warmup-epochs", type=float)
    args = parser.parse_args(argv)
    from yolo_rgbt.config import load_config
    from yolo_rgbt.trainer import PairedOBBTrainer

    overrides = {
        "output_dir": args.output_dir,
        "data_root": args.data_root,
        "batch": args.batch_size,
        "val_batch": args.val_batch_size,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    config = load_config(args.config, **overrides)
    trainer = PairedOBBTrainer(config, args.device, args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
