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
    args = parser.parse_args(argv)
    from yolo_rgbt.config import load_config
    from yolo_rgbt.trainer import PairedOBBTrainer

    overrides = {"output_dir": args.output_dir} if args.output_dir is not None else {}
    config = load_config(args.config, **overrides)
    trainer = PairedOBBTrainer(config, args.device, args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
