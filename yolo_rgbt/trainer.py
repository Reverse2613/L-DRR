"""Native YOLO training loop with paired data and controlled initialization."""

from __future__ import annotations

import hashlib
import json
from copy import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .config import ExperimentConfig
from .data import ATRUMODYOLODataset, CLASSES, move_batch
from .model import PairedYOLO11sOBB, load_local_checkpoint
from .optim import build_optimizer
from .runtime import require_framework
from .validator import PairedOBBValidator

require_framework()
from ultralytics.data import build_dataloader
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.utils import LOGGER, callbacks


class PairedOBBTrainer(OBBTrainer):
    """Single-device experiment; full training is deliberately user-operated."""

    def __init__(
        self, config: ExperimentConfig, device: str = "0", resume: Optional[str | Path] = None,
    ) -> None:
        """Validate local files before native setup creates a run directory.

        Args:
            config: Paired experiment configuration.
            device: One CUDA index or cpu.
            resume: Explicit local last.pt from an interrupted identical run.
        """
        if "," in str(device) or str(device) == "-1":
            raise ValueError("This first prototype supports one explicit device only.")
        self.experiment = config
        required = Path(resume).resolve() if resume is not None else config.pretrained
        if not required.is_file():
            raise FileNotFoundError(f"Download/provide the local detector checkpoint yourself: {required}")
        if resume is None and config.output_dir.exists():
            raise FileExistsError(f"Existing output_dir: {config.output_dir}; choose an independent new run.")
        if resume is not None:
            checkpoint_model = load_local_checkpoint(required, config)
            current = config.as_dict()
            saved = checkpoint_model.experiment_spec
            excluded = {"source", "pretrained"}
            # Checkpoints made before this augmentation release lack these
            # fields. They are compatible only when the current run exactly
            # matches the disabled/default values; a real enhancement change
            # still rejects resume.
            legacy_config_fields = (
                "warmup_epochs",
                "augment", "aug_hflip", "aug_vflip", "aug_rotate_prob", "aug_rotate_deg",
                "aug_scale_prob", "aug_scale_min", "aug_scale_max", "aug_translate_prob",
                "aug_translate_frac", "aug_fill_value",
            )
            legacy_missing = [key for key in legacy_config_fields if key not in saved]
            incompatible_legacy = [
                key for key in legacy_missing
                if current[key] != ExperimentConfig.__dataclass_fields__[key].default
            ]
            if incompatible_legacy:
                raise ValueError(f"Resume uses non-default values for legacy fields: {incompatible_legacy}.")
            if legacy_missing:
                LOGGER.warning(f"Legacy checkpoint lacks backward-compatible fields: {legacy_missing}.")
            changed = [key for key, value in current.items()
                       if key not in excluded and key not in legacy_missing and saved.get(key) != value]
            if changed:
                raise ValueError(f"Resume changes the experiment: {changed}; start an independent run instead.")
        args = config.native_args(str(device))
        if resume is not None:
            args["resume"] = str(required)
        super().__init__(overrides=args)
        # Do not add HUB or other external experiment-service callbacks.
        self.callbacks = callbacks.get_default_callbacks()
        (self.save_dir / "experiment.json").write_text(
            json.dumps(config.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def get_dataset(self) -> Tuple[str, str]:
        """Resolve unmodified ATR train/test folders, without native conversion.

        Returns:
            Training and prototype-validation IR image directory paths.
        """
        root = self.experiment.data_root
        self.data = {
            "path": str(root), "nc": len(CLASSES), "names": dict(enumerate(CLASSES)),
            "channels": 6, "train": str(root / "train" / "images_ir"),
            "val": str(root / self.experiment.validation_split / "images_ir"),
            "test": str(root / "test" / "images_ir"),
        }
        return self.data["train"], self.data["val"]

    def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> PairedYOLO11sOBB:
        """Require pretrained initialization and audit every transferred tensor.

        Args:
            cfg: Native loader argument, retained for API compatibility.
            weights: Already loaded local native or paired model.
            verbose: Log transfer coverage.

        Returns:
            Paired model. Training never silently starts from random weights.
        """
        if weights is None:
            raise ValueError("Pretrained weights are required; random initialization is only allowed in check tools.")
        model = PairedYOLO11sOBB(self.experiment)
        model.load(weights, verbose=verbose)
        report = dict(model.pretrained_report)
        if self.experiment.pretrained.is_file():
            report["checkpoint_sha256"] = hashlib.sha256(self.experiment.pretrained.read_bytes()).hexdigest()
        (self.save_dir / "pretrained_loading.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        model.info()
        return model

    def get_dataloader(
        self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train",
    ) -> Any:
        """Build a paired loader, preserving original boxes and image IDs.

        Args:
            dataset_path: Native path, checked against the explicit split.
            batch_size: Native training batch request.
            rank: Native single-device rank.
            mode: train or val.

        Returns:
            Native InfiniteDataLoader using the paired collate function.
        """
        if mode not in {"train", "val"}:
            raise ValueError("Loader mode must be train or val.")
        split = "train" if mode == "train" else self.experiment.validation_split
        expected = self.experiment.data_root / split / "images_ir"
        if Path(dataset_path).resolve() != expected.resolve():
            raise ValueError("Native dataset path does not match the declared split.")
        dataset = ATRUMODYOLODataset(
            self.experiment.data_root, split, self.experiment.input_size,
            self.experiment.min_gt_side_px, self.experiment.augmentation(), self.experiment.seed,
        )
        previous = self.model.data_audits.get(mode)
        if previous is not None:
            for key in ("annotations_sha256", "sample_ids_sha256", "split", "input_size"):
                if previous[key] != dataset.audit[key]:
                    raise ValueError(f"{mode} data changed since the checkpoint: {key}.")
        self.model.data_audits[mode] = dataset.audit
        audit_file = self.save_dir / f"data_audit_{mode}.json"
        audit_file.write_text(json.dumps(dataset.audit, indent=2, ensure_ascii=False), encoding="utf-8")
        count = dataset.audit["counts"]
        LOGGER.info(f"{split}: {len(dataset)} pairs; {count.get('parser_kept', 0)} GT; "
                    f"{count.get('center_outside_retained', 0)} outside-center GT retained; "
                    f"{count.get('native_loss_small_objects', 0)} explicitly below the "
                    f"{self.experiment.min_gt_side_px:g}px native loss threshold.")
        effective_batch = batch_size if mode == "train" else self.experiment.val_batch
        return build_dataloader(dataset, effective_batch, self.experiment.workers, mode == "train", rank)

    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move/scalе the paired image and metadata batch exactly once.

        Args:
            batch: Raw uint8 paired batch.

        Returns:
            Device-local native loss/prediction batch.
        """
        return move_batch(batch, self.device)

    def get_validator(self) -> PairedOBBValidator:
        """Create the explicit-condition native OBB validator.

        Returns:
            Validator with native AP50 checkpoint selection.
        """
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        return PairedOBBValidator(
            self.experiment, self.test_loader, self.save_dir, copy(self.args), self.callbacks
        )

    def _warmup_batches(self) -> int:
        """Resolve batch-independent warmup epochs or historical step count.

        Returns:
            Effective warmup optimizer iterations. Ultralytics also enforces a
            minimum of 100 iterations in its training loop.
        """
        if self.experiment.warmup_epochs is None:
            return self.experiment.warmup_iterations
        return max(round(self.experiment.warmup_epochs * len(self.train_loader)), 100)

    def build_optimizer(
        self, model: nn.Module, name: str = "SGD", lr: float = 0.01,
        momentum: float = 0.937, decay: float = 0.0005, iterations: int = 100000,
    ) -> torch.optim.Optimizer:
        """Apply the experiment's SGD or AdamW recipe with audited groups.

        Args:
            model: Paired model.
            name: Native optimizer name; must match the experiment exactly.
            lr: Native ``lr0`` value; must equal the new-parameter learning rate.
            momentum: Native momentum; checked for SGD and ignored by AdamW.
            decay: Effective matrix decay after native batch scaling.
            iterations: Reserved native API argument.

        Returns:
            Fully audited optimizer parameter groups.

        Raises:
            ValueError: A native argument differs from the explicit experiment.
                Silent optimizer fallback is intentionally forbidden.
        """
        checks = [("optimizer", name, self.experiment.optimizer),
                  ("lr0", lr, self.experiment.lr_new),
                  ("weight_decay", decay, self.experiment.weight_decay)]
        if self.experiment.optimizer == "SGD":
            checks.append(("momentum", momentum, self.experiment.momentum))
        changed = [(key, observed, expected) for key, observed, expected in checks
                   if (observed != expected if isinstance(observed, str)
                       else abs(observed - expected) > 1e-12)]
        if changed:
            raise ValueError(f"Native optimizer arguments differ from the experiment: {changed}.")
        optimizer = build_optimizer(model, self.experiment)
        summary = {
            "optimizer": type(optimizer).__name__,
            "hyperparameters": {
                "lr_pretrained": self.experiment.lr_pretrained,
                "lr_new": self.experiment.lr_new,
                "lr_final_ratio": self.experiment.lr_final_ratio,
                "momentum": self.experiment.momentum if self.experiment.optimizer == "SGD" else None,
                "warmup_momentum": (self.experiment.warmup_momentum
                                    if self.experiment.optimizer == "SGD" else None),
                "nesterov": self.experiment.nesterov if self.experiment.optimizer == "SGD" else None,
                "weight_decay": self.experiment.weight_decay,
                "warmup_iterations_configured": self.experiment.warmup_iterations,
                "warmup_epochs_configured": self.experiment.warmup_epochs,
                "warmup_iterations_effective": self._warmup_batches(),
            },
            "groups": [
                {key: value for key, value in group.items() if key != "params"}
                for group in optimizer.param_groups
            ],
        }
        (self.save_dir / "optimizer_groups.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return optimizer

    def _setup_train(self, world_size: int) -> None:
        """Skip native AMP's auxiliary detector download, preserving local AMP.

        Args:
            world_size: Must be one. A local GPU check validates loss/backward
                instead of downloading yolov8n.pt for the native AMP heuristic.
        """
        if world_size != 1:
            raise ValueError("Use a single GPU for the initial paired prototype.")
        requested_amp = self.args.amp
        self.args.amp = False
        try:
            super()._setup_train(world_size)
        finally:
            self.args.amp = requested_amp
        self.amp = bool(requested_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        warmup_batches = self._warmup_batches()
        self.args.warmup_epochs = warmup_batches / len(self.train_loader)
        LOGGER.info(
            f"Actual paired input={self.experiment.input_size}; effective batch={self.experiment.batch}; "
            f"optimizer={self.experiment.optimizer}; warmup={warmup_batches} iterations "
            f"({self.args.warmup_epochs:.3f} epochs); AMP={self.amp}. "
            "No auxiliary AMP-check weights downloaded."
        )

    def optimizer_step(self) -> None:
        """Clip unscaled gradients and log an AMP overflow rather than hiding it.

        Returns:
            None. EMA advances only after a real optimizer update.
        """
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.experiment.clip_norm)
        finite = bool(torch.isfinite(norm))
        if not finite and not self.amp:
            raise FloatingPointError("Nonfinite FP32 gradients; optimizer update aborted.")
        scale_before = self.scaler.get_scale()
        if not finite:
            self.scaler.update(new_scale=scale_before / 2.0)
            self.optimizer.zero_grad(set_to_none=True)
            LOGGER.warning(f"AMP nonfinite norm: optimizer/EMA skipped; scale={scale_before}->{scale_before / 2.0}.")
            return
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = self.scaler.get_scale()
        self.optimizer.zero_grad(set_to_none=True)
        skipped = not finite or scale_after < scale_before
        if skipped:
            LOGGER.warning(f"AMP overflow: update/EMA skipped; grad_norm={norm}, scale={scale_before}->{scale_after}.")
        elif self.ema is not None:
            self.ema.update(self.model)
