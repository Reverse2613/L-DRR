# Uses Ultralytics 8.3.0 OBBValidator interfaces, AGPL-3.0.
"""Explicit-condition validation without AutoBackend, clipping, or downloads."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .config import ExperimentConfig
from .data import CLASSES, move_batch
from .model import PairedYOLO11sOBB, load_local_checkpoint
from .runtime import require_framework

require_framework()
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.utils import LOGGER, TQDM, ops
from ultralytics.utils.metrics import OBBMetrics
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import autocast, de_parallel, smart_inference_mode


def expected_image_ids(dataset: Any) -> List[str]:
    """Resolve full-split or explicitly requested diagnostic subset IDs.

    Args:
        dataset: Paired dataset, or a PyTorch Subset of it.

    Returns:
        Expected image ID sequence, used to detect duplicates/missing images.
    """
    if isinstance(dataset, Subset):
        parent = expected_image_ids(dataset.dataset)
        return [parent[index] for index in dataset.indices]
    if not hasattr(dataset, "image_ids"):
        raise ValueError("Validator requires a dataset with an explicit sample ID manifest.")
    return list(dataset.image_ids)


def postprocess_predictions(predictions: Any, config: ExperimentConfig) -> List[torch.Tensor]:
    """Run the native rotated NMS in FP32.

    Args:
        predictions: Native inference tensor or (tensor, auxiliary) tuple.
        config: Confidence, NMS threshold, class count and output cap.

    Returns:
        Per-image float32 [N,7]: cx,cy,w,h,score,class,angle. No box coordinates
        are clipped to the frame. FP32 avoids half-precision covariance overflow.
    """
    tensor = predictions[0] if isinstance(predictions, tuple) else predictions
    if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
        raise FloatingPointError("Nonfinite or malformed paired OBB predictions.")
    return ops.non_max_suppression(
        tensor.float(), config.conf, config.nms_iou, nc=len(CLASSES),
        multi_label=True, agnostic=False, max_det=config.max_det, rotated=True,
    )


class PairedOBBValidator(OBBValidator):
    """Keep native ProbIoU/101-point AP while supplying batch metadata."""

    def __init__(
        self, config: ExperimentConfig, dataloader: Optional[DataLoader], save_dir: Path,
        args: Any, callbacks: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        """Initialize the native OBB metric interface with an explicit dataset.

        Args:
            config: Experiment input/metadata contract.
            dataloader: Native-resolution paired batches.
            save_dir: Output folder.
            args: Native arguments.
            callbacks: Optional local callbacks.
        """
        self.experiment = config
        super().__init__(dataloader, save_dir=save_dir, args=copy(args), _callbacks=callbacks)
        self.args.plots = False
        self.args.save_hybrid = False
        self.args.save_json = False
        self.args.save_txt = False
        self.args.augment = False

    def init_metrics(self, model: nn.Module) -> None:
        """Reset metrics every pass, including an epoch with no true positives.

        Args:
            model: Paired detector with the fixed class mapping.
        """
        # The native get_stats() skips process() if there are no true positives.
        # A fresh object prevents accidentally reusing the previous epoch's AP.
        self.metrics = OBBMetrics(save_dir=self.save_dir, plot=False, on_plot=self.on_plot)
        super().init_metrics(model)
        self.lb = []

    def preprocess(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        """Move the five condition fields together with the paired tensor.

        Args:
            batch: Raw collated uint8 [B,6,H,W] samples.

        Returns:
            FP32 normalized images and device-local labels/conditions.
        """
        return move_batch(batch, self.device)

    def _prepare_batch(self, si: int, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Recover original-pixel GT without clipping genuine border objects.

        Args:
            si: Position within the batch.
            batch: Device-local normalized YOLO targets.

        Returns:
            Native validator fields with pixel/radian [N,5] GT boxes.
        """
        mask = batch["batch_idx"] == si
        cls = batch["cls"][mask].squeeze(-1)
        boxes = batch["bboxes"][mask].float().clone()
        height, width = batch["img"].shape[-2:]
        original = tuple(batch["ori_shape"][si])
        if original != (height, width):
            raise ValueError("This adapter does not silently resize or letterbox the paired frames.")
        boxes[:, :4] *= boxes.new_tensor((width, height, width, height))
        return {"cls": cls, "bbox": boxes, "ori_shape": original,
                "imgsz": (height, width), "ratio_pad": ((1.0, 1.0), (0.0, 0.0))}

    def _prepare_pred(self, pred: torch.Tensor, pbatch: Dict[str, Any]) -> torch.Tensor:
        """Retain predicted original-pixel centers, including outside centers.

        Args:
            pred: Float32 [N,7] native detections.
            pbatch: Identity geometry metadata.

        Returns:
            Original-pixel detections, without the native scale_boxes clipping.
        """
        if tuple(pbatch["ori_shape"]) != tuple(pbatch["imgsz"]):
            raise ValueError("Nonidentity preprocessing requires an explicit coordinate inverse.")
        return pred.float().clone()

    def postprocess(self, preds: Any) -> List[torch.Tensor]:
        """Apply the same FP32 rotated NMS as the prediction exporter.

        Args:
            preds: Native inference output.

        Returns:
            Per-image [N,7] detections.
        """
        return postprocess_predictions(preds, self.experiment)

    def get_stats(self) -> Dict[str, float]:
        """Calculate native metrics and explicitly select by native AP50.

        Returns:
            Native precision/recall/AP50/AP50-95 plus fitness=AP50.
        """
        if not self.stats["tp"]:
            self.nt_per_class = np.zeros(self.nc, dtype=np.int64)
            self.nt_per_image = np.zeros(self.nc, dtype=np.int64)
            stats = self.metrics.results_dict
        else:
            stats = super().get_stats()
        stats["fitness"] = float(stats["metrics/mAP50(B)"])
        return stats

    @smart_inference_mode()
    def __call__(
        self, trainer: Any = None, model: Optional[nn.Module | str | Path] = None,
    ) -> Dict[str, float]:
        """Validate directly with images AND metadata; no auxiliary model.

        Args:
            trainer: Native paired trainer for epoch validation.
            model: Paired model or local paired checkpoint for standalone use.

        Returns:
            Native OBB metrics, optionally mean validation loss components.
            This is ProbIoU/101-point AP, not the old polygon-IoU/VOC07 metric.
        """
        self.training = trainer is not None
        if self.dataloader is None:
            raise ValueError("A paired dataloader is required for actual validation.")
        if self.training:
            self.device, self.data = trainer.device, trainer.data
            model = trainer.ema.ema if trainer.ema is not None else trainer.model
            enabled_amp = bool(trainer.amp)
            self.loss = torch.zeros(3, device=self.device)
        else:
            if isinstance(model, (str, Path)):
                model = load_local_checkpoint(model, self.experiment)
                # Standalone final_eval uses the trainer's already selected device.
                device = self.device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
            if not isinstance(model, PairedYOLO11sOBB):
                raise TypeError("Use a local PairedYOLO11sOBB checkpoint.")
            self.device = next(model.parameters()).device
            enabled_amp = self.experiment.amp and self.device.type == "cuda"
            self.data = {"nc": len(CLASSES), "names": dict(enumerate(CLASSES)),
                         "val": str(self.experiment.data_root / self.experiment.validation_split)}
        model = de_parallel(model).float().eval()
        self.args.half = False
        self.init_metrics(model)
        timers = tuple(Profile(device=self.device) for _ in range(4))
        seen_ids: List[str] = []
        self.run_callbacks("on_val_start")
        iterator = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        for index, raw in enumerate(iterator):
            self.batch_i = index
            self.run_callbacks("on_val_batch_start")
            seen_ids.extend(raw["image_id"])
            with timers[0]:
                batch = self.preprocess(raw)
            with timers[1], autocast(enabled_amp):
                predictions = model.predict(batch["img"], conditions=batch["conditions"])
            with timers[2]:
                if self.training:
                    with autocast(enabled_amp):
                        self.loss += model.loss(batch, predictions)[1]
            with timers[3]:
                detections = self.postprocess(predictions)
            self.update_metrics(detections, batch)
            self.run_callbacks("on_val_batch_end")
        expected = expected_image_ids(self.dataloader.dataset)
        if len(seen_ids) != len(expected) or len(set(seen_ids)) != len(expected) or set(seen_ids) != set(expected):
            raise ValueError("Validation predictions do not cover each requested image exactly once.")
        stats = self.get_stats()
        self.speed = dict(zip(self.speed, (timer.t / len(expected) * 1000 for timer in timers)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            stats.update(trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val"))
        LOGGER.info("Selection metric: native ProbIoU AP50; geometric VOC07 metrics require separate reevaluation.")
        return {key: float(value) for key, value in stats.items()}
