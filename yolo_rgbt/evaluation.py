"""Export original-pixel predictions and reuse the unchanged geometric evaluator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import ATRUMODYOLODataset, CLASSES, move_batch
from .runtime import REPO_ROOT
from .validator import PairedOBBValidator, postprocess_predictions


def _legacy_metrics() -> Any:
    """Load only the self-contained original metric file, not engine imports.

    Returns:
        Read-only legacy module. Evaluation deliberately uses nproc=1 so no
        dynamic-module pickling is required. The kernel is not copied/modified.
    """
    name = "_yolo_rgbt_legacy_mmeval"
    if name not in sys.modules:
        path = REPO_ROOT / "engine/data/dataset/mmeval.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the original geometry evaluator: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module, loader: DataLoader, config: ExperimentConfig, device: torch.device,
) -> List[Dict[str, Any]]:
    """Collect each requested image once, using native FP32 rotated NMS.

    Args:
        model: Paired checkpoint model.
        loader: Full paired split in a deterministic order.
        config: Metadata, confidence and NMS settings.
        device: Inference device.

    Returns:
        Serializable per-image IDs, unclipped pixel/radian boxes, scores and
        eleven-class labels. Object annotations do not enter model.predict().
    """
    model = model.to(device).float().eval()
    result: List[Dict[str, Any]] = []
    for raw in loader:
        batch = move_batch(raw, device)
        with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
            output = model.predict(batch["img"], conditions=batch["conditions"])
        detections = postprocess_predictions(output, config)
        for image_id, pred in zip(raw["image_id"], detections):
            boxes = pred[:, [0, 1, 2, 3, 6]].float().cpu()
            result.append({
                "image_id": image_id, "boxes": boxes.tolist(),
                "scores": pred[:, 4].float().cpu().tolist(),
                "labels": pred[:, 5].to(torch.int64).cpu().tolist(),
            })
    expected = loader.dataset.image_ids
    actual = [row["image_id"] for row in result]
    if len(actual) != len(expected) or len(set(actual)) != len(expected) or set(actual) != set(expected):
        raise ValueError("Prediction export does not cover every requested image exactly once.")
    return result


def prediction_payload(
    dataset: ATRUMODYOLODataset, predictions: Sequence[Mapping[str, Any]], config: ExperimentConfig,
    checkpoint_sha256: str,
) -> Dict[str, Any]:
    """Attach geometry/GT provenance so cached predictions can be reevaluated.

    Args:
        dataset: Original paired split.
        predictions: Collected original-pixel detections.
        config: Effective experiment.
        checkpoint_sha256: Actual checkpoint file checksum.

    Returns:
        JSON-ready prediction artifact with explicit units and reference frame.
    """
    return {
        "schema_version": 1,
        "metadata": {
            "reference": "IR", "box_format": "cx,cy,w,h,angle_radians",
            "class_names": list(CLASSES), "checkpoint_sha256": checkpoint_sha256,
            "dataset_audit": dataset.audit, "experiment": config.as_dict(),
            "nms": "native ProbIoU rotated NMS in FP32",
        },
        "predictions": list(predictions),
    }


def evaluate_payload(
    dataset: ATRUMODYOLODataset, payload: Mapping[str, Any], full_metrics: bool = True,
) -> Dict[str, Any]:
    """Reevaluate the same predictions using polygon IoU and VOC07 AP.

    Args:
        dataset: Original IR-reference annotations.
        payload: Exported prediction artifact.
        full_metrics: Compute AP50/AP75/mean over 0.50:0.05:0.95 if True.

    Returns:
        Geometry metric definitions, per-IoU/class results and GT counts.
        AP50_95 retains VOC07 interpolation and is not native/COCO AP.
    """
    metadata = payload.get("metadata", {})
    if (payload.get("schema_version") != 1 or metadata.get("reference") != "IR"
            or metadata.get("box_format") != "cx,cy,w,h,angle_radians"
            or metadata.get("class_names") != list(CLASSES)):
        raise ValueError("Prediction artifact has a different reference/units/class mapping.")
    audit = metadata.get("dataset_audit", {})
    for key in ("annotations_sha256", "sample_ids_sha256", "split", "input_size"):
        if audit.get(key) != dataset.audit[key]:
            raise ValueError(f"Prediction/GT manifest mismatch: {key}.")
    rows = payload.get("predictions")
    if not isinstance(rows, list):
        raise ValueError("Missing prediction list.")
    mapping = {row["image_id"]: row for row in rows}
    if len(mapping) != len(rows) or set(mapping) != set(dataset.image_ids):
        raise ValueError("Cached predictions have duplicate/missing/extra image IDs.")
    dets: List[List[np.ndarray]] = []
    gts: List[Dict[str, np.ndarray]] = []
    for index, image_id in enumerate(dataset.image_ids):
        row = mapping[image_id]
        boxes = np.asarray(row["boxes"], dtype=np.float32).reshape(-1, 5)
        scores = np.asarray(row["scores"], dtype=np.float32)
        raw_labels = np.asarray(row["labels"])
        if len(raw_labels) and not np.issubdtype(raw_labels.dtype, np.integer):
            raise ValueError(f"{image_id}: class labels must be integers, not rounded estimates.")
        labels = raw_labels.astype(np.int64)
        if scores.shape != (len(boxes),) or labels.shape != scores.shape:
            raise ValueError(f"{image_id}: box/score/label counts differ.")
        if (not np.isfinite(boxes).all() or not np.isfinite(scores).all()
                or (boxes[:, 2:4] <= 0).any() or ((scores < 0) | (scores > 1)).any()
                or ((labels < 0) | (labels >= len(CLASSES))).any()):
            raise ValueError(f"{image_id}: invalid cached detections.")
        dets.append([
            np.concatenate((boxes[labels == cls], scores[labels == cls, None]), axis=1)
            for cls in range(len(CLASSES))
        ])
        gts.append(dataset.get_ann_info(index))
    module = _legacy_metrics()
    thresholds = [round(0.5 + 0.05 * index, 2) for index in range(10)] if full_metrics else [0.5]
    maps: Dict[str, float] = {}
    classes: Dict[str, List[Dict[str, Any]]] = {}
    for threshold in thresholds:
        mean, results = module.eval_rbbox_map(
            dets, gts, iou_thr=threshold, use_07_metric=True, nproc=1, use_torch_sat=False
        )
        key = f"{threshold:.2f}"
        maps[key] = float(mean)
        classes[key] = [
            {"class": name, "AP": float(record["ap"]) if np.isfinite(record["ap"]) else None,
             "GT": int(record["num_gts"]),
             "detections": int(record["num_dets"])}
            for name, record in zip(CLASSES, results)
        ]
    metrics = {"AP50": maps["0.50"]}
    if full_metrics:
        metrics.update(AP75=maps["0.75"], AP50_95=float(np.mean(list(maps.values()))))
    return {
        "metric": "geometric rotated polygon IoU + VOC2007 11-point interpolation",
        "reference": "IR", "metrics": metrics, "AP_by_IoU": maps, "per_class": classes,
        "iou_thresholds": thresholds, "dataset_audit": dataset.audit,
        "prediction_checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "metric_source_sha256": hashlib.sha256((REPO_ROOT / "engine/data/dataset/mmeval.py").read_bytes()).hexdigest(),
    }
