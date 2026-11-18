"""Read paired ATR-UMOD XML/GT and apply optional condition-safe augmentation.

Training may use shared geometry-only transforms from augment.py. Validation
keeps deterministic inputs. YOLO predicts unrestricted box centers, so a center
outside the image is not removed here. Native small-box loss filtering is
audited separately. Evaluation always uses the original IR-reference boxes.
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augment import PairedAugmentConfig, augment_pair


CLASSES = ("car", "suv", "van", "bus", "freight_car", "truck", "motorcycle",
           "trailer", "excavator", "crane", "tank_truck")
CONDITION_COUNTS = {"weather": 10, "illumination": 7, "location": 11}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _files(folder: Path, extensions: set[str]) -> Dict[str, Path]:
    """Index one file per sample ID and reject duplicates/empty files.

    Args:
        folder: Image or annotation directory.
        extensions: Accepted lowercase suffixes.

    Returns:
        Sample ID to file path mapping.
    """
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    result: Dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in extensions:
            continue
        if path.stem in result or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Duplicate/empty sample file: {path}")
        result[path.stem] = path
    if not result:
        raise ValueError(f"No samples in {folder}.")
    return result


def _number(element: ET.Element, key: str, context: str) -> float:
    """Read a required finite numeric XML field, without default values.

    Args:
        element: Parent XML node.
        key: Child field.
        context: File/object identifier for error messages.

    Returns:
        Finite float.
    """
    text = element.findtext(key)
    if text is None:
        raise ValueError(f"{context}: missing {key}.")
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{context}: nonfinite {key}.")
    return value


def parse_conditions(root: ET.Element, context: str) -> Dict[str, torch.Tensor]:
    """Split environment IDs and normalized geometry into separate fields.

    Args:
        root: One image-level XML annotation.
        context: Annotation path.

    Returns:
        weather/illumination/location: int64 scalar tensors.
        height/angle: float32 [1], normalized as (meters-80)/220 and degrees/75.
        No object class, box coordinate/angle, acquisition ID, or date is read
        into this dictionary. location means the image-level environment label.
    """
    result: Dict[str, torch.Tensor] = {}
    for key, count in CONDITION_COUNTS.items():
        text = root.findtext(key)
        if text is None:
            raise ValueError(f"{context}: missing environmental condition {key}.")
        value = int(text)
        if not 0 <= value < count:
            raise ValueError(f"{context}: {key}={value} outside [0,{count - 1}].")
        result[key] = torch.tensor(value, dtype=torch.int64)
    height = _number(root, "UAVheight", context)
    angle = _number(root, "UAVangle", context)
    if not 80.0 <= height <= 300.0 or not 0.0 <= angle <= 75.0:
        raise ValueError(f"{context}: unsupported height={height}m, angle={angle}deg.")
    result["height"] = torch.tensor([(height - 80.0) / 220.0], dtype=torch.float32)
    result["angle"] = torch.tensor([angle / 75.0], dtype=torch.float32)
    return result


def parse_boxes(root: ET.Element, context: str) -> Tuple[np.ndarray, np.ndarray, Counter]:
    """Parse IR boxes with the same long-edge convention as the old loader.

    Args:
        root: Annotation XML.
        context: File path.

    Returns:
        boxes: float32 [N,5], original pixels and angle radians.
        labels: int64 [N], fixed eleven-class order.
        audit: Raw counts and explicitly excluded <=1-pixel degenerate boxes.
            Center/corner bounds and duplicate boxes do not cause removal.
    """
    rows: List[List[float]] = []
    labels: List[int] = []
    audit: Counter = Counter()
    for index, obj in enumerate(root.findall("object")):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASSES:
            raise ValueError(f"{context}, object {index}: unknown class {name!r}.")
        audit["raw_objects"] += 1
        audit[f"raw/{name}"] += 1
        for flag in ("difficult", "truncated"):
            text = obj.findtext(flag)
            if text is not None and int(text) != 0:
                raise ValueError(f"{context}: nonzero {flag} needs an explicit ignore policy.")
        rbox, polygon = obj.find("robndbox"), obj.find("polygon")
        if rbox is not None:
            # The release has a degenerate <=1px box whose center is NaN.
            # Match the old documented side filter BEFORE using coordinates,
            # while explicitly counting its invalid original geometry.
            width = _number(rbox, "w", context)
            height = _number(rbox, "h", context)
            if width <= 1.0 or height <= 1.0:
                audit["parser_small_objects"] += 1
                audit[f"parser_small/{name}"] += 1
                raw_geometry = [rbox.findtext(key) for key in ("cx", "cy", "angle")]
                if any(value is None or not math.isfinite(float(value)) for value in raw_geometry):
                    audit["small_box_nonfinite_geometry"] += 1
                    audit[f"small_box_nonfinite/{name}"] += 1
                continue
            cx, cy, width, height, angle = (
                _number(rbox, key, context) for key in ("cx", "cy", "w", "h", "angle")
            )
        elif polygon is not None:
            points = np.array([
                [_number(polygon, f"x{k}", context), _number(polygon, f"y{k}", context)]
                for k in range(1, 5)
            ], dtype=np.float32)
            (cx, cy), (width, height), angle = cv2.minAreaRect(points)
        else:
            raise ValueError(f"{context}: unsupported box representation.")
        if width <= 1.0 or height <= 1.0:
            audit["parser_small_objects"] += 1
            audit[f"parser_small/{name}"] += 1
            continue
        if width < height:
            width, height = height, width
            angle += 90.0
        # Reproduce the legacy float32 normalized-angle -> radians conversion.
        normalized = np.float32((angle % 180.0) / 180.0)
        rows.append([cx, cy, width, height, float(normalized * np.float32(np.pi))])
        labels.append(CLASSES.index(name))
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5), np.asarray(labels, dtype=np.int64), audit


def normalized_native_boxes(boxes: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Convert to YOLO's equivalent [0,pi/2) box representation.

    Args:
        boxes: Original-pixel/radian float32 [N,5].
        size: Image (height,width).

    Returns:
        float32 [N,5]: cx/W, cy/H, w/W, h/H, radians. Centers outside [0,1]
        and corners outside the frame are preserved. Width/height swapping
        only changes representation, not the geometric rectangle.
    """
    result = boxes.copy()
    if len(result):
        angle = np.remainder(result[:, 4], np.float32(np.pi))
        swap = angle >= np.float32(np.pi / 2)
        result[swap, 2:4] = result[swap, 2:4][:, ::-1]
        result[:, 4] = np.where(swap, angle - np.float32(np.pi / 2), angle)
        height, width = size
        result[:, :4] /= np.asarray((width, height, width, height), dtype=np.float32)
    return result


class ATRUMODYOLODataset(Dataset):
    """Paired native-resolution samples, strict metadata, and IR-reference GT."""

    rect = False
    mosaic = False

    def __init__(
        self, root: str | Path, split: str, input_size: Tuple[int, int] = (512, 640),
        min_gt_side_px: float = 2.0, augmentation: PairedAugmentConfig | None = None,
        seed: int = 0,
    ) -> None:
        """Index and audit the split without decoding all image pixels.

        Args:
            root: Unmodified ATR-UMOD root.
            split: train or test.
            input_size: Fixed (H,W), both multiples of 32.
            min_gt_side_px: Explicit native loss threshold, in input pixels.
            augmentation: Optional condition-safe geometry transforms.
            seed: Base seed for deterministic random augmentation streams.
        """
        if split not in {"train", "test"}:
            raise ValueError("ATR-UMOD split must be train or test.")
        self.root, self.split = Path(root).resolve(), split
        self.input_size = tuple(input_size)
        self.min_gt_side_px = float(min_gt_side_px)
        settings = augmentation or PairedAugmentConfig(enabled=False)
        settings.validate()
        # Validation/test must remain deterministic even if a caller passes an
        # enabled object by mistake.
        self.augmentation = settings if split == "train" else replace(settings, enabled=False)
        self.rng = np.random.default_rng(seed)
        base = self.root / split
        ir = _files(base / "images_ir", IMAGE_EXTENSIONS)
        rgb = _files(base / "images", IMAGE_EXTENSIONS)
        annotations = _files(base / "labels_ir", {".xml"})
        if set(ir) != set(rgb) or set(ir) != set(annotations):
            raise ValueError(f"{base}: image/XML ID sets differ; no samples are silently skipped.")
        self.image_ids = sorted(ir)
        self.im_files = [str(ir[key]) for key in self.image_ids]
        self.rgb_files = [str(rgb[key]) for key in self.image_ids]
        self.records: List[Dict[str, Any]] = []
        self.labels: List[Dict[str, Any]] = []
        totals: Counter = Counter()
        exclusion_examples: List[Dict[str, Any]] = []
        digest = hashlib.sha256()
        for key in self.image_ids:
            with Image.open(ir[key]) as image:
                ir_size = (image.height, image.width)
            with Image.open(rgb[key]) as image:
                rgb_size = (image.height, image.width)
            if ir_size != self.input_size or rgb_size != ir_size:
                raise ValueError(f"{key}: require paired {self.input_size}; got {ir_size}/{rgb_size}.")
            raw = annotations[key].read_bytes()
            digest.update(key.encode("utf-8") + b"\0" + raw)
            xml = ET.fromstring(raw)
            xml_size = (int(_number(xml.find("size"), "height", str(annotations[key]))),
                        int(_number(xml.find("size"), "width", str(annotations[key])))) if xml.find("size") is not None else None
            if xml_size != ir_size:
                raise ValueError(f"{key}: XML and image dimensions disagree.")
            conditions = parse_conditions(xml, str(annotations[key]))
            boxes, classes, audit = parse_boxes(xml, str(annotations[key]))
            totals.update(audit)
            if audit.get("small_box_nonfinite_geometry", 0):
                exclusion_examples.append({
                    "image_id": key, "XML": str(annotations[key]),
                    "reason": "degenerate <=1px box with nonfinite original geometry",
                    "count": audit["small_box_nonfinite_geometry"],
                })
            totals["parser_kept"] += len(boxes)
            outside = ((boxes[:, 0] < 0) | (boxes[:, 0] > ir_size[1])
                       | (boxes[:, 1] < 0) | (boxes[:, 1] > ir_size[0]))
            small = (boxes[:, 2] < self.min_gt_side_px) | (boxes[:, 3] < self.min_gt_side_px)
            totals["center_outside_retained"] += int(outside.sum())
            totals["native_loss_small_objects"] += int(small.sum())
            for label in classes[small]:
                totals[f"native_loss_small/{CLASSES[int(label)]}"] += 1
            for label in classes:
                totals[f"kept/{CLASSES[int(label)]}"] += 1
            native = normalized_native_boxes(boxes, ir_size)
            self.records.append({"boxes": boxes, "classes": classes, "native": native, "conditions": conditions})
            self.labels.append({"cls": classes[:, None].astype(np.float32), "bboxes": native, "shape": ir_size})
        self.audit = {
            "split": split, "images": len(self.image_ids), "input_size": list(self.input_size),
            "reference": "IR", "class_names": list(CLASSES), "counts": dict(totals),
            "annotations_sha256": digest.hexdigest(),
            "sample_ids_sha256": hashlib.sha256("\n".join(self.image_ids).encode()).hexdigest(),
            "random_augmentation": self.augmentation.enabled,
            "augmentation": asdict(self.augmentation),
            "border_policy": "preserve_original_boxes",
            "native_loss_min_side_px": self.min_gt_side_px,
            "nonfinite_degenerate_box_examples": exclusion_examples,
        }

    def __len__(self) -> int:
        """Return the number of verified paired image IDs."""
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Decode one fixed ID without retrying a different image.

        Args:
            index: Position in the sorted ID manifest.

        Returns:
            img: uint8 [6,H,W], IR RGB-channel triplet followed by RGB.
            cls: float32 [N,1]; bboxes: float32 [N,5] normalized xywhr.
            conditions: three int64 scalars and two float32 [1] fields.
            im_file/ori_shape identify the original IR reference frame.
        """
        record = self.records[index]
        with Image.open(self.im_files[index]) as image:
            ir = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(self.rgb_files[index]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        if ir.shape[:2] != self.input_size or rgb.shape != ir.shape:
            raise ValueError(f"{self.image_ids[index]}: image changed after indexing.")
        if self.augmentation.enabled:
            ir, rgb, boxes, keep_mask, augmentation_audit = augment_pair(
                ir, rgb, record["boxes"].copy(), self.augmentation, self.rng
            )
            native = normalized_native_boxes(boxes, self.input_size)
            classes = record["classes"][keep_mask]
        else:
            native = record["native"].copy()
            classes = record["classes"]
            augmentation_audit = {"enabled": False, "dropped_boxes": 0}
        channels = np.concatenate((ir, rgb), axis=2).transpose(2, 0, 1).copy()
        return {
            "img": torch.from_numpy(channels),
            "cls": torch.from_numpy(classes.astype(np.float32)[:, None]),
            "bboxes": torch.from_numpy(native),
            "conditions": {key: value.clone() for key, value in record["conditions"].items()},
            "ori_shape": self.input_size, "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
            "im_file": self.im_files[index], "rgb_file": self.rgb_files[index],
            "image_id": self.image_ids[index], "augmentation": augmentation_audit,
        }

    def get_ann_info(self, index: int) -> Dict[str, np.ndarray]:
        """Return unclipped, original-pixel GT for geometric evaluation.

        Args:
            index: Dataset index.

        Returns:
            bboxes float32 [N,5] in pixels/radians and labels int64 [N].
        """
        record = self.records[index]
        return {"bboxes": record["boxes"].copy(), "labels": record["classes"].copy()}

    @staticmethod
    def collate_fn(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Assemble native YOLO labels while keeping conditions explicit.

        Args:
            samples: Nonempty list of paired samples.

        Returns:
            img [B,6,H,W], bboxes [sum N,5], cls [sum N,1],
            batch_idx [sum N], conditions IDs [B] and geometry [B,1].
        """
        if not samples:
            raise ValueError("Cannot collate an empty paired batch.")
        result: Dict[str, Any] = {
            "img": torch.stack([sample["img"] for sample in samples]),
            "cls": torch.cat([sample["cls"] for sample in samples]),
            "bboxes": torch.cat([sample["bboxes"] for sample in samples]),
            "batch_idx": torch.cat([
                torch.full((len(sample["cls"]),), index, dtype=torch.int64)
                for index, sample in enumerate(samples)
            ]),
            "conditions": {
                key: torch.stack([sample["conditions"][key] for sample in samples])
                for key in samples[0]["conditions"]
            },
        }
        for key in ("ori_shape", "ratio_pad", "im_file", "rgb_file", "image_id", "augmentation"):
            result[key] = [sample[key] for sample in samples]
        return result


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensors and scale uint8 images once, without touching GT geometry.

    Args:
        batch: Collated dataset batch.
        device: Model device.

    Returns:
        Independent mapping with float32 [B,6,H,W] images in [0,1].
    """
    if batch["img"].dtype != torch.uint8:
        raise ValueError("Expected unscaled uint8 images; refusing double normalization.")
    result = dict(batch)
    result["img"] = batch["img"].to(device, non_blocking=True).float().div_(255.0)
    for key in ("cls", "bboxes", "batch_idx"):
        result[key] = batch[key].to(device, non_blocking=True)
    result["conditions"] = {
        key: value.to(device, non_blocking=True) for key, value in batch["conditions"].items()
    }
    return result
