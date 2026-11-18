"""Condition-safe paired augmentations for RGB--IR oriented boxes.

Every enabled transform is applied identically to the IR and RGB images and to
their shared IR-reference rotated boxes. Mosaic, MixUp, Copy-Paste, shear,
perspective, and photometric jitter are intentionally absent: the first three
invalidate one condition tuple per image, shear/perspective turn rectangles into
non-representable quadrilaterals, and colour/intensity changes can contradict
weather and illumination metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PairedAugmentConfig:
    """Explicit probabilities and bounds for the geometry-only train policy.

    Args:
        enabled: Master switch. Validation must pass False.
        hflip: Probability of horizontal mirroring.
        vflip: Probability of vertical mirroring.
        rotate_prob: Probability of one in-plane rotation.
        rotate_deg: Maximum absolute rotation, in degrees.
        scale_prob: Probability of one isotropic zoom.
        scale_min: Minimum zoom factor.
        scale_max: Maximum zoom factor.
        translate_prob: Probability of translation.
        translate_frac: Maximum absolute x/y translation as an image fraction.
        fill_value: Constant border value shared by both modalities.
    """

    enabled: bool = False
    hflip: float = 0.5
    vflip: float = 0.5
    rotate_prob: float = 0.5
    rotate_deg: float = 10.0
    scale_prob: float = 0.5
    scale_min: float = 0.9
    scale_max: float = 1.1
    translate_prob: float = 0.5
    translate_frac: float = 0.05
    fill_value: int = 0

    def validate(self) -> None:
        """Reject values that could silently produce an invalid augmentation."""
        for name in ("hflip", "vflip", "rotate_prob", "scale_prob", "translate_prob"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1].")
        if not 0.0 <= float(self.rotate_deg) <= 45.0:
            raise ValueError("rotate_deg must lie in [0,45] degrees.")
        if not 0.0 < float(self.scale_min) <= float(self.scale_max) <= 2.0:
            raise ValueError("Require 0 < scale_min <= scale_max <= 2.")
        if not 0.0 <= float(self.translate_frac) <= 0.25:
            raise ValueError("translate_frac must lie in [0,0.25].")
        if not 0 <= int(self.fill_value) <= 255:
            raise ValueError("fill_value must lie in [0,255].")


def _flip_boxes_horizontal(boxes: np.ndarray, width: int) -> np.ndarray:
    """Mirror pixel-space rotated boxes across the vertical image axis.

    Args:
        boxes: float32 [N,5] cx,cy,w,h,theta in pixels/radians.
        width: Image width.

    Returns:
        Independent float32 [N,5] boxes; angles are normalized later.
    """
    result = boxes.copy()
    if len(result):
        result[:, 0] = float(width) - result[:, 0]
        result[:, 4] = np.pi - result[:, 4]
    return result


def _flip_boxes_vertical(boxes: np.ndarray, height: int) -> np.ndarray:
    """Mirror pixel-space rotated boxes across the horizontal image axis.

    Args:
        boxes: float32 [N,5] cx,cy,w,h,theta in pixels/radians.
        height: Image height.

    Returns:
        Independent float32 [N,5] boxes; angles are normalized later.
    """
    result = boxes.copy()
    if len(result):
        result[:, 1] = float(height) - result[:, 1]
        # Reflection changes the line slope to -theta. Modulo pi, this equals
        # pi-theta. The later long-edge normalization may swap width/height.
        result[:, 4] = np.pi - result[:, 4]
    return result


def _visible_boxes(
    boxes: np.ndarray, width: int, height: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Drop only boxes whose complete rotated footprint is outside the image.

    Partial border boxes retain their original center, size, and angle. No
    clipping is performed because clipping would change the rotated rectangle.

    Args:
        boxes: float32 [N,5] pixel/radian boxes.
        width: Image width.
        height: Image height.

    Returns:
        Visible boxes, a boolean [N] mask in original order, and the number of
        fully outside boxes.
    """
    if not len(boxes):
        return boxes, np.zeros(0, dtype=bool), 0
    frame = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    keep = []
    for box in boxes:
        polygon = cv2.boxPoints(((float(box[0]), float(box[1])), (float(box[2]), float(box[3])),
                                 float(np.degrees(box[4])))).astype(np.float32)
        area, _ = cv2.intersectConvexConvex(polygon, frame)
        keep.append(float(area) > 1e-6)
    mask = np.asarray(keep, dtype=bool)
    return boxes[mask], mask, int((~mask).sum())


def augment_pair(
    ir: np.ndarray, rgb: np.ndarray, boxes: np.ndarray, config: PairedAugmentConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply one shared geometry augmentation to an RGB--IR pair and boxes.

    Args:
        ir: uint8 [H,W,3] IR image.
        rgb: uint8 [H,W,3] visible image, same shape as ``ir``.
        boxes: float32 [N,5] IR-reference cx,cy,w,h,theta in pixels/radians.
        config: Validated augmentation settings.
        rng: Explicit NumPy generator for reproducibility.

    Returns:
        Augmented IR, RGB, float32 boxes, a boolean [N] visibility mask in
        original box order, and an audit dictionary. Conditions are not passed
        here and therefore cannot be changed accidentally.

    Raises:
        ValueError: Shapes/dtypes or the augmentation settings are invalid.
    """
    config.validate()
    if ir.ndim != 3 or ir.shape[2] != 3 or rgb.shape != ir.shape:
        raise ValueError(f"Expected equal [H,W,3] images; got {ir.shape}/{rgb.shape}.")
    if ir.dtype != np.uint8 or rgb.dtype != np.uint8:
        raise ValueError("Augmentation expects uint8 images before one-time normalization.")
    if boxes.ndim != 2 or boxes.shape[1] != 5 or boxes.dtype != np.float32:
        raise ValueError("Expected float32 [N,5] pixel/radian boxes.")
    if not config.enabled:
        return ir.copy(), rgb.copy(), boxes.copy(), np.ones(len(boxes), dtype=bool), {
            "enabled": False, "hflip": False, "vflip": False,
            "rotation_deg": 0.0, "scale": 1.0, "dx": 0.0, "dy": 0.0, "dropped_boxes": 0,
        }

    height, width = ir.shape[:2]
    result_ir, result_rgb, result_boxes = ir.copy(), rgb.copy(), boxes.copy()
    audit: Dict[str, Any] = {"enabled": True, "hflip": False, "vflip": False,
                             "rotation_deg": 0.0, "scale": 1.0, "dx": 0.0, "dy": 0.0,
                             "dropped_boxes": 0}

    if rng.random() < config.hflip:
        result_ir = np.ascontiguousarray(result_ir[:, ::-1])
        result_rgb = np.ascontiguousarray(result_rgb[:, ::-1])
        result_boxes = _flip_boxes_horizontal(result_boxes, width)
        audit["hflip"] = True
    if rng.random() < config.vflip:
        result_ir = np.ascontiguousarray(result_ir[::-1])
        result_rgb = np.ascontiguousarray(result_rgb[::-1])
        result_boxes = _flip_boxes_vertical(result_boxes, height)
        audit["vflip"] = True

    angle_deg = float(rng.uniform(-config.rotate_deg, config.rotate_deg)
                      if rng.random() < config.rotate_prob else 0.0)
    scale = float(rng.uniform(config.scale_min, config.scale_max)
                  if rng.random() < config.scale_prob else 1.0)
    dx = float(rng.uniform(-config.translate_frac, config.translate_frac) * width
               if rng.random() < config.translate_prob else 0.0)
    dy = float(rng.uniform(-config.translate_frac, config.translate_frac) * height
               if rng.random() < config.translate_prob else 0.0)
    if angle_deg or scale != 1.0 or dx or dy:
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, scale)
        matrix[0, 2] += dx
        matrix[1, 2] += dy
        result_ir = cv2.warpAffine(
            result_ir, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(config.fill_value,) * 3,
        )
        result_rgb = cv2.warpAffine(
            result_rgb, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(config.fill_value,) * 3,
        )
        if len(result_boxes):
            homogeneous = np.concatenate(
                (result_boxes[:, :2], np.ones((len(result_boxes), 1), dtype=np.float32)), axis=1
            )
            result_boxes[:, :2] = homogeneous @ matrix.T.astype(np.float32)
            result_boxes[:, 2:4] *= scale
            # getRotationMatrix2D maps a direction angle theta to theta-angle.
            result_boxes[:, 4] -= np.float32(np.deg2rad(angle_deg))

    result_boxes, keep_mask, dropped = _visible_boxes(result_boxes, width, height)
    audit.update(rotation_deg=angle_deg, scale=scale, dx=dx, dy=dy, dropped_boxes=dropped)
    return (result_ir, result_rgb, result_boxes.astype(np.float32, copy=False), keep_mask, audit)
