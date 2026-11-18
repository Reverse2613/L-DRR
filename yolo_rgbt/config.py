"""Strict, reviewable experiment settings; no modification of native YAMLs."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import yaml

from .augment import PairedAugmentConfig
from .runtime import REPO_ROOT


class StrictLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently keeping the last value."""


def _mapping(loader: StrictLoader, node: yaml.MappingNode) -> Dict[str, Any]:
    """Construct a mapping while identifying repeated keys.

    Args:
        loader: Current safe YAML loader.
        node: Mapping node with its source location.

    Returns:
        Parsed mapping.
    """
    result: Dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f"Duplicate YAML key {key!r} at {key_node.start_mark}.")
        result[key] = loader.construct_object(value_node)
    return result


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _merge(left: Dict[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge nested mappings; scalar/list values are replaced explicitly.

    Args:
        left: Inherited configuration.
        right: Child configuration.

    Returns:
        Independent merged mapping.
    """
    result = copy.deepcopy(left)
    for key, value in right.items():
        result[key] = _merge(result[key], value) if (
            key in result and isinstance(result[key], dict) and isinstance(value, dict)
        ) else copy.deepcopy(value)
    return result


def _read(path: Path, stack: Tuple[Path, ...] = ()) -> Dict[str, Any]:
    """Read relative include files, detecting include cycles.

    Args:
        path: YAML file.
        stack: Current include ancestry.

    Returns:
        Fully expanded mapping.
    """
    path = path.resolve()
    if path in stack:
        raise ValueError(f"Configuration include cycle: {stack + (path,)}")
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=StrictLoader)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping.")
    includes = data.pop("__include__", [])
    if not isinstance(includes, list) or any(not isinstance(item, str) for item in includes):
        raise ValueError(f"{path}: __include__ must be a list of paths.")
    expanded: Dict[str, Any] = {}
    for include in includes:
        expanded = _merge(expanded, _read(path.parent / include, stack + (path,)))
    return _merge(expanded, data)


def _path(value: str) -> Path:
    """Resolve experiment paths against the repository root.

    Args:
        value: Absolute path or repository-relative path.

    Returns:
        Absolute path.
    """
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings shared by the baseline and all controlled variants.

    Images use native [B,6,512,640] tensors ordered IR then RGB. Environment
    inputs are three int64 [B] fields; geometry uses float32 [B,1] height/angle.
    """

    source: Path
    output_dir: Path
    data_root: Path
    pretrained: Path
    anchor_dir: Path
    texts_path: Path
    variant: str = "condition"
    alignment: str = "geometry"
    condition_mode: str = "real"
    enable_film: bool = True
    dynamic_fusion: bool = True
    score_condition: bool = True
    max_shift_px: float = 12.0
    input_size: Tuple[int, int] = (512, 640)
    validation_split: str = "test"
    epochs: int = 36
    batch: int = 8
    val_batch: int = 8
    workers: int = 4
    seed: int = 0
    optimizer: str = "SGD"
    lr_new: float = 0.01
    lr_pretrained: float = 0.001
    lr_final_ratio: float = 0.01
    momentum: float = 0.937
    warmup_momentum: float = 0.8
    nesterov: bool = True
    weight_decay: float = 0.0005
    warmup_iterations: int = 4500
    augment: bool = False
    aug_hflip: float = 0.5
    aug_vflip: float = 0.5
    aug_rotate_prob: float = 0.5
    aug_rotate_deg: float = 10.0
    aug_scale_prob: float = 0.5
    aug_scale_min: float = 0.9
    aug_scale_max: float = 1.1
    aug_translate_prob: float = 0.5
    aug_translate_frac: float = 0.05
    aug_fill_value: int = 0
    clip_norm: float = 10.0
    amp: bool = True
    min_gt_side_px: float = 2.0
    conf: float = 0.001
    nms_iou: float = 0.7
    max_det: int = 300

    def as_dict(self) -> Dict[str, Any]:
        """Return serializable settings for manifests/checkpoints.

        Returns:
            Mapping with absolute paths represented as strings.
        """
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}

    def architecture(self) -> Dict[str, Any]:
        """Describe controls that must match when resuming/evaluating.

        Returns:
            Architecture and metadata input contract.
        """
        fields = ("variant", "alignment", "condition_mode", "enable_film", "dynamic_fusion",
                  "score_condition", "max_shift_px", "input_size", "min_gt_side_px")
        result = {name: getattr(self, name) for name in fields}
        result.update(model="yolo11s-obb", fusion_indices=(4, 6, 10), reference="IR")
        return result

    def augmentation(self) -> PairedAugmentConfig:
        """Return the validated condition-safe train augmentation settings.

        Returns:
            Geometry-only paired augmentation settings. Validation split data
            is forced to disable these settings by the dataset.

        Raises:
            ValueError: A probability or geometric bound is invalid.
        """
        result = PairedAugmentConfig(
            enabled=self.augment, hflip=self.aug_hflip, vflip=self.aug_vflip,
            rotate_prob=self.aug_rotate_prob, rotate_deg=self.aug_rotate_deg,
            scale_prob=self.aug_scale_prob, scale_min=self.aug_scale_min,
            scale_max=self.aug_scale_max, translate_prob=self.aug_translate_prob,
            translate_frac=self.aug_translate_frac, fill_value=self.aug_fill_value,
        )
        result.validate()
        return result

    def native_args(self, device: str = "0") -> Dict[str, Any]:
        """Build native trainer arguments; custom data owns safe augmentation.

        Args:
            device: Single CUDA device or CPU, e.g. "0" or "cpu".

        Returns:
            Ultralytics arguments. The custom loader keeps the actual
            rectangular input; imgsz only records its maximum dimension.
        """
        return {
            "model": str(self.pretrained), "data": str(self.source),
            "project": str(self.output_dir.parent), "name": self.output_dir.name,
            "exist_ok": False, "task": "obb", "device": device,
            "epochs": self.epochs, "batch": self.batch, "nbs": self.batch,
            "imgsz": max(self.input_size), "workers": self.workers, "seed": self.seed,
            "optimizer": self.optimizer, "lr0": self.lr_new,
            "lrf": self.lr_final_ratio, "momentum": self.momentum,
            "weight_decay": self.weight_decay, "cos_lr": True,
            "warmup_epochs": 1.0, "warmup_bias_lr": 0.0,
            "warmup_momentum": self.warmup_momentum,
            "amp": self.amp, "deterministic": True, "freeze": 0,
            "patience": 0, "save": True, "save_period": 60, "val": True,
            "plots": False, "verbose": False, "cache": False,
            "rect": False, "multi_scale": False, "close_mosaic": 0,
            "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
            "degrees": self.aug_rotate_deg if self.augment else 0.0,
            "translate": self.aug_translate_frac if self.augment else 0.0,
            "scale": max(self.aug_scale_max - 1.0, 1.0 - self.aug_scale_min)
                     if self.augment else 0.0,
            "shear": 0.0, "perspective": 0.0,
            "flipud": self.aug_vflip if self.augment else 0.0,
            "fliplr": self.aug_hflip if self.augment else 0.0,
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
            "erasing": 0.0, "auto_augment": "none", "crop_fraction": 1.0,
            "augment": False, "save_hybrid": False, "save_json": False,
            "conf": self.conf, "iou": self.nms_iou, "max_det": self.max_det,
            "box": 7.5, "cls": 0.5, "dfl": 1.5,
        }


def load_config(path: str | Path, **overrides: Any) -> ExperimentConfig:
    """Load and validate an experiment without downloading/checking weights.

    Args:
        path: Experiment YAML path.
        overrides: Explicit fields to replace, e.g. output_dir or batch.

    Returns:
        Validated immutable configuration.

    Raises:
        ValueError: Unknown fields or invalid modes/settings.
    """
    source = Path(path).resolve()
    data = _read(source)
    data.update(overrides)
    allowed = set(ExperimentConfig.__dataclass_fields__) - {"source"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{source}: unknown settings {sorted(unknown)}.")
    required = ("output_dir", "data_root", "pretrained", "anchor_dir", "texts_path")
    if any(name not in data for name in required):
        raise ValueError(f"{source}: explicitly provide {required}.")
    for name in required:
        data[name] = _path(str(data[name]))
    data["source"] = source
    if "input_size" in data:
        data["input_size"] = tuple(data["input_size"])
    config = ExperimentConfig(**data)
    if config.variant not in {"concat", "condition"}:
        raise ValueError("variant must be concat or condition.")
    if config.alignment not in {"off", "visual", "geometry"}:
        raise ValueError("alignment must be off, visual, or geometry.")
    if config.condition_mode not in {"real", "constant"}:
        raise ValueError("condition_mode must be real or constant.")
    if config.variant == "concat" and config.alignment != "off":
        raise ValueError("The simple concat baseline must explicitly disable alignment.")
    if config.validation_split not in {"train", "test"}:
        raise ValueError("This ATR-UMOD prototype supports train/test folders only.")
    optimizer_names = {"sgd": "SGD", "adamw": "AdamW"}
    if not isinstance(config.optimizer, str) or config.optimizer.lower() not in optimizer_names:
        raise ValueError("optimizer must be SGD or AdamW.")
    object.__setattr__(config, "optimizer", optimizer_names[config.optimizer.lower()])
    for name in ("enable_film", "dynamic_fusion", "score_condition", "amp", "nesterov", "augment"):
        if type(getattr(config, name)) is not bool:
            raise ValueError(f"{name} must be a boolean.")
    for name in ("epochs", "batch", "val_batch", "max_det"):
        if type(getattr(config, name)) is not int or getattr(config, name) <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    for name in ("workers", "seed", "warmup_iterations", "aug_fill_value"):
        if type(getattr(config, name)) is not int or getattr(config, name) < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    config.augmentation()
    for name in ("lr_new", "lr_pretrained", "weight_decay", "clip_norm", "max_shift_px", "min_gt_side_px"):
        value = getattr(config, name)
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    for name in ("lr_final_ratio", "momentum", "warmup_momentum"):
        value = getattr(config, name)
        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must lie in (0,1).")
    if len(config.input_size) != 2 or any(type(size) is not int or size < 32 or size % 32 for size in config.input_size):
        raise ValueError("input_size must contain two positive multiples of 32.")
    if not 0 < config.conf < 1 or not 0 < config.nms_iou < 1:
        raise ValueError("conf and nms_iou must lie in (0,1).")
    return config
