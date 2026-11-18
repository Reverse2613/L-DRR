"""Two pretrained YOLO11s backbones, three fusion blocks, one native OBB neck/head."""

from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch import nn

from .conditions import FrozenEnvironmentPrior, geometry_descriptor
from .config import ExperimentConfig
from .data import CLASSES
from .fusion import ConcatFusion, ConditionFusion
from .runtime import require_framework

require_framework()
from ultralytics.nn.modules import OBB
from ultralytics.nn.tasks import BaseModel, OBBModel
from ultralytics.utils import LOGGER


class PairedYOLO11sOBB(BaseModel):
    """Port input-role separation while retaining the native detector graph.

    The pretrained native backbone is the RGB branch. ir_backbone is a separate
    copy of layers 0–10. Both streams finish extraction before fusion at layers
    4/6/10, so fused features do not accidentally feed either backbone.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Build the paired graph, with new modules isolated from common RNG.

        Args:
            config: Native 512x640 experiment and ablation settings.
        """
        super().__init__()
        native = OBBModel("yolo11s-obb.yaml", ch=3, nc=len(CLASSES), verbose=False)
        if native.yaml.get("scale") != "s" or len(native.model) != 24 or not isinstance(native.model[-1], OBB):
            raise ValueError("Expected the pinned YOLO11s-OBB P3/P4/P5 architecture.")
        self.model, self.save = native.model, native.save
        self.yaml = copy.deepcopy(native.yaml)
        self.yaml["ch"] = 6
        self.stride = self.model[-1].stride
        self.nc, self.names, self.task = len(CLASSES), dict(enumerate(CLASSES)), "obb"
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        self.input_size, self.variant = config.input_size, config.variant
        self.alignment = config.alignment
        self.architecture_spec = config.architecture()
        self.experiment_spec = config.as_dict()
        self.min_gt_side_px = config.min_gt_side_px
        self.ir_backbone = copy.deepcopy(nn.Sequential(*list(self.model.children())[:11]))
        self.fusion_indices = (4, 6, 10)
        channels = tuple(self.model[index].cv2.conv.out_channels for index in self.fusion_indices)
        if channels != (256, 256, 512):
            raise ValueError(f"Unexpected YOLO11s backbone pyramid: {channels}.")
        self.feature_channels = channels
        self.loaded_parameter_names: Tuple[str, ...] = ()
        self.pretrained_report: Dict[str, Any] = {}
        self.data_audits: Dict[str, Any] = {}
        # Controls retain the same initialization of common neck/head layers.
        with torch.random.fork_rng(devices=[]):
            if self.variant == "condition":
                self.environment_prior: Optional[FrozenEnvironmentPrior] = FrozenEnvironmentPrior(
                    config.anchor_dir, config.texts_path, config.condition_mode
                )
                self.fusions = nn.ModuleList([
                    ConditionFusion(width, stride, config.max_shift_px, config.alignment,
                                    config.enable_film, config.dynamic_fusion, config.score_condition)
                    for width, stride in zip(channels, (8, 16, 32))
                ])
            else:
                self.environment_prior = None
                self.fusions = nn.ModuleList([ConcatFusion(width) for width in channels])

    @staticmethod
    def _pyramid(branch: Iterable[nn.Module], images: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Extract independent native backbone outputs.

        Args:
            branch: Eleven native backbone layers.
            images: One modality [B,3,H,W].

        Returns:
            Features at indices 4/6/10, strides 8/16/32.
        """
        result: Dict[int, torch.Tensor] = {}
        features = images
        for index, layer in enumerate(branch):
            if layer.f != -1:
                raise ValueError("Unexpected backbone skip graph in the pinned model.")
            features = layer(features)
            if index in (4, 6, 10):
                result[index] = features
        return result

    def predict(
        self, x: torch.Tensor, profile: bool = False, visualize: bool = False,
        augment: bool = False, embed: Optional[List[int]] = None,
        conditions: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Any:
        """Predict with image-level conditions supplied explicitly.

        Args:
            x: Float images [B,6,H,W], IR then RGB.
            profile: Unsupported native profiler (metadata would be absent).
            visualize: Unsupported native single-modality visualizer.
            augment: Must be False; no unsynchronized test-time augmentation.
            embed: Must be None for the OBB experiment.
            conditions: Environment IDs [B] and geometry [B,1].

        Returns:
            Unchanged native OBB training or inference output structure.
        """
        if profile or visualize or augment or embed is not None:
            raise ValueError("Use the paired check/export tools; native TTA/profiling lacks the metadata contract.")
        if (x.ndim != 4 or x.shape[1] != 6 or x.shape[-2:] != self.input_size or len(x) == 0):
            raise ValueError(f"Expected nonempty [B,6,{self.input_size[0]},{self.input_size[1]}], IR then RGB.")
        rgb = self._pyramid(list(self.model.children())[:11], x[:, 3:])
        ir = self._pyramid(self.ir_backbone, x[:, :3])
        fused: Dict[int, torch.Tensor] = {}
        if self.variant == "condition":
            if conditions is None:
                raise ValueError("Condition fusion requires explicit image-level metadata.")
            condition = self.environment_prior(conditions)
            if len(condition) != len(x):
                raise ValueError("Condition/image batch sizes differ.")
            if self.alignment == "off":
                geometry = torch.zeros(len(x), 5, device=x.device)
                radius = torch.ones(len(x), 1, device=x.device)
            else:
                geometry, radius = geometry_descriptor(conditions, len(x), x.device, self.alignment)
            for index, block in zip(self.fusion_indices, self.fusions):
                fused[index] = block(rgb[index], ir[index], condition, geometry, radius)
        else:
            for index, block in zip(self.fusion_indices, self.fusions):
                fused[index] = block(rgb[index], ir[index])

        # Preserve the original neck's skip indices and native OBB head.
        saved: List[Any] = [None] * 11
        for index, feature in fused.items():
            saved[index] = feature
        features = fused[10]
        for layer in list(self.model.children())[11:]:
            if layer.f != -1:
                features = saved[layer.f] if isinstance(layer.f, int) else [
                    features if index == -1 else saved[index] for index in layer.f
                ]
            features = layer(features)
            saved.append(features if layer.i in self.save else None)
        return features

    def loss(self, batch: Dict[str, Any], preds: Any = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute only the native detection objective with explicit metadata.

        Args:
            batch: Paired images, conditions and supervised OBB targets.
            preds: Optional native prediction tuple, e.g. validation output.

        Returns:
            Scalar loss and detached box/cls/DFL [3] vector.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.predict(batch["img"], conditions=batch.get("conditions"))
        result = self.criterion(preds, batch)
        if not torch.isfinite(result[0]):
            raise FloatingPointError("Nonfinite detection loss; no silent replacement.")
        return result

    def init_criterion(self) -> Any:
        """Create the native-objective adapter with correct rectangular units.

        Returns:
            ATRUMODOBBLoss, adding no condition/gate auxiliary loss.
        """
        from .loss import ATRUMODOBBLoss

        return ATRUMODOBBLoss(self, self.min_gt_side_px)

    def _apply(self, fn: Any) -> PairedYOLO11sOBB:
        """Move native detection caches and preserve the environment cache.

        Args:
            fn: Standard PyTorch tensor conversion.

        Returns:
            This model on the requested device/dtype.
        """
        super()._apply(fn)
        self.stride = self.model[-1].stride
        return self

    def info(
        self, detailed: bool = False, verbose: bool = True, imgsz: int = 640,
    ) -> Tuple[int, int, int, None]:
        """Report parameter counts without a fake metadata-free FLOP estimate.

        Args:
            detailed: Reserved native argument.
            verbose: Print the parameter summary.
            imgsz: Reserved native nominal maximum dimension.

        Returns:
            Module, parameter and trainable counts; unmeasured FLOPs as None.
        """
        counts = (len(list(self.modules())), sum(p.numel() for p in self.parameters()),
                  sum(p.numel() for p in self.parameters() if p.requires_grad), None)
        if verbose:
            LOGGER.info(f"Paired YOLO11s-OBB: {counts[1]:,} parameters; FLOPs not measured.")
        return counts

    def load(self, weights: nn.Module | Dict[str, Any], verbose: bool = True) -> PairedYOLO11sOBB:
        """Copy pretrained coverage explicitly; reject unexplained missing keys.

        Args:
            weights: Local native YOLO11s-OBB module, or this paired model.
            verbose: Log loaded tensors and new parameters.

        Returns:
            This model with both backbones and the shared compatible neck/head
            initialized from the native checkpoint. Only eleven-class output
            convolutions and new fusion components may remain newly initialized.
        """
        source = weights.get("model") if isinstance(weights, dict) else weights
        if isinstance(source, PairedYOLO11sOBB):
            if source.architecture_spec != self.architecture_spec:
                raise ValueError("Paired checkpoint architecture/ablation does not match this configuration.")
            if self.environment_prior is not None:
                if source.environment_prior.cache_sha256 != self.environment_prior.cache_sha256:
                    raise ValueError("Paired checkpoint and configured CLIP cache have different origins.")
                torch.testing.assert_close(source.environment_prior.joint_anchor.float().cpu(),
                                           self.environment_prior.joint_anchor.cpu(), rtol=0, atol=0)
            self.load_state_dict(source.float().state_dict(), strict=True)
            self.loaded_parameter_names = source.loaded_parameter_names
            self.pretrained_report = copy.deepcopy(source.pretrained_report)
            self.data_audits = copy.deepcopy(source.data_audits)
            return self
        if (not isinstance(source, OBBModel) or source.yaml.get("scale") != "s"
                or len(source.model) != 24 or not isinstance(source.model[-1], OBB)):
            raise ValueError("Provide the official local yolo11s-obb.pt, not an HBB/different-size checkpoint.")
        source_state = source.float().state_dict()
        target = self.state_dict()
        transferred: Dict[str, torch.Tensor] = {}
        skipped: List[str] = []
        parameters = dict(self.named_parameters())
        for name, tensor in target.items():
            if name.startswith("fusions."):
                skipped.append(name)
                continue
            source_name = "model." + name[len("ir_backbone."):] if name.startswith("ir_backbone.") else name
            value = source_state.get(source_name)
            if value is not None and value.shape == tensor.shape:
                transferred[name] = value
            elif re.fullmatch(r"model\.23\.cv3\.[0-2]\.2\.(weight|bias)", name):
                skipped.append(name)  # Different detection classes, explicitly recorded.
            else:
                raise ValueError(f"Unexplained pretrained mismatch: {name} <- {source_name}.")
        self.load_state_dict(transferred, strict=False)
        self.loaded_parameter_names = tuple(sorted(set(transferred) & set(parameters)))
        self.pretrained_report = {
            "source": "native YOLO11s-OBB", "source_classes": source.model[-1].nc,
            "target_classes": self.nc, "transferred_tensor_count": len(transferred),
            "loaded_parameter_numel": sum(parameters[name].numel() for name in self.loaded_parameter_names),
            "new_or_class_output_tensors": skipped,
            "both_backbones_loaded": True, "shared_compatible_neck_head_loaded": True,
        }
        if verbose:
            LOGGER.info(f"Pretrained transfer: {len(transferred)} tensors; {len(skipped)} explicitly new tensors.")
        return self


def load_local_checkpoint(path: str | Path, config: Optional[ExperimentConfig] = None) -> PairedYOLO11sOBB:
    """Read a local paired checkpoint without any download fallback.

    Args:
        path: Native best.pt/last.pt produced by the paired trainer.
        config: Optional expected architecture, for strict evaluation/resume.

    Returns:
        FP32 CPU paired model with its frozen cache retained.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Local checkpoint required: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(model, PairedYOLO11sOBB):
        raise ValueError(f"{path}: not a paired YOLO11s-OBB experiment checkpoint.")
    model = model.float()
    if isinstance(checkpoint, dict) and "train_args" in checkpoint:
        saved_args = checkpoint["train_args"]
    elif isinstance(model.args, Mapping):
        saved_args = dict(model.args)
    else:
        saved_args = vars(model.args)
    if not isinstance(saved_args, Mapping) or any(key not in saved_args for key in ("box", "cls", "dfl")):
        raise ValueError("Checkpoint lacks its native detection-loss argument record.")
    model.args = SimpleNamespace(**saved_args)
    if config is not None and model.architecture_spec != config.architecture():
        raise ValueError("Evaluation configuration differs from the checkpoint's architecture.")
    if model.names != dict(enumerate(CLASSES)):
        raise ValueError("Checkpoint class mapping is not the fixed ATR-UMOD IR task.")
    if model.environment_prior is not None and model.environment_prior.joint_anchor.dtype != torch.float32:
        raise ValueError("Checkpoint CLIP vectors are not raw FP32.")
    model.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return model
