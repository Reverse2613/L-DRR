"""Module-aware AdamW grouping for transferred weights and newly added layers."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn

from .config import ExperimentConfig


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.AdamW:
    """Exclude bias/norm/scalar parameters from decay and separate learning rates.

    Args:
        model: Paired model with explicit loaded_parameter_names.
        config: Pretrained/new learning rates and matrix weight decay.

    Returns:
        AdamW optimizer covering every trainable parameter exactly once.
        Each group records role and param_names for reproducible auditing.
    """
    loaded = set(model.loaded_parameter_names)
    normalizations = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
                      nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    buckets: Dict[tuple[str, str], Dict[str, Any]] = {}
    seen: set[int] = set()
    for prefix, module in model.named_modules():
        for local_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            name = f"{prefix}.{local_name}" if prefix else local_name
            if id(parameter) in seen:
                raise ValueError(f"Duplicate optimizer parameter: {name}")
            seen.add(id(parameter))
            role = "pretrained" if name in loaded else "new"
            kind = "no_decay" if (
                local_name == "bias" or parameter.ndim <= 1
                or isinstance(module, normalizations + (nn.Embedding,))
            ) else "matrix"
            key = role, kind
            if key not in buckets:
                buckets[key] = {
                    "params": [], "param_names": [], "role": role, "kind": kind,
                    "lr": config.lr_pretrained if role == "pretrained" else config.lr_new,
                    "weight_decay": 0.0 if kind == "no_decay" else config.weight_decay,
                }
            buckets[key]["params"].append(parameter)
            buckets[key]["param_names"].append(name)
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != expected:
        raise ValueError("Optimizer coverage differs from trainable model parameters.")
    groups: List[Dict[str, Any]] = [buckets[key] for key in sorted(buckets)]
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))
