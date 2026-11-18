"""Module-aware optimizer grouping for transferred and newly added weights."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch import nn

from .config import ExperimentConfig


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    """Build SGD or AdamW groups with explicit learning-rate and decay rules.

    The four groups are the Cartesian product of parameter provenance
    (``pretrained`` versus ``new``) and decay eligibility (``matrix`` versus
    ``no_decay``). The provenance split lets newly initialized fusion/class
    layers learn faster, while the decay split keeps bias, normalization,
    embedding, and scalar parameters out of weight decay.

    Args:
        model: Paired model exposing ``loaded_parameter_names``.
        config: Validated optimizer, learning-rate, momentum, and decay settings.

    Returns:
        Optimizer covering every trainable parameter exactly once. Each group
        also carries ``role`` and ``param_names`` for the JSON audit log.

    Raises:
        ValueError: A parameter is duplicated/missing, or the optimizer is
            unsupported.
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

    if config.optimizer == "SGD":
        # Nesterov SGD is YOLO's native long-schedule choice. Momentum is
        # ramped by the native trainer from warmup_momentum to momentum.
        return torch.optim.SGD(
            groups, momentum=config.momentum, dampening=0.0, nesterov=config.nesterov
        )
    if config.optimizer == "AdamW":
        # Retained only for reproducing historical YOLO runs. AdamW ignores
        # ``momentum`` and uses beta_1/beta_2 below.
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")
