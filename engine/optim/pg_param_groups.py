"""模块一专用AdamW参数分组：按参数所属模块和角色分类。

归一化、bias、Embedding和标量仿射参数不衰减；两个HGNetv2支路的
仿射参数使用较小学习率。新融合模块位于backbone内部，但不属于预训练支路。
"""

import math
from typing import Any, Dict, List, Tuple

import torch.nn as nn


def build_pg_param_groups(
    model: nn.Module, lr: float, weight_decay: float, backbone_affine_lr: float,
) -> List[Dict[str, Any]]:
    r"""为模块一生成互斥、完备的四个参数组，并保留逐参数名称。

    Args:
        model: 已组装的RTv4_PG，具有backbone.backbone_rgb/backbone_ir。
        lr: 主体及新模块基础学习率，推荐1e-4。
        weight_decay: Conv/Linear等主体矩阵权重的衰减，推荐1e-4。
        backbone_affine_lr: 两个预训练支路Norm/bias/标量参数学习率，推荐1e-5。

    Returns:
        可直接传给AdamW的参数组；group_name和param_names用于核验与保存。

    Raises:
        ValueError: 模型结构、数值、参数归属或覆盖检查不符合要求。
    """
    if not all(math.isfinite(value) for value in (lr, weight_decay, backbone_affine_lr)):
        raise ValueError("优化器学习率与衰减必须是有限数值。")
    if lr <= 0 or backbone_affine_lr <= 0 or weight_decay < 0:
        raise ValueError("学习率必须为正，weight_decay不能为负。")
    detector = getattr(model, "module", model)
    if not getattr(detector, "requires_conditions", False):
        raise ValueError("pg_module1参数分组只适用于显式条件PG模型。")
    backbone = getattr(detector, "backbone", None)
    branches = [getattr(backbone, name, None) for name in ("backbone_rgb", "backbone_ir")]
    if not all(isinstance(branch, nn.Module) for branch in branches):
        raise ValueError("PG模型必须具有独立的backbone_rgb和backbone_ir。")
    branch_ids = [{id(parameter) for parameter in branch.parameters()} for branch in branches]
    if branch_ids[0] & branch_ids[1]:
        raise ValueError("两个预训练主干支路不能共享参数对象。")
    pretrained_ids = branch_ids[0] | branch_ids[1]

    # 以直接所属模块判断Norm/Embedding，而不是猜名称包含norm或embed。
    norm_types: Tuple[type, ...] = (nn.modules.batchnorm._NormBase, nn.LayerNorm, nn.GroupNorm)
    if hasattr(nn, "RMSNorm"):
        norm_types += (nn.RMSNorm,)
    owners: Dict[int, Tuple[nn.Module, str]] = {}
    for _, module in detector.named_modules():
        for local_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            if id(parameter) in owners:
                raise ValueError("同一可训练参数被多个模块持有，不能静默猜测其衰减角色。")
            owners[id(parameter)] = (module, local_name)

    definitions = (
        ("pretrained_weights", lr, weight_decay),
        ("pretrained_affine", backbone_affine_lr, 0.0),
        ("new_weights", lr, weight_decay),
        ("new_no_decay", lr, 0.0),
    )
    groups: List[Dict[str, Any]] = [
        {"group_name": name, "lr": group_lr, "initial_lr": group_lr,
         "weight_decay": decay, "params": [], "param_names": []}
        for name, group_lr, decay in definitions
    ]
    expected = {id(parameter) for parameter in detector.parameters() if parameter.requires_grad}
    visited: List[int] = []
    for name, parameter in detector.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) not in owners:
            raise ValueError(f"参数{name}未找到直接所属模块。")
        module, local_name = owners[id(parameter)]
        no_decay = (isinstance(module, norm_types + (nn.Embedding, nn.EmbeddingBag))
                    or local_name == "bias" or parameter.ndim <= 1)
        index = (0 if id(parameter) in pretrained_ids else 2) + int(no_decay)
        groups[index]["params"].append(parameter)
        groups[index]["param_names"].append(name)
        visited.append(id(parameter))
    if len(visited) != len(set(visited)) or set(visited) != expected:
        raise ValueError("参数分组必须使每个可训练参数对象恰好出现一次。")
    if set(owners) != expected:
        raise ValueError("存在未覆盖的参数所属模块。")
    result = [group for group in groups if group["params"]]
    for group in result:
        print(f"[PG optimizer] {group['group_name']}: {len(group['params'])} tensors, "
              f"{sum(parameter.numel() for parameter in group['params']):,} parameters, "
              f"lr={group['lr']:.1e}, weight_decay={group['weight_decay']:.1e}")
    return result
