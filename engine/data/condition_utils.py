"""显式六条件的batch组装：只传图像级属性，不传目标框或类别标签。"""

from typing import Any, Dict, Mapping, Sequence, Union

import torch
import torch.nn as nn


CONDITION_SHAPES = {
    "height": (1,), "angle": (1,), "time": (2,),
    "weather": (), "illumination": (), "location": (),
}
CATEGORY_COUNTS = {"weather": 10, "illumination": 7, "location": 11}


def requires_explicit_conditions(model: nn.Module) -> bool:
    r"""检查模型是否需要显式条件，支持DDP及torch.compile包装。

    Args:
        model: 原模型或带module/_orig_mod的包装模型。

    Returns:
        是否声明requires_conditions=True；旧图像模型返回False。
    """
    seen = set()
    current = model
    while id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "requires_conditions", False):
            return True
        wrapped = getattr(current, "module", None)
        if not isinstance(wrapped, nn.Module):
            wrapped = getattr(current, "_orig_mod", None)
        if not isinstance(wrapped, nn.Module):
            return False
        current = wrapped
    raise ValueError("模型包装出现循环，无法确定显式条件接口。")


def stack_batch_conditions(
    targets: Sequence[Mapping[str, Any]], device: Union[str, torch.device]
) -> Dict[str, torch.Tensor]:
    r"""从各样本元数据严格提取白名单六条件并堆成batch。

    Args:
        targets: B个样本字典；每项包含conditions，连续量分别为[1]/[1]/[2]
            float32，三个原始类别编号为int64标量。其他GT字段不会被提取。
        device: 返回张量所在设备；搬运只改变设备，不改变dtype或条件数值。

    Returns:
        height/angle各[B,1]、time[B,2] float32，三个类别各[B] int64。

    Raises:
        ValueError: batch为空，字段缺失、形状/类型错误或条件越界。
    """
    if not targets:
        raise ValueError("显式条件batch不能为空。")
    result: Dict[str, torch.Tensor] = {}
    for field, expected_shape in CONDITION_SHAPES.items():
        expected_dtype = torch.int64 if field in CATEGORY_COUNTS else torch.float32
        values = []
        for index, target in enumerate(targets):
            conditions = target.get("conditions")
            if not isinstance(conditions, Mapping):
                raise ValueError(f"batch样本{index}缺少conditions字典。")
            value = conditions.get(field)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"batch样本{index}缺少张量条件{field}。")
            if tuple(value.shape) != expected_shape or value.dtype != expected_dtype:
                raise ValueError(f"batch样本{index}的{field}需要{expected_shape}/{expected_dtype}。")
            values.append(value.to(device))
        stacked = torch.stack(values, dim=0)
        if not torch.isfinite(stacked).all():
            raise ValueError(f"batch条件{field}含NaN/Inf。")
        if field in ("height", "angle") and ((stacked < 0) | (stacked > 1)).any():
            raise ValueError(f"batch条件{field}应已规范化到[0,1]，不自动裁剪。")
        if field in CATEGORY_COUNTS and ((stacked < 0) | (stacked >= CATEGORY_COUNTS[field])).any():
            raise ValueError(f"batch条件{field}编号应在0到{CATEGORY_COUNTS[field] - 1}。")
        if field == "time" and not torch.allclose(
            stacked.norm(dim=1), torch.ones(len(targets), device=stacked.device), atol=1e-5, rtol=0
        ):
            raise ValueError("batch条件time需要已有的一天周期[sin,cos]。")
        result[field] = stacked
    return result
