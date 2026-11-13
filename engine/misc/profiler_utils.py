"""
engine/misc/profiler_utils.py
"""

import copy
from calflops import calculate_flops
from typing import Any, Dict, Mapping, Set, Tuple

import torch
import torch.nn as nn


class _ConditionedProfileModel(nn.Module):
    r"""仅供calflops的图像适配器，显式携带已声明的合成六条件。"""

    def __init__(self, model: nn.Module, conditions: Mapping[str, Any]) -> None:
        r"""保存部署模型与同设备的profiler合成条件。

        Args:
            model: 已复制并deploy的PG模型。
            conditions: 仅用于profiler的条件batch；支持F组完整缓存文本。
        """
        super().__init__()
        self.model = model
        self.conditions = conditions

    def forward(self, images: torch.Tensor) -> Dict[str, Any]:
        r"""让calflops传单个图像张量，同时调用严格的条件接口。

        Args:
            images: profiler图像输入[1,6,H,W]。

        Returns:
            原PG模型的检测输出。
        """
        return self.model(images, conditions=self.conditions)

def stats(
    cfg: Any,
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 640),
) -> Tuple[int, Set[str]]:
    r"""按实际分辨率和输入契约统计部署态FLOPs。

    Args:
        cfg: 已实例化模型的运行配置。
        input_shape: 默认图像规格[B,C,H,W]；配置分辨率优先。

    Returns:
        总参数量和供现有solver打印的统计字符串集合。
    """

    # 1. 深度穿透读取真实配置尺寸 (优先查 yaml_cfg 字典)
    eval_size = None
    if hasattr(cfg, "yaml_cfg") and isinstance(cfg.yaml_cfg, dict) and "eval_spatial_size" in cfg.yaml_cfg:
        eval_size = cfg.yaml_cfg["eval_spatial_size"]
    elif hasattr(cfg, "eval_spatial_size") and cfg.eval_spatial_size is not None:
        eval_size = cfg.eval_spatial_size

    # 2. 显式条件模型使用真实6通道契约，不能复制一张图冒充两模态。
    channels = getattr(cfg.model, "profiling_channels", 3)
    if eval_size is not None:
        input_shape = (1, channels, int(eval_size[0]), int(eval_size[1]))
    else:
        base_size = cfg.train_dataloader.collate_fn.base_size
        input_shape = (1, channels, base_size, base_size)

    # 3. 统计部署态 FLOPs
    model_for_info = copy.deepcopy(cfg.model).deploy()

    if getattr(model_for_info, "requires_conditions", False):
        device = next(model_for_info.parameters()).device
        # 仅profiler的合成条件：120m、30度、正午、晴天、正常光照、场景0。
        # 正常训练/推理入口仍要求真实条件，这里不为模型设置默认条件。
        conditions = {
            "height": torch.tensor([[(120.0 - 80.0) / 220.0]], device=device),
            "angle": torch.tensor([[30.0 / 75.0]], device=device),
            "time": torch.tensor([[0.0, -1.0]], device=device),
            "weather": torch.tensor([0], dtype=torch.int64, device=device),
            "illumination": torch.tensor([1], dtype=torch.int64, device=device),
            "location": torch.tensor([0], dtype=torch.int64, device=device),
        }
        prior = getattr(getattr(model_for_info, "backbone", None), "condition_encoder", None)
        if callable(getattr(prior, "profiling_conditions", None)):
            # F组不能使用虚构日期查表，显式取一条已缓存文本；不影响旧模型。
            conditions = prior.profiling_conditions()
        print("Profiler: 6通道图像对＋显式统计条件（仅用于计算量统计）。")
        # calflops对kwargs各值调用.to，且自动图像使用未初始化new_empty。
        # 因此用适配器携带条件，并显式传有限图像，避免严格数值检查误报。
        flops, macs, _ = calculate_flops(
            model=_ConditionedProfileModel(model_for_info, conditions),
            args=[torch.zeros(input_shape, device=device)],
            output_as_string=True, output_precision=4, print_detailed=False,
        )
    else:
        flops, macs, _ = calculate_flops(
            model=model_for_info, input_shape=input_shape, output_as_string=True,
            output_precision=4, print_detailed=False,
        )
    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info

    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" % (flops, macs, params)}
