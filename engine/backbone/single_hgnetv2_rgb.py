"""
engine/backbone/single_hgnetv2_rgb.py

基准模型 (Baseline): 单模态可见光独立骨干网络 (Unimodal Infrared).
仅接收与处理可见光特征，彻底切除热可见光模态输入，作为全天候可见光性能的坚实下界基准.
"""

from typing import List
import torch
import torch.nn as nn

from .hgnetv2 import HGNetv2
from ..core import register


@register()
class SingleHGNetv2_RGB(nn.Module):
    r"""单模态可见光 HGNetv2-B0 骨干网络.

    输入: 兼容 [B, 6, H, W] 或 [B, 3, H, W].
          若为 6 通道成对张量，强制只取后 3 通道 (RGB) 执行前向传播.
    输出: List[torch.Tensor] 对应可见光 [S3, S4, S5] 多尺度特征.
    """

    def __init__(
        self,
        name: str = "B0",
        return_idx: List[int] = [1, 2, 3],
        freeze_at: int = -1,
        freeze_norm: bool = False,
        use_lab: bool = True,
        pretrained: bool = True,
        local_model_dir: str = "./pretrain/hgnetv2/",
    ) -> None:
        super().__init__()
        # 仅实例化单个可见光主干
        self.backbone_rgb = HGNetv2(
            name=name,
            return_idx=return_idx,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            use_lab=use_lab,
            pretrained=pretrained,
            local_model_dir=local_model_dir,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # 严格防御性切片：只提取可见光 3 通道
        if x.shape[1] == 6:
            x_rgb = x[:, 3:, :, :]
        elif x.shape[1] == 3:
            x_rgb = x
        else:
            raise ValueError(f"输入通道数必须为 3 或 6，当前为: {x.shape[1]}")

        return self.backbone_rgb(x_rgb)