"""
engine/backbone/dual_hgnetv2_add.py

基准模型 (Baseline): 双流 HGNetv2 逐元素直接相加主干网络 (Direct Element-wise Addition).
在 S3, S4, S5 阶段直接执行双模态特征张量相加，用于严谨验证相加与拼接的数学秩差异.
"""

from typing import List
import torch
import torch.nn as nn

from .hgnetv2 import HGNetv2
from ..core import register


@register()
class DualHGNetv2_Add(nn.Module):
    r"""双流 HGNetv2 特征直接逐元素相加基准主干.

    输入: [B, 6, H, W] (0:3 为 IR, 3:6 为 RGB)
    输出: List[torch.Tensor] 对应 [S3, S4, S5], 融合方式为直接相加 (f_ir + f_rgb)
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
        # 1. 红外独立主干
        self.backbone_ir = HGNetv2(
            name=name,
            return_idx=return_idx,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            use_lab=use_lab,
            pretrained=pretrained,
            local_model_dir=local_model_dir,
        )
        # 2. 可见光独立主干
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
        # 兼容算力分析工具传入的 3 通道 Dummy Tensor
        if x.shape[1] == 3:
            x = torch.cat([x, x], dim=1)
        assert x.shape[1] == 6, f"预期输入为 6 通道张量，当前为: {x.shape[1]}"

        x_ir = x[:, :3, :, :]
        x_rgb = x[:, 3:, :, :]

        feats_ir: List[torch.Tensor] = self.backbone_ir(x_ir)
        feats_rgb: List[torch.Tensor] = self.backbone_rgb(x_rgb)

        # 逐阶段执行纯粹的代数相加
        fused_features: List[torch.Tensor] = [
            f_ir + f_rgb for f_ir, f_rgb in zip(feats_ir, feats_rgb)
        ]
        return fused_features