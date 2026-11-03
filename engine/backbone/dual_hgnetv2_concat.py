"""
engine/backbone/dual_hgnetv2_concat.py

双流 HGNetv2 多模态特征拼接骨干网络 (Naive Concat Baseline).
支持双路独立 ImageNet 预训练权重加载，并在多尺度特征层 (S3, S4, S5) 执行通道级直接拼接与线性降维.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn

from .hgnetv2 import HGNetv2
from ..core import register


class ConvBNAct(nn.Module):
    r"""标准的 1x1 卷积 + 批归一化 + 激活函数融合块."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


@register()
class DualHGNetv2_Concat(nn.Module):
    r"""双流 HGNetv2 特征级直接拼接主干网络.

    输入:
        x (torch.Tensor): 形状为 [Batch, 6, Height, Width] 的堆叠张量.
                         其中 x[:, :3, :, :] 为热红外图像 (TIR),
                         x[:, 3:, :, :] 为可见光彩色图像 (RGB).

    输出:
        List[torch.Tensor]: 包含 3 个多尺度融合特征图的列表:
            - F3: [Batch, 256, H/8, W/8]
            - F4: [Batch, 512, H/16, W/16]
            - F5: [Batch, 1024, H/32, W/32]
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
        """初始化双流主干网络.

        Args:
            name (str): HGNetv2 模型规格 ('B0', 'B1', 'B2' 等), 默认 'B0'.
            return_idx (List[int]): 需返回的特征阶段索引, 默认为 [1, 2, 3] 对应 S3, S4, S5.
            freeze_at (int): 冻结主干阶段层数, -1 表示不冻结.
            freeze_norm (bool): 是否冻结 BatchNorm 统计量.
            use_lab (bool): 是否使用可学习仿射变换 (Learnable Affine Block).
            pretrained (bool): 是否加载官方 ImageNet 预训练权重.
            local_model_dir (str): 预训练权重本地缓存目录.
        """
        super().__init__()

        # 1. 实例化热红外 (TIR) 独立主干
        self.backbone_ir = HGNetv2(
            name=name,
            return_idx=return_idx,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            use_lab=use_lab,
            pretrained=pretrained,
            local_model_dir=local_model_dir,
        )

        # 2. 实例化可见光 (RGB) 独立主干
        self.backbone_rgb = HGNetv2(
            name=name,
            return_idx=return_idx,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            use_lab=use_lab,
            pretrained=pretrained,
            local_model_dir=local_model_dir,
        )

        # 3. 针对 B0 规格的各阶段输出通道定义: [S3=256, S4=512, S5=1024]
        self.stage_channels: List[int] = [256, 512, 1024]

        # 4. 构建三组 1x1 卷积融合降维层: 将 2*C 降维回原版 C
        # S3: 256*2 -> 256 | S4: 512*2 -> 512 | S5: 1024*2 -> 1024
        self.fusion_layers = nn.ModuleList([
            ConvBNAct(in_channels=c * 2, out_channels=c)
            for c in self.stage_channels
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""前向传播逻辑.

        Args:
            x (torch.Tensor): 输入张量, 正常训练时形状为 [B, 6, H, W].
                             若检测到形状为 [B, 3, H, W] (如 calflops 算力分析工具传入),
                             将自动沿通道复制以适配双流骨干算力统计.
        """
        # ======================================================================
        # 兼容性防御：自动适配外部 profiler 工具 (如 calflops) 默认传入的 3 通道 Dummy Tensor
        # ======================================================================
        if x.shape[1] == 3:
            # 自动将 [B, 3, H, W] 广播复制为 [B, 6, H, W]，确保双流主干均能执行前向以计算真实 FLOPs
            x = torch.cat([x, x], dim=1)
        assert x.shape[1] == 6, f"输入张量通道数必须为 6 (3通道IR + 3通道RGB)，当前为: {x.shape[1]}"

        # 物理通道切分
        x_ir = x[:, :3, :, :]   # [B, 3, H, W] 热红外图像
        x_rgb = x[:, 3:, :, :]  # [B, 3, H, W] 可见光图像

        # 双流并行提取多尺度特征
        feats_ir: List[torch.Tensor] = self.backbone_ir(x_ir)    # [S3_ir, S4_ir, S5_ir]
        feats_rgb: List[torch.Tensor] = self.backbone_rgb(x_rgb) # [S3_rgb, S4_rgb, S5_rgb]

        # 在各尺度上执行通道拼接与线性降维
        fused_features: List[torch.Tensor] = []
        for i, (f_ir, f_rgb) in enumerate(zip(feats_ir, feats_rgb)):
            # 1. 沿通道维度拼接: [B, C, H_i, W_i] + [B, C, H_i, W_i] -> [B, 2*C, H_i, W_i]
            concat_f = torch.cat([f_ir, f_rgb], dim=1)
            # 2. 1x1 卷积降维回: [B, C, H_i, W_i]
            fused_f = self.fusion_layers[i](concat_f)
            fused_features.append(fused_f)

        return fused_features