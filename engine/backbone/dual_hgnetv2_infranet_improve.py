"""
engine/backbone/dual_hgnetv2_infranet_improve.py

正统级联深度 + 全链路预训练增强版 InfraNet 双流主干网络 (DualHGNetv2_InfraNet).
论文: InfraNet: Quality-Aware RGB Guidance for Efficient Infrared Object Detection

核心特性:
    1. 真实级联递进拓扑: A3 -> Stage 2 -> A4 -> Stage 3 -> A5, 纵向深度层层累积;
    2. 全链路预训练武装: 级联过程中的初始提取与层间下采样全部复用 ImageNet 预训练权重的 HG_Stage;
    3. 纯净 QualGate: 剔除冗余大卷积, 还原公式 (4) 纯净加权, 参数量仅 ~14M, 极速收敛.
"""

from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn

from .hgnetv2 import HGNetv2
from .qual_gate import QualGate
from ..core import register


@register()
class DualHGNetv2_InfraNet_Improve(nn.Module):
    r"""InfraNet 正统级联非对称双流主干网络."""

    def __init__(
        self,
        name: str = "B0",
        return_idx: List[int] = [1, 2, 3],
        freeze_at: int = -1,
        freeze_norm: bool = False,
        use_lab: bool = True,
        pretrained: bool = True,
        local_model_dir: str = "./pretrain/hgnetv2/",
        eval_mode: str = "rgb_ir",
    ) -> None:
        super().__init__()
        self.return_idx: List[int] = return_idx
        self.eval_mode: str = eval_mode

        # ======================================================================
        # 1. 主分支: 完整热红外 (TIR) 骨干网络 (ImageNet 预训练)
        # ======================================================================
        self.backbone_ir = HGNetv2(
            name=name,
            return_idx=return_idx,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            use_lab=use_lab,
            pretrained=pretrained,
            local_model_dir=local_model_dir,
        )

        # ======================================================================
        # 2. 辅助分支推进器: 实例化一个预训练的 HGNetv2, 拆解其预训练 Stage
        # ======================================================================
        pretrained_rgb_net = HGNetv2(
            name=name,
            return_idx=return_idx,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            use_lab=use_lab,
            pretrained=pretrained,
            local_model_dir=local_model_dir,
        )

        # (1) 提取初始表征 R (Stride 8, 256 通道) 的预训练组合
        self.rgb_stem = pretrained_rgb_net.stem
        self.rgb_stage0 = pretrained_rgb_net.stages[0]  # Stride 4, 通道 64
        self.rgb_stage1 = pretrained_rgb_net.stages[1]  # Stride 8, 通道 256 -> 产生 R

        # (2) 级联推进器: 将融合特征 A3 下采样并深度提取为 Stride 16, 512 通道
        self.rgb_stage2_cascade = pretrained_rgb_net.stages[2]  # 自带 stride 2 下采样 + 预训练 HG_Blocks

        # (3) 级联推进器: 将融合特征 A4 下采样并深度提取为 Stride 32, 1024 通道
        self.rgb_stage3_cascade = pretrained_rgb_net.stages[3]  # 自带 stride 2 下采样 + 预训练 HG_Blocks

        self.stage_channels: List[int] = [256, 512, 1024]
        c3, c4, c5 = self.stage_channels

        # ======================================================================
        # 3. 三组纯净 QualGate (原论文公式 4, 零冗余参数)
        # ======================================================================
        self.qual_gate_1 = QualGate(aux_channels=c3, ir_channels_list=[c3, c4, c5], reduction=4)
        self.qual_gate_2 = QualGate(aux_channels=c4, ir_channels_list=[c4, c5], reduction=4)
        self.qual_gate_3 = QualGate(aux_channels=c5, ir_channels_list=[c5], reduction=4)

    def forward(
        self, x: torch.Tensor
    ) -> Union[List[torch.Tensor], Dict[str, object]]:
        if x.shape[1] == 3:
            if not self.training and self.eval_mode == "ir":
                return self.backbone_ir(x)
            else:
                x = torch.cat([x, x], dim=1)

        assert x.shape[1] == 6, f"通道数应为 6，当前为: {x.shape[1]}"

        x_ir = x[:, :3, :, :]
        x_rgb = x[:, 3:, :, :]

        # ----------------------------------------------------------------------
        # 1. 主红外流: 提取多尺度金字塔 [C3, C4, C5]
        # ----------------------------------------------------------------------
        feats_ir: List[torch.Tensor] = self.backbone_ir(x_ir)
        c3, c4, c5 = feats_ir[0], feats_ir[1], feats_ir[2]

        if not self.training and self.eval_mode == "ir":
            return feats_ir

        # ----------------------------------------------------------------------
        # 2. 正统级联辅助流: 深度递进融合 (A3 -> Stage 2 -> A4 -> Stage 3 -> A5)
        # ----------------------------------------------------------------------
        # (1) 提取初始 RGB 表征 R (全预训练武装: 640 -> 320 -> 160 -> 80)
        x_rgb_stem = self.rgb_stem(x_rgb)
        x_rgb_s0 = self.rgb_stage0(x_rgb_stem)
        r = self.rgb_stage1(x_rgb_s0)  # [B, 256, 80, 80]

        # (2) 融合节点 1: R 与 [C3, C4, C5] 融合产生 A3
        a3, q3 = self.qual_gate_1(a_in=r, c_list=[c3, c4, c5])

        # (3) 级联推进 1: 将上一级融合好的 A3 经预训练 Stage2 下采样并深度强化
        a3_down = self.rgb_stage2_cascade(a3)  # [B, 512, 40, 40] - 包含了 A3 累加的多模态先验！

        # (4) 融合节点 2: A3_down 与 [C4, C5] 融合产生 A4
        a4, q4 = self.qual_gate_2(a_in=a3_down, c_list=[c4, c5])

        # (5) 级联推进 2: 将上一级融合好的 A4 经预训练 Stage3 下采样并深度强化
        a4_down = self.rgb_stage3_cascade(a4)  # [B, 1024, 20, 20] - 包含了 A3 和 A4 深度累积的信息！

        # (6) 融合节点 3: A4_down 与 [C5] 融合产生 A5
        a5, q5 = self.qual_gate_3(a_in=a4_down, c_list=[c5])

        feats_aux = [a3, a4, a5]

        if not self.training and self.eval_mode == "rgb_ir":
            return feats_aux

        return {
            "feats_main": feats_ir,
            "feats_aux": feats_aux,
            "quality_scores": {"q3": q3, "q4": q4, "q5": q5},
        }