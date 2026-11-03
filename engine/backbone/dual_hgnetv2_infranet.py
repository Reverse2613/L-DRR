"""
engine/backbone/dual_hgnetv2_infranet.py

双流非对称质量感知红外引导主干网络 (DualHGNetv2_InfraNet).
论文: InfraNet: Quality-Aware RGB Guidance for Efficient Infrared Object Detection (arXiv:2607.03795)

核心机理:
    1. 主分支 (IR Pathway): 采用完整的 HGNetv2 提取多尺度特征金字塔 [C3, C4, C5];
    2. 辅助分支 (RGB Pathway): 采用轻量化 Stem 提取初始表征 R (Stride 8);
    3. 渐进式多尺度 QualGate 拓扑:
       - QualGate_1 (Stride 8):  输入 R 与 [C3, C4, C5] -> 输出 A3 (256 通道)
       - QualGate_2 (Stride 16): 输入 Downsample(A3) 与 [C4, C5] -> 输出 A4 (512 通道)
       - QualGate_3 (Stride 32): 输入 Downsample(A4) 与 [C5] -> 输出 A5 (1024 通道)
    4. 灵活推理模式 (eval_mode):
       - 'ir':     纯红外推理部署 (InfraNet-IR), 剪除 RGB 分支与 QualGate, 零额外开销;
       - 'rgb_ir': 双光自适应融合推理 (InfraNet-RGB-IR), 输出 QualGate 质量调控后的 [A3, A4, A5].
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .hgnetv2 import HGNetv2
from .qual_gate import QualGate
from ..core import register


class DownsampleBlock(nn.Module):
    r"""辅助分支层级间下采样与通道跃迁块 (Conv3x3 s=2 + BN + SiLU)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@register()
class DualHGNetv2_InfraNet(nn.Module):
    r"""InfraNet 非对称双流质量感知主干网络 (完整支持单红外与双光推理).

    输入:
        x (torch.Tensor): 
            - 训练期: 必须为 [Batch, 6, H, W] (IR: 0:3, RGB: 3:6);
            - 推理期: 若 eval_mode='ir'，可为 [Batch, 3, H, W] 或 [Batch, 6, H, W];
                     若 eval_mode='rgb_ir'，必须为 [Batch, 6, H, W].

    输出:
        - 训练期 (self.training == True):
            Dict:
                - "feats_main": List[Tensor] -> [C3, C4, C5] (来自 IR 主干)
                - "feats_aux":  List[Tensor] -> [A3, A4, A5] (来自 QualGate 融合金字塔)
                - "quality_scores": Dict[str, Tensor] -> {"q3": q3, "q4": q4, "q5": q5}
        - 推理期 (self.training == False):
            - eval_mode='ir':     List[Tensor] -> [C3, C4, C5] (纯红外独立运行)
            - eval_mode='rgb_ir': List[Tensor] -> [A3, A4, A5] (双光质量自适应融合运行)
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
        eval_mode: str = "ir",  # 新增: 控制推理模式 ('ir' 或 'rgb_ir')
    ) -> None:
        """初始化 InfraNet 双流主干网络.

        Args:
            name (str): HGNetv2 规格 ('B0', 'B1' 等), 默认为 'B0'.
            return_idx (List[int]): 需返回的特征阶段索引, 默认为 [1, 2, 3] 对应 S3, S4, S5.
            freeze_at (int): 冻结层数, 默认 -1.
            freeze_norm (bool): 是否冻结 BatchNorm 统计量.
            use_lab (bool): 是否使用可学习仿射变换 (Learnable Affine Block).
            pretrained (bool): 是否加载 ImageNet 预训练权重.
            local_model_dir (str): 预训练权重本地存储路径.
            eval_mode (str): 推理评估模式, 可选 'ir' (默认) 或 'rgb_ir'.
        """
        super().__init__()
        assert eval_mode in ["ir", "rgb_ir"], (
            f"eval_mode 参数必须为 'ir' 或 'rgb_ir'，当前为: '{eval_mode}'"
        )
        self.return_idx: List[int] = return_idx
        self.eval_mode: str = eval_mode

        # ======================================================================
        # 1. 实例化主分支: 完整热红外 (TIR) 骨干网络
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

        # 针对 B0 规格的标准输出通道: [S3=256, S4=512, S5=1024]
        self.stage_channels: List[int] = [256, 512, 1024]
        c3, c4, c5 = self.stage_channels

        # ======================================================================
        # 2. 实例化辅助分支: 轻量级可见光 (RGB) 初始特征提取器 (Stride 8)
        # ======================================================================
        # 实例化可见光主干 (同样拥有完整的 ImageNet Pretrained 强大起跑线！)
        self.backbone_rgb = HGNetv2(
            name=name,
            return_idx=return_idx,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            use_lab=use_lab,
            pretrained=pretrained,
            local_model_dir=local_model_dir,
        )

        # ======================================================================
        # 3. 实例化 3 组渐进式非对称 QualGate (忠实还原图 2 输入阶梯 4, 3, 2)
        # ======================================================================
        # Site 1 (Stride 8):  接收 R 与 [C3, C4, C5] -> 4 个输入
        self.qual_gate_1 = QualGate(
            aux_channels=c3,
            ir_channels_list=[c3, c4, c5],
            reduction=4,
        )

        # Site 2 (Stride 16): 接收 Downsample(A3) 与 [C4, C5] -> 3 个输入
        self.qual_gate_2 = QualGate(
            aux_channels=c4,
            ir_channels_list=[c4, c5],
            reduction=4,
        )

        # Site 3 (Stride 32): 接收 Downsample(A4) 与 [C5] -> 2 个输入
        self.qual_gate_3 = QualGate(
            aux_channels=c5,
            ir_channels_list=[c5],
            reduction=4,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Union[List[torch.Tensor], Dict[str, object]]:
        r"""前向传播计算逻辑."""
        # 兼容性防御：外部工具（如 profiler/calflops）可能传入 3 通道 Dummy Tensor
        if x.shape[1] == 3:
            if not self.training and self.eval_mode == "ir":
                # 纯红外推理模式，直接前向
                return self.backbone_ir(x)
            else:
                # 若处于训练期或双光评估模式，将 3 通道复制扩展为 6 通道防御崩溃
                x = torch.cat([x, x], dim=1)

        assert x.shape[1] == 6, f"预期输入通道数为 6 (3通道IR + 3通道RGB)，当前为: {x.shape[1]}"

        # 物理通道解耦
        x_ir = x[:, :3, :, :]   # [B, 3, H, W] 热红外图像
        x_rgb = x[:, 3:, :, :]  # [B, 3, H, W] 可见光图像

        # ======================================================================
        # 1. 主路径: 提取多尺度红外特征金字塔 [C3, C4, C5]
        # ======================================================================
        feats_ir: List[torch.Tensor] = self.backbone_ir(x_ir)
        c3, c4, c5 = feats_ir[0], feats_ir[1], feats_ir[2]

        # ======================================================================
        # 2. 推理期模式分支 A: InfraNet-IR 模式 (彻底剪除 RGB 与 QualGate)
        # ======================================================================
        if not self.training and self.eval_mode == "ir":
            return feats_ir

        # ======================================================================
        # 3. 执行辅助 RGB 支路与三级阶梯 QualGate 融合计算
        # (训练期必走；推理期若 eval_mode='rgb_ir' 也走这里)
        # ======================================================================
        # (1) 提取初始 RGB 表征 R (Stride 8, 通道 256)

        feats_rgb: List[torch.Tensor] = self.backbone_rgb(x_rgb)
        r3, r4, r5 = feats_rgb[0], feats_rgb[1], feats_rgb[2]

        # (2) 融合节点 1 (Site 3, Stride 8): 注入 [C3, C4, C5]
        a3, q3 = self.qual_gate_1(a_in=r3, c_list=[c3, c4, c5])

        # (3) 融合节点 2 (Site 4, Stride 16): 下采样后注入 [C4, C5]

        a4, q4 = self.qual_gate_2(a_in=r4, c_list=[c4, c5])

        # (4) 融合节点 3 (Site 5, Stride 32): 下采样后注入 [C5]

        a5, q5 = self.qual_gate_3(a_in=r5, c_list=[c5])

        feats_aux = [a3, a4, a5]

        # ======================================================================
        # 4. 推理期模式分支 B: InfraNet-RGB-IR 模式 (输出质量调控后的双光融合特征)
        # ======================================================================
        if not self.training and self.eval_mode == "rgb_ir":
            return feats_aux

        # ======================================================================
        # 5. 训练期: 返回双流特征与质量评分，供多损失联合优化
        # ======================================================================
        quality_scores = {"q3": q3, "q4": q4, "q5": q5}
        return {
            "feats_main": feats_ir,
            "feats_aux": feats_aux,
            "quality_scores": quality_scores,
        }