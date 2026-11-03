"""
engine/backbone/qual_gate.py

Quality-Aware Fusion Module (QualGate) 核心实现.
论文: InfraNet: Quality-Aware RGB Guidance for Efficient Infrared Object Detection (arXiv:2607.03795)

功能描述:
    1. 接收可见光特征 a_in 与多尺度红外特征 c_list;
    2. 基于任务自适应计算可见光质量可信度标量 q in (0, 1);
    3. 对可见光特征执行质量门控抑制: w_rgb(q) * a_in = q * a_in;
    4. 对红外特征执行多尺度通道对齐与反向补偿放大: alpha(q) = clip(1.5 - q, 1.0, 1.5);
    5. 输出高质量融合特征，彻底阻断恶劣环境下退化可见光噪声的负迁移倒灌.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAdapter(nn.Module):
    r"""多尺度通道与空间分辨率适配器.
    
    将不同尺度、不同通道数的红外特征 C_k 映射到目标空间尺度 (H, W) 与目标通道数 C_target.
    
    输入:
        x (torch.Tensor): 形状为 [Batch, C_in, H_k, W_k] 的特征图.
        target_size (Tuple[int, int]): 目标空间分辨率 (H_target, W_target).
        
    输出:
        torch.Tensor: 形状为 [Batch, C_target, H_target, W_target] 的对齐后特征图.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """初始化通道适配器.

        Args:
            in_channels (int): 输入红外特征的通道数.
            out_channels (int): 目标对齐通道数 (与 a_in 通道一致).
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(
        self, x: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        # 1. 通道投影降维/升维
        x = self.proj(x)

        # 2. 空间几何分辨率安全对齐
        if x.shape[-2:] != target_size:
            x = F.interpolate(
                x,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return x


class QualGate(nn.Module):
    r"""QualGate: 质量感知门控融合模块.

    计算流程:
        1. q = Sigmoid(MLP_qual(GAP(a_in)))
        2. alpha(q) = clamp(1.5 - q, min=1.0, max=1.5)
        3. 对每个红外尺度 k:
           F_k = ChannelAdapter_k(C_k) 对齐至 a_in 的尺寸与通道
           g_k = Sigmoid(MLP_gate_k(GAP(F_k)))
        4. a_out = (q * a_in) + sum_k [ alpha(q) * g_k * F_k ]
    """

    def __init__(
        self,
        aux_channels: int,
        ir_channels_list: List[int],
        reduction: int = 4,
    ) -> None:
        """初始化 QualGate 模块.

        Args:
            aux_channels (int): 当前融合节点辅助可见光特征 a_in 的通道数.
            ir_channels_list (List[int]): 红外主干网络提供的多尺度特征通道列表 (如 [256, 512, 1024]).
            reduction (int): MLP 瓶颈层的通道缩减比例，默认值为 4.
        """
        super().__init__()
        self.aux_channels: int = aux_channels
        self.ir_channels_list: List[int] = ir_channels_list

        # ======================================================================
        # 1. 质量评估网络 MLP_qual: 输出标量可信度 q，整个通道和空间都共用q
        # ======================================================================
        mid_dim = max(16, aux_channels // reduction)
        self.mlp_qual = nn.Sequential(
            nn.Linear(aux_channels, mid_dim, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid_dim, 1, bias=True),
            nn.Sigmoid(),
        )

        # ======================================================================
        # 2. 多尺度红外对齐适配器与内容门控网络
        # ======================================================================
        self.adapters = nn.ModuleList([
            ChannelAdapter(in_channels=c_ir, out_channels=aux_channels)
            for c_ir in ir_channels_list
        ])

        # 为每个红外尺度建立通道级内容门控，各个通道不一样的权重，但是同一通道不同空间位置共享同一个值
        self.content_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(aux_channels, mid_dim, bias=False),
                nn.SiLU(inplace=True),
                nn.Linear(mid_dim, aux_channels, bias=True),
                nn.Sigmoid(),
            )
            for _ in ir_channels_list
        ])

    def forward(
        self, a_in: torch.Tensor, c_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""QualGate 前向传播.

        Args:
            a_in (torch.Tensor): 辅助可见光特征, 形状为 [B, C_aux, H, W].
            c_list (List[torch.Tensor]): 红外多尺度特征列表, 例如:
                - C_1: [B, C_ir1, H1, W1]
                - C_2: [B, C_ir2, H2, W2]
                - C_3: [B, C_ir3, H3, W3]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - a_out (torch.Tensor): 质量调控后的融合特征图 [B, C_aux, H, W].
                - q (torch.Tensor): 当前批次各图像的质量可靠性系数 [B, 1].
        """
        batch_size, c_aux, h, w = a_in.shape
        target_size: Tuple[int, int] = (h, w)
        assert c_aux == self.aux_channels, (
            f"输入 a_in 通道数 ({c_aux}) 与初始化通道数 ({self.aux_channels}) 不匹配!"
        )
        assert len(c_list) == len(self.ir_channels_list), (
            f"输入红外特征尺度数 ({len(c_list)}) 与注册数 ({len(self.ir_channels_list)}) 不匹配!"
        )

        # ======================================================================
        # 步骤 1: 计算可靠性控制标量 q in (0, 1)
        # ======================================================================
        # 全局平均池化: [B, C_aux, H, W] -> [B, C_aux]
        # # 将 F.adaptive_avg_pool2d 替换为原生的 .mean(dim=(-2, -1))
        gap_a = a_in.mean(dim=(-2, -1))
        q = self.mlp_qual(gap_a)  # [B, 1]

        # 转换为广播张量: [B, 1, 1, 1]
        q_broad = q.view(batch_size, 1, 1, 1)

        # ======================================================================
        # 步骤 2: 计算红外放大补偿系数 alpha(q)
        # alpha(q) = clip(1.5 - q, 1.0, 1.5)
        # ======================================================================
        alpha_q = torch.clamp(1.5 - q_broad, min=1.0, max=1.5)  # [B, 1, 1, 1]

        # ======================================================================
        # 步骤 3: 可见光抑制门控 (RGB Suppression)
        # ======================================================================
        suppressed_rgb = q_broad * a_in  # [B, C_aux, H, W]

        # ======================================================================
        # 步骤 4: 多尺度红外对齐、门控与加权注入 (IR Injection & Amplification)
        # ======================================================================
        ir_fused = torch.zeros_like(a_in)

        for k, c_k in enumerate(c_list):
            # 空间与通道适配对齐: [B, C_irk, H_k, W_k] -> [B, C_aux, H, W]
            f_k = self.adapters[k](c_k, target_size=target_size)

            # 内容门控计算: [B, C_aux, H, W] -> [B, C_aux] -> [B, C_aux, 1, 1]
            # 使用原生 .mean(dim=(-2, -1)) 代替池化
            gap_f = f_k.mean(dim=(-2, -1))
            g_k = self.content_gates[k](gap_f).view(batch_size, c_aux, 1, 1)

            # 注入累加: alpha(q) * g_k * F_k
            ir_fused = ir_fused + (alpha_q * g_k * f_k)

        # ======================================================================
        # 步骤 5: 最终融合与残差平滑
        # ======================================================================
        a_out = suppressed_rgb + ir_fused

        return a_out, q