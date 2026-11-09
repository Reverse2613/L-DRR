"""CG-DMF：为互补后的RGB/IR支路打分，进行条件化局部选择。

每个尺度使用独立实例；Query区分支路顺序，分数由视觉、条件底分和
条件兼容项相加。输出一个融合特征图，接入原HybridEncoder。
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class LocalModalityScore(nn.Module):
    r"""评价一条支路，相比较的另一条支路始终放在第二个位置。"""

    def __init__(self, channels: int, condition_dim: int, query_dim: int) -> None:
        r"""建立有序Query、视觉分数及两类条件读出。

        Args:
            channels: 被评价支路的通道C。
            condition_dim: DT-CPN输出维度，默认使用128。
            query_dim: 局部Query/条件Key维度，方案使用32。
        """
        super().__init__()
        self.query = nn.Conv2d(2 * channels, query_dim, 1)
        self.key = nn.Linear(condition_dim, query_dim)
        self.condition_bias = nn.Linear(condition_dim, 1)
        self.visual_score = nn.Sequential(nn.GELU(), nn.Conv2d(query_dim, 1, 1))
        self.scale = math.sqrt(query_dim)
        # Query保留非零常规初始化；只将打分读出置零，初始两路各占0.5。
        # Query与Key同时全零会让兼容项缺少启动梯度，因此不能这样初始化。
        for layer in (self.key, self.condition_bias, self.visual_score[1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self, branch: torch.Tensor, other: torch.Tensor, condition: torch.Tensor,
    ) -> torch.Tensor:
        r"""计算被评价支路的三项局部分数。

        Args:
            branch: 被评价的互补后特征 [B,C,H,W]。
            other: 用于比较的另一支路 [B,C,H,W]。
            condition: c [B,condition_dim]。

        Returns:
            分数s [B,1,H,W]，包含视觉项、条件底分和Query-Key兼容项。
        """
        query = self.query(torch.cat((branch, other), dim=1))
        key = self.key(condition)[:, :, None, None]
        # FP32累加兼容项与分数，避免AMP下的点积累加/差分溢出。
        compatibility = (query.float() * key.float()).sum(dim=1, keepdim=True) / self.scale
        visual = self.visual_score(query).float()
        baseline = self.condition_bias(condition).float()[:, :, None, None]
        return visual + baseline + compatibility


class CGDMF(nn.Module):
    r"""单尺度动态融合：互补后的两路特征经空间权重选择，再重整通道。"""

    def __init__(
        self, channels: int, condition_dim: int = 128,
        query_dim: int = 32, norm_groups: int = 32,
    ) -> None:
        r"""建立两套有序打分与末端1×1输出。

        Args:
            channels: 特征通道，S3/S4/S5为256/512/1024。
            condition_dim: 条件向量维度，默认128。
            query_dim: Query/Key维度，默认32。
            norm_groups: 输出GroupNorm组数，默认32且必须整除channels。
        """
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in (channels, condition_dim, query_dim, norm_groups)):
            raise ValueError("CG-DMF维度和归一化组数必须是正整数。")
        if channels % norm_groups:
            raise ValueError("CG-DMF的norm_groups必须整除channels。")
        self.channels = channels
        self.condition_dim = condition_dim
        self.rgb_score = LocalModalityScore(channels, condition_dim, query_dim)
        self.ir_score = LocalModalityScore(channels, condition_dim, query_dim)
        self.out_projection = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False),
                                            nn.GroupNorm(norm_groups, channels))

    def _validate_inputs(self, rgb: torch.Tensor, ir: torch.Tensor, condition: torch.Tensor) -> None:
        r"""检查模态对应、batch、浮点类型、设备和非有限值。

        Args:
            rgb: 互补后RGB支路 [B,C,H,W]。
            ir: 互补后IR支路 [B,C,H,W]。
            condition: c [B,condition_dim]，与本模块同设备。
        """
        if not all(isinstance(v, torch.Tensor) for v in (rgb, ir, condition)):
            raise ValueError("CG-DMF的两路特征和条件必须都是张量。")
        if (rgb.ndim != 4 or rgb.shape[1] != self.channels
                or any(size <= 0 for size in rgb.shape) or ir.shape != rgb.shape):
            raise ValueError(f"CG-DMF两路特征必须同为非空[B,{self.channels},H,W]。")
        if condition.shape != (rgb.shape[0], self.condition_dim):
            raise ValueError(f"CG-DMF条件必须为[{rgb.shape[0]},{self.condition_dim}]。")
        allowed = (torch.float32, torch.float16, torch.bfloat16)
        if rgb.dtype not in allowed or ir.dtype != rgb.dtype or condition.dtype not in allowed:
            raise ValueError("两路特征需要相同浮点类型，条件需要float32/float16/bfloat16。")
        device = self.out_projection[0].weight.device
        if any(v.device != device for v in (rgb, ir, condition)):
            raise ValueError(f"CG-DMF输入应与模型同在{device}。")
        if not all(torch.isfinite(v).all() for v in (rgb, ir, condition)):
            raise ValueError("CG-DMF输入含NaN/Inf，不自动替换。")

    def compute_weights(
        self, rgb: torch.Tensor, ir: torch.Tensor, condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""计算两模态局部贡献，不持久保存计算图或特征热图。

        Args:
            rgb: 互补后RGB支路 [B,C,H,W]。
            ir: 互补后IR支路 [B,C,H,W]。
            condition: c [B,condition_dim]。

        Returns:
            M_rgb、M_ir各[B,1,H,W]，通道广播使用，逐位置和为1。
        """
        self._validate_inputs(rgb, ir, condition)
        s_rgb = self.rgb_score(rgb, ir, condition)
        s_ir = self.ir_score(ir, rgb, condition)
        rgb_weight = (s_rgb - s_ir).sigmoid()
        return rgb_weight, 1 - rgb_weight

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        r"""先凸组合两支路，再进行1×1通道重整与归一化。

        Args:
            rgb: CM-CME更新后的RGB支路 [B,C,H,W]。
            ir: CM-CME更新后的IR支路 [B,C,H,W]。
            condition: DT-CPN的条件向量 [B,condition_dim]。

        Returns:
            融合特征Y [B,C,H,W]；维度与原检测编码器输入一致。
        """
        m_rgb, m_ir = self.compute_weights(rgb, ir, condition)
        fused = m_rgb * rgb + m_ir * ir
        return self.out_projection(fused)
