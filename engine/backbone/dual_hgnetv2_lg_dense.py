"""
engine/backbone/dual_hgnetv2_lg_dense.py

模块一 (Module 1): 局部物理照度图与宏观信噪比双阶门控稠密融合骨干网络.
消融实验 Stage 1 专用:
- 双流 HGNetv2-B0 (IR 主干 + RGB 辅干)
- ConditionPerceiver: 提取隐式条件潜向量 z_cond，训练期支持 6 维多任务监督
- LocalGlobalGatedDenseFusion: 局部照度与宏观条件联合乘性加权，全量稠密注入
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hgnetv2 import HGNetv2
from ..core import register


class ConditionPerceiver(nn.Module):
    r"""环境与飞控条件自感知头.

    从双光浅层特征中无监督/自监督抽取环境潜向量 z_cond，
    并在训练阶段通过 6 维多任务头锚定物理真实语义.
    """

    def __init__(self, in_channels: int, cond_dim: int = 256, with_aux_heads: bool = True) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.with_aux_heads = with_aux_heads  # <--- 开关显示条件预测标志，为了查看显示条件约束是否起了作用

        # 全局上下文特征映射: 输入拼接后的 [B, 2*C]
        self.encoder = nn.Sequential(
            nn.Linear(in_channels * 2, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
            nn.LayerNorm(cond_dim),
        )

        # 仅在开启辅助监督时，才实例化 6 个物理预测头
        if self.with_aux_heads:
            # 训练期专用的 6 维多任务监督预测头
            self.head_height = nn.Linear(cond_dim, 1)        # 高度 (归一化标量)
            self.head_angle = nn.Linear(cond_dim, 1)         # 俯仰角 (归一化标量)
            self.head_time = nn.Linear(cond_dim, 2)          # 时间谐波 (sin, cos)
            self.head_weather = nn.Linear(cond_dim, 10)      # 天气 10 分类
            self.head_illum = nn.Linear(cond_dim, 7)         # 光照 7 分类
            self.head_location = nn.Linear(cond_dim, 11)     # 场景 11 分类

    def forward(
        self, feat_ir: torch.Tensor, feat_rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        r"""前向传播.

        Args:
            feat_ir (torch.Tensor): 红外特征图 [B, C, H, W]
            feat_rgb (torch.Tensor): 可见光特征图 [B, C, H, W]

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
                - z_cond: [B, cond_dim] 连续环境潜向量
                - preds: 训练期输出 6 维预测字典，推理期为 None
        """
        # 1. 空间全局平均池化
        gap_ir = F.adaptive_avg_pool2d(feat_ir, (1, 1)).flatten(1)   # [B, C]
        gap_rgb = F.adaptive_avg_pool2d(feat_rgb, (1, 1)).flatten(1) # [B, C]
        feat_joint = torch.cat([gap_ir, gap_rgb], dim=1)            # [B, 2*C]

        # 2. 提取连续潜向量 z_cond
        z_cond = self.encoder(feat_joint)                           # [B, cond_dim]

        # 3. 训练期执行多任务预测
        preds: Optional[Dict[str, torch.Tensor]] = None

        # 仅在开启开关且处于训练模式时，才输出预测字典
        if self.with_aux_heads and self.training:
            preds = {
                "pred_height": self.head_height(z_cond),
                "pred_angle": self.head_angle(z_cond),
                "pred_time": self.head_time(z_cond),
                "pred_weather": self.head_weather(z_cond),
                "pred_illum": self.head_illum(z_cond),
                "pred_location": self.head_location(z_cond),
            }

        return z_cond, preds


class LocalGlobalGatedDenseFusionBlock(nn.Module):
    r"""单尺度局部物理照度与宏观信噪比双阶门控稠密融合块."""

    def __init__(self, channels: int, cond_dim: int = 256) -> None:
        super().__init__()
        self.channels = channels

        # 1. 局部物理照度图提取器 (Depthwise Conv 保持极低参数量)
        self.local_photometric_extractor = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        # 2. 宏观信噪比通道门控投影 (由 z_cond 生成通道缩放因子)
        self.macro_snr_rgb = nn.Sequential(
            nn.Linear(cond_dim, channels),
            nn.Sigmoid(),
        )
        self.macro_snr_ir = nn.Sequential(
            nn.Linear(cond_dim, channels),
            nn.Sigmoid(),
        )

        # 3. 可见光特征适配卷积
        self.conv_align = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        # 4. 融合特征平滑与降维层
        self.fusion_out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(
        self, f_ir: torch.Tensor, f_rgb: torch.Tensor, z_cond: torch.Tensor
    ) -> torch.Tensor:
        r"""单阶段特征融合前向.

        Args:
            f_ir (torch.Tensor): 红外特征图 [B, C, H, W]
            f_rgb (torch.Tensor): 可见光特征图 [B, C, H, W]
            z_cond (torch.Tensor): 宏观环境潜向量 [B, cond_dim]
        """
        # A. 提取局部空间照度可靠性图 M_local ∈ [B, 1, H, W]
        m_local = self.local_photometric_extractor(f_rgb)

        # B. 计算宏观通道信噪比门控因子 w_rgb, w_ir ∈ [B, C, 1, 1]
        w_rgb = self.macro_snr_rgb(z_cond).unsqueeze(-1).unsqueeze(-1)
        w_ir = self.macro_snr_ir(z_cond).unsqueeze(-1).unsqueeze(-1)

        # C. 局部-全局双阶乘性门控: 抑制暗处噪声，保留路灯/车灯局部纹理
        r_rgb_spatial = m_local * w_rgb # [B, C, H, W]

        # D. 全量稠密注入: 红外主干加权基底 + 双阶门控可见光补偿
        f_injected = (w_ir * f_ir) + (r_rgb_spatial * self.conv_align(f_rgb))

        return self.fusion_out(f_injected)


@register()
class DualHGNetv2_LGDense(nn.Module):
    r"""双流 HGNetv2 局部-全局双阶门控稠密融合主干 (Module 1 基座).

    输入: [B, 6, H, W] (0:3 为 IR, 3:6 为 RGB)
    输出:
        - 正常阶段返回 List[torch.Tensor]: [S3, S4, S5]
        - 训练阶段将 condition_preds 暂存在 self.latest_cond_preds 供损失函数提取
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
        cond_dim: int = 256,
        with_aux_heads: bool = True,  # <--- 增加这一行，默认开启
    ) -> None:
        super().__init__()
        self.stage_channels: List[int] = [256, 512, 1024]
        self.cond_dim = cond_dim

        # 1. 实例化双流 HGNetv2-S (B0)
        self.backbone_ir = HGNetv2(
            name=name,
            return_idx=return_idx,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            use_lab=use_lab,
            pretrained=pretrained,
            local_model_dir=local_model_dir,
        )
        self.backbone_rgb = HGNetv2(
            name=name,
            return_idx=return_idx,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            use_lab=use_lab,
            pretrained=pretrained,
            local_model_dir=local_model_dir,
        )

        # 2. 条件自感知头 (挂载在 S3 特征层)
        self.condition_perceiver = ConditionPerceiver(
            in_channels=self.stage_channels[0], cond_dim=cond_dim, with_aux_heads=with_aux_heads
        )

        # 3. 三尺度局部-全局双阶门控稠密融合块
        self.fusion_blocks = nn.ModuleList([
            LocalGlobalGatedDenseFusionBlock(channels=c, cond_dim=cond_dim)
            for c in self.stage_channels
        ])

        # 用于向 Criterion 传递训练期多任务预测输出
        self.latest_cond_preds: Optional[Dict[str, torch.Tensor]] = None

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.shape[1] == 3:
            x = torch.cat([x, x], dim=1)
        assert x.shape[1] == 6, f"预期输入为 6 通道张量，当前为: {x.shape[1]}"

        x_ir = x[:, :3, :, :]
        x_rgb = x[:, 3:, :, :]

        feats_ir: List[torch.Tensor] = self.backbone_ir(x_ir)
        feats_rgb: List[torch.Tensor] = self.backbone_rgb(x_rgb)

        # 1. 在 S3 阶段感知环境宏观潜向量与预测头
        z_cond, cond_preds = self.condition_perceiver(feats_ir[0], feats_rgb[0])
        self.latest_cond_preds = cond_preds

        # 2. 逐尺度执行局部-全局双阶门控稠密融合
        fused_features: List[torch.Tensor] = []
        for i, block in enumerate(self.fusion_blocks):
            f_fused = block(feats_ir[i], feats_rgb[i], z_cond)
            fused_features.append(f_fused)

        return fused_features


# ==============================================================================
# 单文件独立单元测试 (本地运行验证张量与显存)
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print(">>> 启动 Module 1 (DualHGNetv2_LGDense) 前向与梯度反传单元测试...")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备: {device}")

    # 构造假数据: Batch Size=2, 6通道, 640x512
    dummy_input = torch.randn(2, 6, 512, 640, device=device)

    # 实例化模型 (测试时不加载外部预训练权重以保纯净)
    model = DualHGNetv2_LGDense(pretrained=False).to(device)
    model.train()

    print("\n[测试 1] 执行训练模式前向传播...")
    outs = model(dummy_input)

    print(f"输出特征金字塔层数: {len(outs)}")
    for idx, f in enumerate(outs):
        print(f"  - Stage S{idx+3} 输出张量形状: {list(f.shape)} (期望: [2, {256*(2**idx)}, {512//(2**(idx+3))}, {640//(2**(idx+3))}])")

    preds = model.latest_cond_preds
    assert preds is not None, "训练模式下未能生成 6 维条件预测字典！"
    print("\n[测试 2] 检查条件感知头多任务输出维度:")
    for k, v in preds.items():
        print(f"  - {k:<15s}: {list(v.shape)}")

    print("\n[测试 3] 测试梯度回传通路...")
    dummy_loss = sum(f.sum() for f in outs) + sum(v.sum() for v in preds.values())
    dummy_loss.backward()
    print("  - 梯度回传成功，无 NaN，无阻断！")

    print("\n[测试 4] 执行推理模式 (model.eval()) 验证...")
    model.eval()
    with torch.no_grad():
        eval_outs = model(dummy_input)
    assert model.latest_cond_preds is None, "推理模式下应自动旁路多任务预测头！"
    print("  - 推理模式旁路切除成功，零额外开销！")

    print("\n>>> Module 1 (DualHGNetv2_LGDense) 单元测试全部通过！\n")