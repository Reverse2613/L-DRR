"""
engine/rtv4/rtv4_infranet.py

RT-DETRv4 / RiO-DETR 核心模型容器 (升级支持 InfraNet 双流特权共享 Head 机制).
"""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from ..core import register


__all__ = ['RTv4_Infranet', ]


@register()
class RTv4_Infranet(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        teacher_encoder_output: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        r"""前向传播计算.

        Args:
            x (torch.Tensor): 输入图像张量 (训练时 [B, 6, H, W]; 推理时 [B, 3, H, W] 或 [B, 6, H, W]).
            targets (Optional[List[Dict]]): 标注信息字典列表.
            teacher_encoder_output: 蒸馏教师特征 (若不启用则为 None).

        Returns:
            Dict[str, object]: 包含分类预测、旋转边界框预测及辅助预测的字典.
        """
        # 1. 提取多尺度骨干特征
        x_backbone = self.backbone(x)

        # ======================================================================
        # 分支 A: 训练状态下的 InfraNet 双流共享 Head 机制
        # (检测到主干返回字典，包含 feats_main 与 feats_aux)
        # ======================================================================
        if self.training and isinstance(x_backbone, dict) and "feats_main" in x_backbone:
            feats_main: List[torch.Tensor] = x_backbone["feats_main"]
            feats_aux: List[torch.Tensor] = x_backbone["feats_aux"]
            quality_scores: Dict[str, torch.Tensor] = x_backbone.get("quality_scores", {})

            # ------------------------------------------------------------------
            # (1) 主红外分支前向 (Main Stream: P3, P4, P5 -> Main Loss)
            # ------------------------------------------------------------------
            enc_out_main = self.encoder(feats_main)
            student_distill_output = None
            if isinstance(enc_out_main, tuple) and len(enc_out_main) == 2:
                fpn_main, student_distill_output = enc_out_main
            else:
                fpn_main = enc_out_main

            dec_out_main = self.decoder(fpn_main, targets)

            # ------------------------------------------------------------------
            # (2) 辅助融合分支前向 (Aux Stream: A3, A4, A5 -> Aux Loss)
            # 复用同一个 encoder 和同一个 decoder (参数共享，强化特征流形对齐)
            # ------------------------------------------------------------------
            enc_out_aux = self.encoder(feats_aux)
            if isinstance(enc_out_aux, tuple) and len(enc_out_aux) == 2:
                fpn_aux, _ = enc_out_aux
            else:
                fpn_aux = enc_out_aux

            dec_out_aux = self.decoder(fpn_aux, targets)

            # ------------------------------------------------------------------
            # (3) 结果打包: 将 dec_main 作为主干，挂载 aux_infranet 与质量监控
            # ------------------------------------------------------------------
            dec_out_main["aux_infranet"] = dec_out_aux
            dec_out_main["quality_scores"] = quality_scores

            if student_distill_output is not None and teacher_encoder_output is not None:
                dec_out_main["student_distill_output"] = student_distill_output
                dec_out_main["teacher_encoder_output"] = teacher_encoder_output

            return dec_out_main

        # ======================================================================
        # 分支 B: 评估/推理状态 (或普通单模态训练)
        # (此时 x_backbone 已经由主干网络根据 eval_mode 剪枝为标准的 List[Tensor])
        # ======================================================================
        encoder_output = self.encoder(x_backbone)

        student_distill_output = None
        if self.training and isinstance(encoder_output, tuple) and len(encoder_output) == 2:
            x_fpn_features, student_distill_output = encoder_output
        else:
            x_fpn_features = encoder_output

        x_decoder_out = self.decoder(x_fpn_features, targets)

        if self.training and student_distill_output is not None and teacher_encoder_output is not None:
            x_decoder_out["student_distill_output"] = student_distill_output
            x_decoder_out["teacher_encoder_output"] = teacher_encoder_output

        return x_decoder_out

    def deploy(self):
        r"""模型转换为推理部署形态."""
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self