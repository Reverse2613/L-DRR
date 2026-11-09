"""模块一检测器：显式条件主干 + 原HybridEncoder + 原OBB Decoder。

六条件通过独立参数传入；目标框/类别仅供训练期DN和损失使用。
"""

from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn

from ..core import register


@register()
class RTv4_PG(nn.Module):
    r"""组装条件引导模块一与现有旋转检测体系，不添加新损失。"""

    __inject__ = ["backbone", "encoder", "decoder"]
    requires_conditions = True
    profiling_channels = 6

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module) -> None:
        r"""保存三个可配置组件。

        Args:
            backbone: DualHGNetv2_PG，输出三尺度融合特征。
            encoder: 原HybridEncoder，输出三尺度224维特征。
            decoder: 原RTDETRTransformerv2OBB；本轮不启用GC。
        """
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, x: torch.Tensor, targets: Optional[List[Dict[str, Any]]] = None,
        teacher_encoder_output: Optional[torch.Tensor] = None,
        conditions: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        r"""通过显式条件完成旋转检测前向。

        Args:
            x: 图像对[B,6,H,W]，IR在前、RGB在后。
            targets: 训练GT列表，仅送入decoder的DN逻辑；推理时忽略。
            teacher_encoder_output: 保留原框架可选教师接口，本配置不使用。
            conditions: 必需的六条件字典；训练与推理均通过此参数显式传入。

        Returns:
            原检测器输出字典，含pred_logits[B,300,11]、pred_boxes[B,300,5]
            以及训练期原有辅助/DN输出；不增加条件预测损失或GC输出。

        Raises:
            ValueError: 显式条件缺失，不从GT或默认值补齐。
        """
        if conditions is None:
            raise ValueError("RTv4_PG需要显式conditions，训练和推理均不能省略。")
        features = self.backbone(x, conditions)
        encoded = self.encoder(features)
        student_output = None
        if self.training and isinstance(encoded, tuple) and len(encoded) == 2:
            encoded, student_output = encoded
        outputs = self.decoder(encoded, targets if self.training else None)
        if self.training and student_output is not None and teacher_encoder_output is not None:
            outputs["student_distill_output"] = student_output
            outputs["teacher_encoder_output"] = teacher_encoder_output
        return outputs

    def deploy(self) -> "RTv4_PG":
        r"""沿用组件已有的部署转换，保留显式条件输入。

        Returns:
            已切换eval模式的当前模型。
        """
        self.eval()
        for module in self.modules():
            if hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()
        return self
