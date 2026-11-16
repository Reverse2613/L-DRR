"""第二原型：保留B＋RGB→IR微校准，只替换三尺度融合为联合增益＋混洗互补。

使用独立注册类，原DualHGNetv2_PG/CGDMF源码不改；正在训练的模型无需重启。
公共参数与条件契约保留，新增层初始化不推进后续Encoder/Decoder的随机状态。
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import torch.nn as nn

from ..core import register
from .dt_cpn import DEFAULT_TEXTS_PATH, REPO_ROOT
from .dual_hgnetv2_pg import DualHGNetv2_PG
from .gain_complement_fusion import GainComplementFusion


@register()
class DualHGNetv2_PG_GainComp(DualHGNetv2_PG):
    r"""继承母体数据流，复用五项显式条件与IR监督，不添加新损失。"""

    def __init__(
        self, name: str = "B0", return_idx: Optional[Sequence[int]] = None,
        freeze_at: int = -1, freeze_norm: bool = False, use_lab: bool = True,
        pretrained: bool = True, local_model_dir: Union[str, Path] = "./pretrain/hgnetv2/",
        anchor_dir: Union[str, Path] = REPO_ROOT / "pretrain/condition_anchors/clip_vit_b16_64",
        texts_path: Union[str, Path] = DEFAULT_TEXTS_PATH,
        condition_mode: str = "real", enable_exchange: bool = True,
        dynamic_fusion: bool = True, condition_prior: str = "dt_cpn",
        enable_film: bool = True, condition_reference_row: int = 0,
        condition_random_seed: int = 20261112, enable_rgb_ir_alignment: bool = False,
        alignment_max_shift_px: float = 8.0,
        enable_joint_gain: bool = True, enable_complement: bool = True,
    ) -> None:
        r"""先建立完整母体，再升级融合层；新配置显式声明原母体全部选项。

        Args:
            name: 原HGNetv2版本，本轮B0。
            return_idx: 原输出索引，本轮[1,2,3]，步长8/16/32。
            freeze_at: 原骨干冻结规则。
            freeze_norm: 原BatchNorm冻结规则。
            use_lab: 原可学习仿射块开关。
            pretrained: 是否严格加载本地ImageNet预训练，本轮True。
            local_model_dir: 原骨干权重目录。
            anchor_dir: 原冻结CLIP缓存，本轮三条件512维表。
            texts_path: 原条件类别文本映射。
            condition_mode: 原真实/零条件控制。
            enable_exchange: 原双向差分消息，本轮False。
            dynamic_fusion: 本轮True；False时不能同时启用新融合分支。
            condition_prior: 原条件编码，本轮frozen_clip_joint。
            enable_film: 原FiLM，本轮True。
            condition_reference_row: 原常量条件对照行号。
            condition_random_seed: 原随机条件对照种子。
            enable_rgb_ir_alignment: 本轮True，沿用母体微校准。
            alignment_max_shift_px: 沿用当前母体的显式像素上限，本轮12.0。
            enable_joint_gain: 联合增益开关；False不增加该参数。
            enable_complement: 通道混洗互补开关；False不增加该参数。

        Raises:
            ValueError: 开关非法，或关闭动态融合却启用新分支。
            RuntimeError: 母体权重或新融合公共参数不兼容。
        """
        if any(type(flag) is not bool for flag in (enable_joint_gain, enable_complement)):
            raise ValueError("联合增益与混洗互补开关必须是bool。")
        if (enable_joint_gain or enable_complement) and dynamic_fusion is not True:
            raise ValueError("启用新融合分支需要dynamic_fusion=True，不能静默绕过。")
        super().__init__(
            name=name, return_idx=return_idx, freeze_at=freeze_at, freeze_norm=freeze_norm,
            use_lab=use_lab, pretrained=pretrained, local_model_dir=local_model_dir,
            anchor_dir=anchor_dir, texts_path=texts_path, condition_mode=condition_mode,
            enable_exchange=enable_exchange, dynamic_fusion=dynamic_fusion,
            condition_prior=condition_prior, enable_film=enable_film,
            condition_reference_row=condition_reference_row, condition_random_seed=condition_random_seed,
            enable_rgb_ir_alignment=enable_rgb_ir_alignment, alignment_max_shift_px=alignment_max_shift_px,
        )
        self.enable_joint_gain = enable_joint_gain
        self.enable_complement = enable_complement
        # 保持fusion_layers.i.rgb_score/ir_score/out_projection的原键；只增加
        # joint_gain/complement子键。from_base负责fork_rng与严格公共权重核验。
        self.fusion_layers = nn.ModuleList([
            GainComplementFusion.from_base(layer, enable_joint_gain, enable_complement)
            for layer in self.fusion_layers
        ])
