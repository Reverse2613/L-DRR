"""模块一主干：双流HGNetv2 → DT-CPN → CM-CME → CG-DMF。

一个条件向量跨尺度使用，但每个尺度的互补与融合参数独立。
输入按仓库约定将IR放在前3通道、RGB放在后3通道。
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn

from ..core import register
from .cg_dmf import CGDMF
from .cm_cme import CMCME
from .dt_cpn import DEFAULT_TEXTS_PATH, DTCPN, REPO_ROOT
from .hgnetv2 import HGNetv2


LOGGER = logging.getLogger(__name__)


@register()
class DualHGNetv2_PG(nn.Module):
    r"""条件引导的双流B0主干，输出三张融合特征图。

    Attributes:
        requires_conditions: 正常前向必须显式提供六条件。
        profiling_channels: profiler应构造6通道图像，不能静默复制3通道。
        condition_encoder: DT-CPN，只计算一次c [B,128]。
        exchange_layers: 三个独立CM-CME实例。
        fusion_layers: 三个独立CG-DMF实例。
    """

    requires_conditions = True
    profiling_channels = 6

    def __init__(
        self, name: str = "B0", return_idx: Optional[Sequence[int]] = None,
        freeze_at: int = -1, freeze_norm: bool = False, use_lab: bool = True,
        pretrained: bool = True, local_model_dir: Union[str, Path] = "./pretrain/hgnetv2/",
        anchor_dir: Union[str, Path] = REPO_ROOT / "pretrain/condition_anchors/clip_vit_b16_64",
        texts_path: Union[str, Path] = DEFAULT_TEXTS_PATH,
    ) -> None:
        r"""建立双独立预训练主干和三尺度条件融合。

        Args:
            name: 本版本固定B0，与256/512/1024通道规格一致。
            return_idx: 固定[1,2,3]，对应步长8/16/32。
            freeze_at: 沿用HGNetv2冻结规则，-1不冻结主干。
            freeze_norm: 是否冻结主干BatchNorm。
            use_lab: 是否启用HGNetv2原有可学习仿射块。
            pretrained: 是否严格读取本地ImageNet预训练，训练配置应为True。
            local_model_dir: 包含PPHGNetV2_B0_stage1.pth的本地目录。
            anchor_dir: 已生成的三个CLIP锚点与manifest目录。
            texts_path: 原始条件文本JSON，用于核对缓存类别编号。

        Raises:
            ValueError: B0或尺度规格不符。
            FileNotFoundError: 预训练文件缺失；不会联网下载或随机补齐。
            RuntimeError: 本地预训练不能严格匹配主干。
        """
        super().__init__()
        indices = [1, 2, 3] if return_idx is None else list(return_idx)
        if name != "B0" or indices != [1, 2, 3]:
            raise ValueError("模块一当前固定HGNetv2-B0与return_idx=[1,2,3]。")
        self.stage_channels = (256, 512, 1024)
        self.pretrained_path: Optional[str] = None
        self.pretrained_sha256: Optional[str] = None
        weight_path = Path(local_model_dir) / "PPHGNetV2_B0_stage1.pth"
        if pretrained and not weight_path.is_file():
            raise FileNotFoundError(f"缺少本地主干预训练：{weight_path}；不会自动下载。")

        # 将HGNetv2内部下载/吞异常路径关闭，由此处显式严格加载本地文件。
        kwargs = dict(name=name, return_idx=indices, freeze_at=freeze_at,
                      freeze_norm=freeze_norm, use_lab=use_lab, pretrained=False)
        self.backbone_ir = HGNetv2(**kwargs)
        self.backbone_rgb = HGNetv2(**kwargs)
        if pretrained:
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
            try:
                self.backbone_ir.load_state_dict(state, strict=True)
                self.backbone_rgb.load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise RuntimeError(f"{weight_path}不能严格匹配双流HGNetv2-B0。") from error
            self.pretrained_path = str(weight_path.resolve())
            self.pretrained_sha256 = hashlib.sha256(weight_path.read_bytes()).hexdigest()
            LOGGER.info("RGB与IR主干均严格加载本地预训练：%s", self.pretrained_path)

        self.condition_encoder = DTCPN(anchor_dir, texts_path)
        self.exchange_layers = nn.ModuleList([CMCME(channels) for channels in self.stage_channels])
        self.fusion_layers = nn.ModuleList([CGDMF(channels) for channels in self.stage_channels])

    def forward(
        self, images: torch.Tensor, conditions: Mapping[str, torch.Tensor],
    ) -> List[torch.Tensor]:
        r"""提取双流特征，先互补，再选择融合。

        Args:
            images: [B,6,H,W]；前3通道IR、后3通道RGB。
            conditions: 批量六条件；连续量已规范化/周期编码，类别为原始编号。
                所有张量与模型同设备，详见DTCPN的输入契约。

        Returns:
            [Y3,Y4,Y5]；512×640输入对应[B,256,64,80]、
            [B,512,32,40]、[B,1024,16,20]。
        """
        if (not isinstance(images, torch.Tensor) or images.ndim != 4
                or images.shape[1] != 6 or any(size <= 0 for size in images.shape)):
            raise ValueError("模块一图像必须是非空[B,6,H,W]，不自动复制3通道。")
        condition = self.condition_encoder(conditions)
        if condition.shape[0] != images.shape[0]:
            raise ValueError("图像batch与条件batch不一致。")
        features_ir = self.backbone_ir(images[:, :3])
        features_rgb = self.backbone_rgb(images[:, 3:])
        if len(features_ir) != 3 or len(features_rgb) != 3:
            raise RuntimeError("双流主干必须各输出S3/S4/S5三个尺度。")
        fused: List[torch.Tensor] = []
        for index, (rgb, ir) in enumerate(zip(features_rgb, features_ir)):
            # CM-CME与CG-DMF均使用命名顺序RGB、IR；不要受图像打包顺序影响。
            rgb_updated, ir_updated = self.exchange_layers[index](rgb, ir, condition)
            fused.append(self.fusion_layers[index](rgb_updated, ir_updated, condition))
        return fused
