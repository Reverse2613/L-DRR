"""模块一主干：双流HGNetv2 → DT-CPN → CM-CME → CG-DMF。

一个条件向量跨尺度使用，但每个尺度的互补与融合参数独立。
输入按仓库约定将IR放在前3通道、RGB放在后3通道。
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn

from ..core import register
from .cg_dmf import CGDMF
from .cm_cme import CMCME
from .dt_cpn import DEFAULT_TEXTS_PATH, DTCPN, REPO_ROOT
from .frozen_clip_condition import FrozenCLIPConditionPrior
from .frozen_clip_controls import ConstantCLIPConditionPrior, RandomCLIPConditionPrior
from .frozen_six_condition import FrozenSixCLIPConditionPrior
from .hgnetv2 import HGNetv2
from .rgb_ir_micro_alignment import RGBIRMicroAlignment, build_alignment_geometry


LOGGER = logging.getLogger(__name__)


@register()
class DualHGNetv2_PG(nn.Module):
    r"""条件引导的双流B0主干，输出三张融合特征图。

    Attributes:
        requires_conditions: 正常前向必须显式提供主干声明的图像级条件。
        profiling_channels: profiler应构造6通道图像，不能静默复制3通道。
        condition_encoder: 原DT-CPN输出[B,128]；联合冻结CLIP先验输出[B,512]。
        exchange_layers: 三个独立CM-CME实例。
        fusion_layers: 三个独立CG-DMF实例。
        rgb_ir_alignments: 仅微校准开关为True时存在的三个空间重采样模块。
    """

    requires_conditions = True
    profiling_channels = 6

    def __init__(
        self, name: str = "B0", return_idx: Optional[Sequence[int]] = None,
        freeze_at: int = -1, freeze_norm: bool = False, use_lab: bool = True,
        pretrained: bool = True, local_model_dir: Union[str, Path] = "./pretrain/hgnetv2/",
        anchor_dir: Union[str, Path] = REPO_ROOT / "pretrain/condition_anchors/clip_vit_b16_64",
        texts_path: Union[str, Path] = DEFAULT_TEXTS_PATH,
        condition_mode: str = "real", enable_exchange: bool = True,
        dynamic_fusion: bool = True,
        condition_prior: str = "dt_cpn",
        enable_film: bool = True,
        condition_reference_row: int = 0,
        condition_random_seed: int = 20261112,
        enable_rgb_ir_alignment: bool = False,
        alignment_max_shift_px: float = 8.0,
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
            anchor_dir: 对应条件表示的离线CLIP缓存及manifest目录。
            texts_path: 原始条件文本JSON，用于核对缓存类别编号。
            condition_mode: real使用真实条件；zero将c置零，保留FiLM/打分偏置。
                zero需从同一ImageNet预训练起点重新训练，才能检验条件的增益。
            enable_exchange: False只去掉双向消息，保留工作投影和FiLM主路径。
            dynamic_fusion: False固定两路各0.5，保留末端输出投影/归一化。
            condition_prior: dt_cpn保留原六条件128维路径；frozen_clip_joint只查
                weather/illumination/location联合文本的冻结512维缓存。
                该选项仅替换条件编码，CM-CME/CG-DMF沿用原机制和零初始化。
                frozen_clip_constant/random为相同512维接口的常量/随机编码对照。
                frozen_clip_six读取完整六条件原文对应的冻结512维缓存。
            enable_film: False只绕过条件仿射调制，工作投影与CG-DMF保留。
            condition_reference_row: 常量对照的预设原CLIP缓存行号。
            condition_random_seed: 随机对照表的种子，不消耗训练全局随机状态。
            enable_rgb_ir_alignment: 第一新增模块；False保留旧计算/权重结构。
                True在工作投影/Norm之后、FiLM之前只重采样RGB；本原型要求
                enable_exchange=False，显式读取已有height/angle元数据。
            alignment_max_shift_px: 三尺度统一的每分量输入像素偏移上限。
                高度/倾角调节经验包络；本轮512×640输入即原图单位。

        Raises:
            ValueError: B0或尺度规格不符。
            FileNotFoundError: 预训练文件缺失；不会联网下载或随机补齐。
            RuntimeError: 本地预训练不能严格匹配主干。
        """
        super().__init__()
        indices = [1, 2, 3] if return_idx is None else list(return_idx)
        if name != "B0" or indices != [1, 2, 3]:
            raise ValueError("模块一当前固定HGNetv2-B0与return_idx=[1,2,3]。")
        if condition_mode not in ("real", "zero"):
            raise ValueError("condition_mode只能是real或zero。")
        if condition_prior not in (
            "dt_cpn", "frozen_clip_joint", "frozen_clip_constant", "frozen_clip_random",
            "frozen_clip_six",
        ):
            raise ValueError("condition_prior不是已实现的显式条件编码选项。")
        if any(type(flag) is not bool for flag in (
            enable_exchange, dynamic_fusion, enable_film, enable_rgb_ir_alignment,
        )):
            raise ValueError("enable_exchange、dynamic_fusion、enable_film和微校准开关必须是bool。")
        if enable_rgb_ir_alignment and enable_exchange:
            raise ValueError("首个微校准原型只支持enable_exchange=False，避免同时改变双向消息。")
        self.condition_mode = condition_mode
        self.enable_exchange = enable_exchange
        self.dynamic_fusion = dynamic_fusion
        self.condition_prior = condition_prior
        self.enable_film = enable_film
        self.enable_rgb_ir_alignment = enable_rgb_ir_alignment
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

        if condition_prior == "dt_cpn":
            self.condition_encoder = DTCPN(anchor_dir, texts_path)
        elif condition_prior == "frozen_clip_constant":
            self.condition_encoder = ConstantCLIPConditionPrior(
                anchor_dir, texts_path, reference_row=condition_reference_row,
            )
        elif condition_prior == "frozen_clip_random":
            self.condition_encoder = RandomCLIPConditionPrior(
                anchor_dir, texts_path, random_seed=condition_random_seed,
            )
        elif condition_prior == "frozen_clip_six":
            self.condition_encoder = FrozenSixCLIPConditionPrior(anchor_dir, texts_path)
        else:
            self.condition_encoder = FrozenCLIPConditionPrior(anchor_dir, texts_path)
        self.condition_dim = getattr(self.condition_encoder, "output_dim", 128)
        self.required_condition_fields = getattr(
            self.condition_encoder, "required_fields",
            ("height", "angle", "time", "weather", "illumination", "location"),
        )
        # 512维先验直接进入原有FiLM/方向倾向/Key/底分映射。
        # 只扩大这些Linear的输入维度；空间运算、Query、消息和输出通道不改。
        self.exchange_layers = nn.ModuleList([
            CMCME(channels, condition_dim=self.condition_dim, enable_film=enable_film)
            for channels in self.stage_channels
        ])
        self.fusion_layers = nn.ModuleList([
            CGDMF(channels, condition_dim=self.condition_dim) for channels in self.stage_channels
        ])
        if enable_rgb_ir_alignment:
            self.required_condition_fields = tuple(dict.fromkeys(
                (*self.required_condition_fields, "height", "angle"),
            ))
            # 新模块初始化不推进全局CPU随机状态，保证后续Encoder/Decoder与B
            # 同seed初始化完全相同。旧配置不实例化新模块，不增加state_dict键。
            with torch.random.fork_rng(devices=[]):
                self.rgb_ir_alignments = nn.ModuleList([
                    RGBIRMicroAlignment(channels, stride, alignment_max_shift_px)
                    for channels, stride in zip(self.stage_channels, (8, 16, 32))
                ])
            LOGGER.info("启用RGB→IR微校准：每分量输入像素上限=%s，新增height/angle输入。",
                        alignment_max_shift_px)

    def forward(
        self, images: torch.Tensor, conditions: Mapping[str, Any],
    ) -> List[torch.Tensor]:
        r"""提取双流特征，先互补，再选择融合。

        Args:
            images: [B,6,H,W]；前3通道IR、后3通道RGB。
            conditions: 原模式为批量六条件；联合CLIP模式只读取三个原始
                类别编号[B]。所有张量与模型同设备，详见条件编码器契约。
                六条件联合模式接收condition_text的B条原文字符串，不读取GT框/类别。
                微校准开启时另需规范化height/angle各float32 [B,1]。

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
        # 消融只移除逐图条件信息；各Linear的bias仍可学习全局常量。
        # 仍严格核验输入条件，不能用伪造元数据绕过数据检查。
        if self.condition_mode == "zero":
            condition = torch.zeros_like(condition)
        features_ir = self.backbone_ir(images[:, :3])
        features_rgb = self.backbone_rgb(images[:, 3:])
        if len(features_ir) != 3 or len(features_rgb) != 3:
            raise RuntimeError("双流主干必须各输出S3/S4/S5三个尺度。")
        fused: List[torch.Tensor] = []
        if self.enable_rgb_ir_alignment:
            if "height" not in conditions or "angle" not in conditions:
                raise ValueError("启用RGB→IR微校准必须显式提供height/angle，不补默认值。")
            geometry, radius_fraction = build_alignment_geometry(
                conditions["height"], conditions["angle"],
            )
        for index, (rgb, ir) in enumerate(zip(features_rgb, features_ir)):
            # CM-CME与CG-DMF均使用命名顺序RGB、IR；不要受图像打包顺序影响。
            exchange = self.exchange_layers[index]
            if self.enable_exchange:
                rgb_updated, ir_updated = exchange(rgb, ir, condition)
            else:
                # 保留S和Z，避免把“去消息”混成“同时去投影/Norm/FiLM”。
                s_rgb, s_ir = exchange.rgb_work(rgb), exchange.ir_work(ir)
                if self.enable_rgb_ir_alignment:
                    s_rgb = self.rgb_ir_alignments[index](s_rgb, s_ir, geometry, radius_fraction)
                if self.enable_film:
                    rgb_updated = exchange._apply_film(s_rgb, exchange.rgb_film(condition))
                    ir_updated = exchange._apply_film(s_ir, exchange.ir_film(condition))
                else:
                    rgb_updated, ir_updated = s_rgb, s_ir
            fusion = self.fusion_layers[index]
            if self.dynamic_fusion:
                fused.append(fusion(rgb_updated, ir_updated, condition))
            else:
                # 只去动态选择，保留同一输出空间；这不是原始特征直接相加。
                fused.append(fusion.out_projection(0.5 * (rgb_updated + ir_updated)))
        return fused
