"""单张 RGB/IR 配对图像的模块一可视化，不训练、不下载、不计算 AP。

先按训练配置严格加载 checkpoint，再用只读 hook 观察真实模块一前向。
CG-DMF 权重是互补后支路的任务贡献，不是经过校准的传感器可靠性；
CM-CME 的门控按通道取平均显示，原始逐通道门控也保存在 maps.npz。
同一张图只换条件的 --compare-xml 是推理干预，不能替代重训消融。
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig
from engine.data.dataset.atrumod_dataset import ATRUMODPairDataset
from engine.data.transforms.container import Compose
from engine.data.condition_utils import ConditionBatch, TEXT_CONDITION_FIELD
from engine.data.condition_texts import build_six_condition_text, load_condition_descriptions


LEGACY_FIELDS = ("height", "angle", "time", "weather", "illumination", "location")
CATEGORY_COUNTS = {"weather": 10, "illumination": 7, "location": 11}
XML_FIELDS = {"height": "UAVheight", "angle": "UAVangle", "time": "time",
              "weather": "weather", "illumination": "illumination", "location": "location"}
VALID_TRANSFORMS = {"PairedResizeOBB", "PairedSanitizeBoundingBoxesOBB",
                    "PairedConvertPILToTensor", "PairedConvertOBB"}


@dataclass(frozen=True)
class InputPair:
    """实际图像路径、条件 XML 及自动配对来源。"""

    ir: Path
    rgb: Path
    xml: Optional[Path]
    split: Optional[str]


def file_sha256(path: Path) -> str:
    """分块计算文件校验值，避免额外读取整份 checkpoint 到内存。

    Args:
        path: 已存在的本地文件。

    Returns:
        文件内容的 SHA256 十六进制字符串。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_pair_file(folder: Path, stem: str) -> Path:
    """查找同名 RGB 图片，多个扩展名同时存在时明确报错。

    Args:
        folder: RGB 图像目录。
        stem: IR 图像文件名去掉扩展名后的 ID。

    Returns:
        唯一的同名 RGB 图片路径。
    """
    matches = [path for path in folder.iterdir()
               if path.stem == stem and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if len(matches) != 1:
        raise ValueError(f"{folder} 中 ID={stem} 的 RGB 配对数为 {len(matches)}；请用 --rgb-image 明确指定。")
    return matches[0].resolve()


def resolve_inputs(args: argparse.Namespace, cfg: YAMLConfig) -> InputPair:
    """由配置寻找 ATR-UMOD 配对；外部图像必须明确提供另一模态。

    Args:
        args: 单 IR 路径或显式 RGB/IR/XML 路径。
        cfg: 用于识别 train/val 图像和标签目录的模型配置。

    Returns:
        已校验存在的两张图片，以及条件 XML 的候选路径。
    """
    ir = (args.image or args.ir_image).expanduser().resolve()
    if not ir.is_file():
        raise FileNotFoundError(f"IR 图片不存在：{ir}")
    matching = None
    split = None
    for name in ("val_dataloader", "train_dataloader"):
        dataset = cfg.yaml_cfg.get(name, {}).get("dataset", {})
        folder = dataset.get("img_folder")
        if folder and ir.parent == Path(folder).expanduser().resolve():
            matching, split = dataset, name
            break
    if args.rgb_image:
        rgb = args.rgb_image.expanduser().resolve()
    elif matching is not None:
        rgb_folder = matching.get("img_folder_rgb")
        if rgb_folder is None:
            rgb_folder = ATRUMODPairDataset._deduce_rgb_folder(str(ir.parent))
        rgb = find_pair_file(Path(rgb_folder), ir.stem)
    else:
        raise ValueError("该 IR 路径不属于配置的 ATR-UMOD 目录；外部图片必须提供 --rgb-image。")
    if not rgb.is_file():
        raise FileNotFoundError(f"RGB 图片不存在：{rgb}")
    xml = args.xml.expanduser().resolve() if args.xml else None
    manual = any(getattr(args, field) is not None for field in LEGACY_FIELDS)
    if xml is None and matching is not None and not manual:
        xml = Path(matching["ann_folder"]) / f"{ir.stem}.xml"
    return InputPair(ir=ir, rgb=rgb, xml=xml, split=split)


def required_condition_fields(backbone: nn.Module) -> Tuple[str, ...]:
    """读取新编码器字段声明；原 DTCPN 的字段契约固定为六条件。

    Args:
        backbone: 包含 condition_encoder 的模块一主干。

    Returns:
        模型实际需要的字段名元组，不按条件向量维度猜测。
    """
    fields = tuple(getattr(backbone.condition_encoder, "required_fields", LEGACY_FIELDS))
    if (not fields or len(set(fields)) != len(fields)
            or (fields != (TEXT_CONDITION_FIELD,) and not set(fields).issubset(LEGACY_FIELDS))):
        raise ValueError(f"无法识别条件编码器 required_fields={fields}。")
    return fields


def condition_tensors(
    raw: Mapping[str, Any], fields: Sequence[str], device: torch.device,
    texts_path: Optional[Path] = None,
) -> ConditionBatch:
    """严格构造单图条件张量，沿用数据集的物理单位和周期编码。

    Args:
        raw: 未归一化条件；高度为米、倾角为度、时间为 HH:MM/ISO 字符串。
        fields: 模型需要的字段；三条件模式不读取 H/角度/时间。
        device: 条件张量与模型使用的 CUDA 设备。
        texts_path: F组官方描述表；默认沿用仓库码表，原模式不读取。

    Returns:
        连续量 [1,1]/time[1,2] float32，类别 [1] int64。
        F组返回长度1的condition_text列表，保持XML数字/日期原文。
    """
    if tuple(fields) == (TEXT_CONDITION_FIELD,):
        if any(field not in raw or raw[field] is None or not str(raw[field]).strip()
               for field in LEGACY_FIELDS):
            raise ValueError("F组需要完整六条件原文，不能只提供三属性。")
        descriptions = load_condition_descriptions(
            texts_path or REPO_ROOT / "configs/dataset/atrumod_condition_texts.json",
        )
        text = build_six_condition_text(
            {XML_FIELDS[field]: str(raw[field]) for field in LEGACY_FIELDS}, descriptions,
        )
        return {TEXT_CONDITION_FIELD: [text]}
    result: Dict[str, torch.Tensor] = {}
    for field in fields:
        if field not in raw or raw[field] is None or str(raw[field]).strip() == "":
            raise ValueError(f"缺少真实条件 {field}；不会根据图片或文件名猜测。")
        value = raw[field]
        if field in CATEGORY_COUNTS:
            # 不接受 1.5/"1.0" 之类的编号，也不把越界编号裁剪为合法值。
            if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                raise ValueError(f"{field} 必须是整数原始类别编号。")
            index = int(value)
            if not 0 <= index < CATEGORY_COUNTS[field]:
                raise ValueError(f"{field}={index} 不在 0–{CATEGORY_COUNTS[field] - 1}。")
            result[field] = torch.tensor([index], dtype=torch.int64, device=device)
        elif field in ("height", "angle"):
            number = float(value)
            lower, upper = (80.0, 300.0) if field == "height" else (0.0, 75.0)
            if not math.isfinite(number) or not lower <= number <= upper:
                raise ValueError(f"{field}={number} 不在 {lower}–{upper}。")
            normalized = (number - 80.0) / 220.0 if field == "height" else number / 75.0
            result[field] = torch.tensor([[normalized]], dtype=torch.float32, device=device)
        else:
            match = re.search(r"(?:T|^)(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$", str(value).strip())
            if match is None:
                raise ValueError(f"time={value!r} 不是数据集支持的真实时刻。")
            hour, minute, second = int(match[1]), int(match[2]), int(match[3] or 0)
            if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
                raise ValueError(f"time={value!r} 的时分秒越界。")
            phase = 2.0 * math.pi * (hour + minute / 60.0 + second / 3600.0) / 24.0
            result[field] = torch.tensor([[math.sin(phase), math.cos(phase)]],
                                         dtype=torch.float32, device=device)
    return result


def read_conditions(
    xml: Optional[Path], args: argparse.Namespace, fields: Sequence[str], device: torch.device,
    texts_path: Optional[Path] = None,
) -> Tuple[ConditionBatch, Dict[str, Any]]:
    """读取 XML 或完整的显式 CLI 条件，不混用两种来源补默认值。

    Args:
        xml: 用户指定/自动配对的 XML；反事实运行可传另一张真实样本的 XML。
        args: 手工条件字段；使用 XML 时这些字段必须全部省略。
        fields: 当前条件编码器必需字段。
        device: 模型所在设备。
        texts_path: F组的原始描述表；其他模式保持原有解析。

    Returns:
        单图条件张量，以及包含来源和原始值的记录。
    """
    manual = {field: getattr(args, field) for field in LEGACY_FIELDS
              if getattr(args, field) is not None}
    source_fields = LEGACY_FIELDS if tuple(fields) == (TEXT_CONDITION_FIELD,) else fields
    if xml is not None:
        if manual:
            raise ValueError("--xml 和手工条件参数不能同时使用；请明确选择条件来源。")
        if not xml.is_file():
            raise FileNotFoundError(f"条件 XML 不存在：{xml}")
        root = ET.parse(xml).getroot()
        raw = {field: root.findtext(XML_FIELDS[field]) for field in source_fields}
        source: Dict[str, Any] = {"xml": str(xml.resolve()), "xml_sha256": file_sha256(xml)}
    else:
        raw = manual
        source = {"source": "explicit CLI conditions"}
    missing = [field for field in source_fields
               if field not in raw or raw[field] is None or not str(raw[field]).strip()]
    if missing:
        raise ValueError(f"缺少真实条件 {missing}；该模型需要 {list(fields)}。"
                         "请提供 --xml 或完整的手工条件参数，不会补默认值。")
    tensors = condition_tensors(raw, fields, device, texts_path)
    source["raw_conditions"] = {field: raw[field] for field in source_fields}
    source["model_conditions"] = {
        field: value.cpu().tolist() if isinstance(value, torch.Tensor) else value
        for field, value in tensors.items()
    }
    return tensors, source


def prepare_images(pair: InputPair, cfg: YAMLConfig, device: torch.device) -> torch.Tensor:
    """应用配置的确定性验证预处理，不将目标框或类别送入模型。

    Args:
        pair: 对应同一场景的 RGB/IR 文件，两张图片必须同尺寸。
        cfg: 模块一验证 transforms 配置。
        device: 前向设备。

    Returns:
        [1,6,H,W] float32，前三通道 IR、后三通道 RGB。
    """
    with Image.open(pair.ir) as source:
        ir = source.convert("RGB")
    with Image.open(pair.rgb) as source:
        rgb = source.convert("RGB")
    if ir.size != rgb.size:
        raise ValueError(f"RGB/IR 尺寸不一致：RGB={rgb.size}, IR={ir.size}；请先提供配准图像。")
    specification = cfg.yaml_cfg["val_dataloader"]["dataset"]["transforms"]
    ops = specification.get("ops")
    if specification.get("type") != "Compose" or not isinstance(ops, list):
        raise ValueError("工具要求配置具有明确的 Compose 验证预处理。")
    if any(not isinstance(op, dict) or op.get("type") not in VALID_TRANSFORMS for op in ops):
        raise ValueError("验证预处理含工具未核验的算子；不会忽略或替换这些算子。")
    policy = specification.get("policy")
    if policy is not None and policy.get("name") != "default":
        raise ValueError("单图可视化只支持确定性的 default 验证预处理。")
    # Compose 的算子就是配置中的原算子；None target 避免创建虚构 GT。
    transforms = Compose(ops=copy.deepcopy(ops))
    transformed = transforms({"ir": ir, "rgb": rgb}, None)[0]
    if not isinstance(transformed, torch.Tensor) or transformed.ndim != 3 or transformed.shape[0] != 6:
        raise ValueError("验证预处理未生成 [6,H,W] 图像对。")
    if transformed.dtype != torch.float32 or not torch.isfinite(transformed).all():
        raise ValueError("图像预处理结果不是有限的 float32 张量。")
    if ((transformed < 0) | (transformed > 1)).any():
        raise ValueError("RGB叠加底图需要[0,1]图像；请使用scale=True，不会静默裁剪显示。")
    return transformed.unsqueeze(0).to(device)


def load_model(args: argparse.Namespace, cfg: YAMLConfig, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """严格加载整网权重，并保留正常检测推理路径。

    Args:
        args: checkpoint 路径及明确的 EMA/model 权重选择。
        cfg: 模块一模型配置；只在内存里关闭重复 ImageNet 初始化。
        device: CUDA 设备。

    Returns:
        eval 模型，以及权重文件/模型类型/epoch 记录。
    """
    model_name = cfg.yaml_cfg.get("model")
    if model_name != "RTv4_PG":
        raise ValueError("此工具用于 RTv4_PG 模块一配置。")
    backbone_name = cfg.yaml_cfg[model_name]["backbone"]
    if not isinstance(backbone_name, str) or backbone_name not in cfg.yaml_cfg:
        raise ValueError("配置没有可识别的模块一主干。")
    cfg.yaml_cfg[backbone_name]["pretrained"] = False
    model = cfg.model.to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if args.weights == "ema":
        if not isinstance(checkpoint.get("ema"), dict) or "module" not in checkpoint["ema"]:
            raise ValueError("checkpoint 没有 EMA 权重；若确需原始模型，请明确指定 --weights model。")
        state = checkpoint["ema"]["module"]
    else:
        if "model" not in checkpoint:
            raise ValueError("checkpoint 没有 model 权重。")
        state = checkpoint["model"]
    model.load_state_dict(state, strict=True)
    if not all(hasattr(model.backbone, key) for key in ("condition_encoder", "exchange_layers", "fusion_layers")):
        raise ValueError("主干没有完整的模块一结构。")
    metadata = {"checkpoint": str(args.checkpoint.resolve()), "weights": args.weights,
                "checkpoint_sha256": file_sha256(args.checkpoint), "epoch": checkpoint.get("last_epoch"),
                "backbone": type(model.backbone).__name__}
    return model, metadata


class Module1Trace:
    """临时只读 hook，保存当前前向的引用；退出时移除全部 hook。"""

    def __init__(self, backbone: nn.Module) -> None:
        """准备观察器，不改变模型结构、参数或消融开关。

        Args:
            backbone: 三尺度模块一主干。
        """
        self.backbone = backbone
        self.cache: Dict[str, Tuple[Tuple[Any, ...], Any]] = {}
        self.handles: list[RemovableHandle] = []

    def capture(self, name: str, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        """保存真实输入输出；返回 None，绝不替换模块输出。

        Args:
            name: 唯一观察点名称。
            module: 当前被观察模块。
            inputs: 实际前向输入元组。
            output: 实际张量或张量元组，不跨前向复用。
        """
        self.cache[name] = (inputs, output)

    def __enter__(self) -> Module1Trace:
        """注册特征、门控、消息和融合分数的观察点。

        Returns:
            当前观察器；捕获张量随当前 inference_mode 前向生成。
        """
        observed = {"condition": self.backbone.condition_encoder,
                    "raw_rgb": self.backbone.backbone_rgb, "raw_ir": self.backbone.backbone_ir}
        for index, (exchange, fusion) in enumerate(zip(self.backbone.exchange_layers, self.backbone.fusion_layers)):
            prefix = f"S{index + 3}"
            observed[f"{prefix}.exchange"] = exchange
            observed[f"{prefix}.fusion"] = fusion
            observed[f"{prefix}.out"] = fusion.out_projection
            for branch in ("rgb", "ir"):
                for suffix in ("work", "film", "state"):
                    observed[f"{prefix}.{branch}_{suffix}"] = getattr(exchange, f"{branch}_{suffix}")
                score = getattr(fusion, f"{branch}_score")
                observed[f"{prefix}.{branch}_score"] = score
                for suffix in ("query", "key", "visual_score", "condition_bias"):
                    observed[f"{prefix}.{branch}_score.{suffix}"] = getattr(score, suffix)
            for direction in ("rgb_to_ir", "ir_to_rgb"):
                message = getattr(exchange, direction)
                observed[f"{prefix}.{direction}"] = message
        for name, module in observed.items():
            self.handles.append(module.register_forward_hook(partial(self.capture, name)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """无论前向成功或失败均移除 hook。

        Args:
            exc_type: 上下文异常类型。
            exc: 异常实例。
            traceback: 异常回溯。
        """
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def tensor_array(value: torch.Tensor) -> np.ndarray:
    """转为有限的 CPU float32 数组，不修改原张量。

    Args:
        value: 任意形状的被观察张量。

    Returns:
        与输入同形状的独立 numpy 数组。
    """
    if not torch.isfinite(value).all():
        raise ValueError("观察到 NaN/Inf，拒绝生成误导热图。")
    return value.detach().float().cpu().numpy().copy()


def spatial_map(value: torch.Tensor, reduction: str = "mean") -> np.ndarray:
    """将单图 [1,C,H,W] 显示成原生网格 [H,W]。

    Args:
        value: 空间响应张量，单图 batch=1。
        reduction: mean 用于门控，rms 用于有正负值的特征/注入消息。

    Returns:
        未经 min-max 归一化的原生网格 float32 map。
    """
    if value.ndim != 4 or value.shape[0] != 1:
        raise ValueError("空间可视化需要 [1,C,H,W] 张量。")
    if reduction not in ("mean", "rms"):
        raise ValueError(f"不支持 reduction={reduction}。")
    reduced = value.float().mean(dim=1) if reduction == "mean" else value.float().square().mean(dim=1).sqrt()
    return tensor_array(reduced[0])


def array_statistics(value: np.ndarray) -> Dict[str, float]:
    """统计原生网格/张量，不按上采样后的显示像素加权。

    Args:
        value: 有限非空数组。

    Returns:
        最小/最大/均值/标准差及 10%、50%、90% 分位数。
    """
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError("统计输入必须是非空有限数组。")
    return {"min": float(value.min()), "max": float(value.max()), "mean": float(value.mean()),
            "std": float(value.std()), "q10": float(np.quantile(value, 0.1)),
            "median": float(np.quantile(value, 0.5)), "q90": float(np.quantile(value, 0.9))}


def extract_maps(trace: Module1Trace) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """从真实 hook 得到三尺度 map，并核对可视化权重确实生成了融合输入。

    Args:
        trace: 一次真实 backbone 前向的观察结果。

    Returns:
        NPZ 数组字典，以及各尺度统计；CG-DMF 为 [H,W]，原始门控为 [C,H,W]。
    """
    cache, backbone = trace.cache, trace.backbone
    dynamic, exchange_enabled = bool(backbone.dynamic_fusion), bool(backbone.enable_exchange)
    film_enabled = bool(getattr(backbone, "enable_film", True))
    encoded_condition = cache["condition"][1]
    if dynamic:
        effective_condition = cache["S3.fusion"][0][2]
    elif film_enabled:
        effective_condition = cache["S3.rgb_film"][0][0]
    elif exchange_enabled:
        effective_condition = cache["S3.exchange"][0][2]
    else:
        effective_condition = (torch.zeros_like(encoded_condition)
                               if backbone.condition_mode == "zero" else encoded_condition)
    arrays = {"encoded_condition": tensor_array(encoded_condition[0]),
              "effective_condition": tensor_array(effective_condition[0])}
    statistics: Dict[str, Any] = {}
    for index, fusion in enumerate(backbone.fusion_layers):
        stage = f"S{index + 3}"
        local: Dict[str, np.ndarray] = {}
        works = {branch: cache[f"{stage}.{branch}_work"][1] for branch in ("rgb", "ir")}
        # FiLM 是函数式运算；根据真实 hook 的 S、gamma/beta 重建 Z，原公式没有改动。
        updated = {}
        for branch in ("rgb", "ir"):
            if film_enabled:
                gamma, beta = cache[f"{stage}.{branch}_film"][1].chunk(2, dim=1)
                updated[branch] = (1 + gamma[:, :, None, None]) * works[branch] + beta[:, :, None, None]
            else:
                updated[branch] = works[branch]
            local[f"raw_{branch}_rms"] = spatial_map(cache[f"raw_{branch}"][1][index], "rms")
            local[f"work_{branch}_rms"] = spatial_map(works[branch], "rms")
        if exchange_enabled:
            actual_rgb, actual_ir = cache[f"{stage}.exchange"][1]
            for branch in ("rgb", "ir"):
                _, support, need = cache[f"{stage}.{branch}_state"][1]
                local[f"{branch}_support"] = spatial_map(support)
                local[f"{branch}_need"] = spatial_map(need)
            for direction, receiver in (("rgb_to_ir", "ir"), ("ir_to_rgb", "rgb")):
                gate, message = cache[f"{stage}.{direction}"][1]
                injection = gate * message
                local[f"{direction}_gate"] = spatial_map(gate)
                local[f"{direction}_gate_channels"] = tensor_array(gate[0])
                local[f"{direction}_injection_rms"] = spatial_map(injection, "rms")
                updated[receiver] = updated[receiver] + injection
            if not torch.allclose(updated["rgb"], actual_rgb, atol=1e-5, rtol=1e-4) or not torch.allclose(
                updated["ir"], actual_ir, atol=1e-5, rtol=1e-4
            ):
                raise ValueError(f"{stage} 的观察结果不能重现实际双向更新。")
            updated = {"rgb": actual_rgb, "ir": actual_ir}
        for branch in ("rgb", "ir"):
            local[f"updated_{branch}_rms"] = spatial_map(updated[branch], "rms")
        if dynamic:
            rgb_score, ir_score = (cache[f"{stage}.{branch}_score"][1] for branch in ("rgb", "ir"))
            rgb_weight = (rgb_score - ir_score).sigmoid()
            components = {}
            for branch in ("rgb", "ir"):
                score_name = f"{stage}.{branch}_score"
                query = cache[f"{score_name}.query"][1].float()
                key = cache[f"{score_name}.key"][1].float()[:, :, None, None]
                visual = cache[f"{score_name}.visual_score"][1].float()
                baseline = cache[f"{score_name}.condition_bias"][1].float()[:, :, None, None]
                compatibility = (query * key).sum(dim=1, keepdim=True) / getattr(fusion, f"{branch}_score").scale
                if not torch.allclose(visual + baseline + compatibility, cache[score_name][1], atol=1e-5, rtol=1e-4):
                    raise ValueError(f"{stage}/{branch} 的分数分解与真实前向不一致。")
                components[branch] = (visual, baseline.expand_as(visual), compatibility)
            for position, name in enumerate(("visual", "condition_bias", "compatibility")):
                local[f"score_delta_{name}"] = spatial_map(components["rgb"][position] - components["ir"][position])
            local["score_delta_total"] = spatial_map(rgb_score - ir_score)
        else:
            # equal_fusion 的实际代码使用固定各0.5；这个 map 明确标记为固定消融。
            rgb_weight = torch.full_like(updated["rgb"][:, :1], 0.5)
        ir_weight = 1 - rgb_weight
        weighted = rgb_weight * updated["rgb"] + ir_weight * updated["ir"]
        if not torch.allclose(weighted, cache[f"{stage}.out"][0][0], atol=1e-5, rtol=1e-4):
            raise ValueError(f"{stage} 权重无法重现实际融合输入，拒绝导出错误热图。")
        local["rgb_weight"], local["ir_weight"] = spatial_map(rgb_weight), spatial_map(ir_weight)
        local["fused_rms"] = spatial_map(cache[f"{stage}.out"][1], "rms")
        statistics[stage] = {"grid_shape": list(local["rgb_weight"].shape),
                             "maps": {name: array_statistics(array) for name, array in local.items()
                                      if not name.endswith("_channels")},
                             "channel_gate_statistics": {name: array_statistics(array) for name, array in local.items()
                                                         if name.endswith("_channels")},
                             "ir_preferred_fraction": float((local["ir_weight"] > 0.5).mean())}
        arrays.update({f"{stage}_{name}": value for name, value in local.items()})
    return arrays, statistics


def run_forward(model: nn.Module, images: torch.Tensor, conditions: Mapping[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """运行一次正常整网推理，hook 模块一，不改变参数、开关或 BN 状态。

    Args:
        model: 已严格加载权重并切换 eval 的 RTv4_PG。
        images: [1,6,H,W] float32。
        conditions: 当前模型声明字段对应的单图条件张量。

    Returns:
        原生分辨率的观察数组和统计。
    """
    with torch.inference_mode(), Module1Trace(model.backbone) as trace:
        outputs = model(images, conditions=conditions)
        for field in ("pred_logits", "pred_boxes"):
            if field not in outputs or not torch.isfinite(outputs[field]).all():
                raise ValueError(f"正常整网推理的 {field} 缺失或含 NaN/Inf。")
        return extract_maps(trace)


def draw_map(ax: Any, value: Optional[np.ndarray], title: str, limit: Optional[float] = None, signed: bool = False) -> None:
    """显示原生网格；控制响应使用统一0–1，特征/分数使用数值色标。

    Args:
        ax: Matplotlib 子图。
        value: [H,W] map；缺失响应用 None 显示 N/A。
        title: 说明响应含义与通道聚合方式。
        limit: 对照面板共享的上界，不做每张图的 min-max 伪归一化。
        signed: 有符号差值使用关于零对称的色标。
    """
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    if value is None:
        ax.text(0.5, 0.5, "N/A: branch disabled", ha="center", va="center", transform=ax.transAxes)
        return
    maximum = limit if limit is not None else float(np.abs(value).max())
    maximum = max(maximum, 1e-8)
    artist = ax.imshow(value, cmap="coolwarm" if signed else "viridis",
                       vmin=-maximum if signed else 0, vmax=maximum, interpolation="nearest")
    ax.figure.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)


def save_stage_figures(arrays: Mapping[str, np.ndarray], images: torch.Tensor, output: Path, label: str) -> None:
    """为每个尺度保存权重、门控、消息和前后特征响应图。

    Args:
        arrays: 当前真实前向导出的原生 map。
        images: [1,6,H,W] 验证预处理图像。
        output: 已创建的独立结果目录。
        label: real/changed_conditions，仅用于注明条件来源。
    """
    rgb = tensor_array(images[0, 3:]).transpose(1, 2, 0)
    ir = tensor_array(images[0, :3]).transpose(1, 2, 0)
    # 先给一张简洁总览；各尺度完整CM/CG诊断另存S3/S4/S5.png。
    overview, panels = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    overview.suptitle(f"{label}: actual module1 weights over RGB (display interpolation)")
    for row, stage in enumerate(("S3", "S4", "S5")):
        for column, name in enumerate((None, "rgb_weight", "ir_weight")):
            ax = panels[row, column]
            ax.imshow(rgb)
            ax.axis("off")
            if name is None:
                ax.set_title(f"{stage}: RGB input")
                continue
            artist = ax.imshow(arrays[f"{stage}_{name}"],
                               extent=(-0.5, rgb.shape[1] - 0.5, rgb.shape[0] - 0.5, -0.5),
                               cmap="viridis", vmin=0, vmax=1, interpolation="bilinear", alpha=0.55)
            ax.set_title(f"{stage}: {'M_RGB' if name == 'rgb_weight' else 'M_IR'} on RGB")
            overview.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
    overview.savefig(output / "weights_on_rgb.png", dpi=130)
    plt.close(overview)
    for stage in ("S3", "S4", "S5"):
        get = lambda name: arrays.get(f"{stage}_{name}")
        fig, axes = plt.subplots(4, 4, figsize=(17, 14), constrained_layout=True)
        fig.suptitle(f"{stage} / {label}: native maps; feature response = channel RMS", fontsize=13)
        for ax, picture, name in ((axes[0, 0], rgb, "RGB input"), (axes[0, 1], ir, "IR input")):
            ax.imshow(np.clip(picture, 0, 1)); ax.set_title(name); ax.axis("off")
        # 所有条件权重/门控统一叠加到可见光 RGB 底图，避免更换底图误判位置。
        overlays = ((axes[0, 2], "rgb_weight", "M_RGB on RGB"),
                    (axes[0, 3], "ir_weight", "M_IR on RGB"),
                    (axes[1, 0], "rgb_to_ir_gate", "RGB -> IR gate mean on RGB"),
                    (axes[1, 1], "ir_to_rgb_gate", "IR -> RGB gate mean on RGB"))
        for ax, name, title in overlays:
            ax.imshow(np.clip(rgb, 0, 1))
            value = get(name)
            if value is not None:
                artist = ax.imshow(value, extent=(-0.5, rgb.shape[1] - 0.5,
                                   rgb.shape[0] - 0.5, -0.5), cmap="viridis", vmin=0, vmax=1,
                                   interpolation="bilinear", alpha=0.55)
                fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
            else:
                ax.text(0.5, 0.5, "N/A: exchange disabled", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title + " (display interpolation)", fontsize=10)
            ax.axis("off")
        for ax, name, title in ((axes[1, 2], "rgb_weight", "M_RGB / native grid"),
                                 (axes[1, 3], "ir_weight", "M_IR / native grid")):
            draw_map(ax, get(name), title, limit=1.0)
        raw_limit = max(float(get(f"raw_{branch}_rms").max()) for branch in ("rgb", "ir"))
        updated_limit = max(float(get(f"updated_{branch}_rms").max()) for branch in ("rgb", "ir"))
        for ax, name, title, limit in zip(axes[2], ("raw_rgb_rms", "raw_ir_rms", "updated_rgb_rms", "updated_ir_rms"),
                                         ("Raw RGB features", "Raw IR features", "Updated RGB features", "Updated IR features"),
                                         (raw_limit, raw_limit, updated_limit, updated_limit)):
            draw_map(ax, get(name), title, limit=limit)
        injections = [get(f"{direction}_injection_rms") for direction in ("rgb_to_ir", "ir_to_rgb")]
        injection_limit = max((float(value.max()) for value in injections if value is not None), default=1.0)
        for ax, name, title in zip(axes[3], ("rgb_to_ir_injection_rms", "ir_to_rgb_injection_rms", "fused_rms", "score_delta_total"),
                                  ("RGB -> IR injected message RMS", "IR -> RGB injected message RMS", "Fused features RMS", "s_RGB - s_IR")):
            draw_map(ax, get(name), title, limit=injection_limit if "injection" in name else None,
                     signed=name == "score_delta_total")
        fig.savefig(output / f"{stage}.png", dpi=130)
        plt.close(fig)
        if get("score_delta_total") is not None:
            fig, axes = plt.subplots(1, 4, figsize=(17, 4), constrained_layout=True)
            names = ("visual", "condition_bias", "compatibility", "total")
            shared = max(float(np.abs(get(f"score_delta_{name}")).max()) for name in names)
            for ax, name in zip(axes, names):
                draw_map(ax, get(f"score_delta_{name}"), f"RGB-IR: {name}", limit=shared, signed=True)
            fig.savefig(output / f"{stage}_scores.png", dpi=130)
            plt.close(fig)


def save_comparison(real: Mapping[str, np.ndarray], changed: Mapping[str, np.ndarray], output: Path) -> Dict[str, Any]:
    """固定图像、只改变条件，比较三个尺度的 IR 选择权重。

    Args:
        real: 原条件前向的 map。
        changed: 用户指定另一份真实 XML 条件的前向 map。
        output: 独立结果目录。

    Returns:
        原生权重差的统计，差值是 changed - real，不是检测 AP 增益。
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    fig.suptitle("Same RGB/IR images, only condition metadata changed (inference intervention)")
    statistics: Dict[str, Any] = {}
    for row, stage in enumerate(("S3", "S4", "S5")):
        original, altered = real[f"{stage}_ir_weight"], changed[f"{stage}_ir_weight"]
        delta = altered - original
        draw_map(axes[row, 0], original, f"{stage}: real M_IR", limit=1)
        draw_map(axes[row, 1], altered, f"{stage}: changed conditions M_IR", limit=1)
        draw_map(axes[row, 2], delta, f"{stage}: M_IR change", limit=1, signed=True)
        statistics[stage] = {"signed_change": array_statistics(delta),
                             "mean_absolute_change": float(np.abs(delta).mean()),
                             "max_absolute_change": float(np.abs(delta).max())}
    fig.savefig(output / "condition_comparison.png", dpi=130)
    plt.close(fig)
    statistics["effective_condition_l2_change"] = float(np.linalg.norm(
        changed["effective_condition"] - real["effective_condition"]))
    return statistics


def build_parser() -> argparse.ArgumentParser:
    """建立双光图像对可视化 CLI。

    Returns:
        默认 CUDA、EMA 权重的参数解析器。
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="ATR-UMOD IR 图片，自动找同名 RGB 和 XML")
    group.add_argument("--ir-image", type=Path, help="显式 IR 图片，外部图片还需 --rgb-image")
    parser.add_argument("--rgb-image", type=Path)
    parser.add_argument("--xml", type=Path, help="本图真实条件 XML，与手工条件参数互斥")
    parser.add_argument("--compare-xml", type=Path, help="保持图像不变，改用此 XML 的条件再次前向")
    for name in ("height", "angle"):
        parser.add_argument(f"--{name}", type=float, help="真实物理值，height=米、angle=度")
    parser.add_argument("--time", help="真实时刻 HH:MM/ISO 时间")
    for name in CATEGORY_COUNTS:
        parser.add_argument(f"--{name}", type=int, help=f"XML 原始编号：0–{CATEGORY_COUNTS[name] - 1}")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--device", default="cuda", help="CUDA 设备，例如 cuda:0；不自动回退 CPU")
    parser.add_argument("--output-dir", type=Path, help="独立输出目录，已存在时拒绝覆盖")
    return parser


def main() -> None:
    """加载权重并导出双光图像对观察，最后写 JSON 标记导出成功。"""
    args = build_parser().parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("此工具要求可用 CUDA GPU，不自动回退 CPU。")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{args.checkpoint}")
    cfg = YAMLConfig(str(args.config))
    pair = resolve_inputs(args, cfg)
    output = args.output_dir or REPO_ROOT / "outputs/module1_visualizations" / args.checkpoint.parent.name / pair.ir.stem
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"结果目录已存在：{output}；请指定新的 --output-dir。")
    model, metadata = load_model(args, cfg, device)
    fields = required_condition_fields(model.backbone)
    texts_path = Path(model.backbone.condition_encoder.texts_path) if fields == (TEXT_CONDITION_FIELD,) else None
    conditions, provenance = read_conditions(pair.xml, args, fields, device, texts_path)
    images = prepare_images(pair, cfg, device)
    arrays, statistics = run_forward(model, images, conditions)
    changed_arrays = changed_statistics = changed_provenance = None
    if args.compare_xml is not None:
        comparison_args = copy.copy(args)
        for field in LEGACY_FIELDS:
            setattr(comparison_args, field, None)
        changed_conditions, changed_provenance = read_conditions(
            args.compare_xml.expanduser().resolve(), comparison_args, fields, device, texts_path)
        changed_arrays, changed_statistics = run_forward(model, images, changed_conditions)
    metadata.update({"config": str(args.config.resolve()), "config_sha256": file_sha256(args.config),
                     "ir_image": str(pair.ir), "rgb_image": str(pair.rgb), "auto_pair_split": pair.split,
                     "condition_fields": list(fields), "condition_dim": int(arrays["effective_condition"].size),
                     "condition_mode": model.backbone.condition_mode,
                     "enable_exchange": model.backbone.enable_exchange, "dynamic_fusion": model.backbone.dynamic_fusion,
                     "enable_film": getattr(model.backbone, "enable_film", True),
                     "condition_prior": getattr(model.backbone, "condition_prior", "dt_cpn"),
                     "input_shape": list(images.shape), "device": str(device), "optimizer_steps": 0,
                     "conditions": provenance, "stages": statistics,
                     "interpretation": "Task contribution after exchange; not calibrated sensor quality or AP gain."})
    output.mkdir(parents=True, exist_ok=False)
    real_output = output / "real"
    real_output.mkdir()
    np.savez_compressed(real_output / "maps.npz", **arrays)
    save_stage_figures(arrays, images, real_output, "real")
    if changed_arrays is not None:
        altered_output = output / "changed_conditions"
        altered_output.mkdir()
        np.savez_compressed(altered_output / "maps.npz", **changed_arrays)
        save_stage_figures(changed_arrays, images, altered_output, "same images / changed metadata")
        metadata["changed_conditions"] = {"conditions": changed_provenance, "stages": changed_statistics}
        metadata["comparison"] = save_comparison(arrays, changed_arrays, output)
    (output / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"已导出三尺度权重/门控/特征响应：{output}")
    for stage, record in statistics.items():
        print(f"{stage}: mean M_IR={record['maps']['ir_weight']['mean']:.4f}, "
              f"IR-preferred positions={record['ir_preferred_fraction']:.1%}")


if __name__ == "__main__":
    main()
