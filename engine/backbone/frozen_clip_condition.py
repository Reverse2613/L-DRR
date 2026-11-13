"""三属性联合 CLIP 条件：原始512维、无学习参数、无降维。

每个 weather/illumination/location 组合对应一整段固定模板文本。
CLIP 离线编码整段文本；本模块只查组合缓存，不运行 CLIP，
不读取几何/时间，也不将三个单属性向量相加或拼接。
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Tuple, Union

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXTS_PATH = REPO_ROOT / "configs/dataset/atrumod_condition_texts.json"
JOINT_FIELDS = ("weather", "illumination", "location")
JOINT_COUNTS = {"weather": 10, "illumination": 7, "location": 11}
JOINT_CONDITION_TEMPLATE = "weather: {weather}; illumination: {illumination}; location: {location}."
JOINT_INDEX_FORMULA = "(weather * 7 + illumination) * 11 + location"
JOINT_FEATURE_FILENAME = "joint_condition_anchor.pt"
JOINT_CACHE_KIND = "joint_three_attribute_clip"
CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
CLIP_FEATURE_DIM = 512
JOINT_ROW_COUNT = 10 * 7 * 11


def load_joint_condition_texts(path: Union[str, Path]) -> Dict[str, List[str]]:
    """核验并读取原始类别描述，保持原文和整数编号顺序。

    Args:
        path: schema_version=1 的 ATR-UMOD 条件文本 JSON。

    Returns:
        weather 10条、illumination 7条、location 11条描述；第 k 行对应
        原始编号 k。不修改原文的大小写、空格或拼写。

    Raises:
        FileNotFoundError: 原文文件不存在。
        ValueError: 版本、编号、行数或文本不符合契约。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("原始条件文本必须是 schema_version=1。")
    result: Dict[str, List[str]] = {}
    for field in JOINT_FIELDS:
        rows, count = data.get(field), JOINT_COUNTS[field]
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError(f"{field}原始文本必须有{count}行。")
        if any(
            not isinstance(row, dict) or type(row.get("id")) is not int
            or row["id"] != index or not isinstance(row.get("text"), str)
            or not row["text"].strip()
            for index, row in enumerate(rows)
        ):
            raise ValueError(f"{field}必须按原始整数编号排列且文本非空。")
        result[field] = [row["text"] for row in rows]
    return result


def build_joint_texts(texts: Mapping[str, List[str]]) -> Tuple[List[str], List[List[int]]]:
    """按确定的组合编号生成770段联合条件文本。

    Args:
        texts: 已核验的三属性原文列表，长度分别为10、7、11。

    Returns:
        (joint_texts, combination_ids)：770段完整模板文本及每行的
        [weather_id, illumination_id, location_id]。行号为
        (weather * 7 + illumination) * 11 + location，location 变化最快。
    """
    joint_texts: List[str] = []
    combination_ids: List[List[int]] = []
    for weather in range(JOINT_COUNTS["weather"]):
        for illumination in range(JOINT_COUNTS["illumination"]):
            for location in range(JOINT_COUNTS["location"]):
                joint_texts.append(JOINT_CONDITION_TEMPLATE.format(
                    weather=texts["weather"][weather],
                    illumination=texts["illumination"][illumination],
                    location=texts["location"][location],
                ))
                combination_ids.append([weather, illumination, location])
    return joint_texts, combination_ids


def joint_texts_sha256(texts: List[str]) -> str:
    """计算联合文本及其行顺序的确定性校验值。

    Args:
        texts: 按组合行号排列的770段文本。

    Returns:
        UTF-8、无 ASCII 转义、紧凑 JSON 列表内容的 SHA-256。
    """
    payload = json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_frozen_joint_clip_table(
    anchor_dir: Union[str, Path],
    texts_path: Union[str, Path] = DEFAULT_TEXTS_PATH,
) -> torch.Tensor:
    """核验联合文本缓存并读取未额外变换的原始512维 CLIP 特征。

    Args:
        anchor_dir: build_joint_condition_anchors.py 生成的联合缓存目录。
            旧64维表或旧的28行单属性512维缓存不能代替此缓存。
        texts_path: 用于核验类别编号和原文的 JSON。

    Returns:
        无梯度的 CPU float32 表 [770,512]；冻结 CLIP 对每整段文本输出
        的 text_embeds，未额外 L2 归一化、随机投影或训练适配。

    Raises:
        FileNotFoundError: 缺少 manifest、源文本或联合缓存。
        ValueError: 模板、编号、原文、CLIP 来源或张量校验失败。
    """
    directory, source = Path(anchor_dir), Path(texts_path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("联合条件缓存 manifest 必须是 schema_version=1。")
    if manifest.get("cache_kind") != JOINT_CACHE_KIND:
        raise ValueError("需要770行三属性联合文本缓存，不能使用旧单属性缓存。")
    if (manifest.get("template") != JOINT_CONDITION_TEMPLATE
            or manifest.get("condition_counts") != JOINT_COUNTS
            or manifest.get("condition_dim") != CLIP_FEATURE_DIM
            or manifest.get("id_order") != list(JOINT_FIELDS)
            or manifest.get("index_formula") != JOINT_INDEX_FORMULA
            or manifest.get("feature_transform") != "none"):
        raise ValueError("联合文本模板、组合编号或特征变换记录不符合本实验。")
    if manifest.get("texts_file_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("原始条件文本与联合缓存的来源校验值不一致。")
    model_record = manifest.get("model")
    if not isinstance(model_record, dict) or model_record.get("model_id") != CLIP_MODEL_ID:
        raise ValueError("本实验需要 CLIP ViT-B/16 的原始512维联合文本特征。")
    source_texts = load_joint_condition_texts(source)
    joint_texts, combination_ids = build_joint_texts(source_texts)
    if (manifest.get("texts_by_id") != source_texts
            or manifest.get("combination_ids") != combination_ids
            or manifest.get("joint_texts_by_index") != joint_texts
            or manifest.get("joint_texts_sha256") != joint_texts_sha256(joint_texts)):
        raise ValueError("联合缓存行与原始条件编号/模板文本不对应。")
    artifacts = manifest.get("artifacts")
    record = artifacts.get(JOINT_FEATURE_FILENAME) if isinstance(artifacts, dict) else None
    expected_shape = (JOINT_ROW_COUNT, CLIP_FEATURE_DIM)
    if not isinstance(record, dict) or record.get("shape") != list(expected_shape):
        raise ValueError("manifest 中联合 CLIP 特征形状必须是[770,512]。")
    feature_path = directory / JOINT_FEATURE_FILENAME
    if record.get("sha256") != hashlib.sha256(feature_path.read_bytes()).hexdigest():
        raise ValueError("联合512维文本特征文件校验值不符，拒绝加载。")
    features = torch.load(feature_path, map_location="cpu", weights_only=True)
    if not isinstance(features, torch.Tensor) or features.shape != expected_shape:
        raise ValueError("联合 CLIP 特征必须是[770,512]张量。")
    if features.dtype != torch.float32 or not torch.isfinite(features).all():
        raise ValueError("联合 CLIP 特征必须是有限 float32 值。")
    if (torch.linalg.vector_norm(features, dim=1) <= 1e-7).any():
        raise ValueError("联合 CLIP 特征含零向量或数值近零的行。")
    return features.detach().clone()


class FrozenCLIPConditionPrior(nn.Module):
    """三属性联合 CLIP 查表，无可学习条件表示。

    Attributes:
        required_fields: 唯一读取的三个类别字段，额外几何/时间字段被忽略。
        output_dim: 固定为512，不经过64维或128维的投影。
        joint_anchor: float32 原始联合文本表 [770,512]，持久冻结 buffer。
        ema_copy_buffers: 显式要求 EMA 精确复制缓存，避免浮点滑动平均漂移。
    """

    required_fields = JOINT_FIELDS
    output_dim = CLIP_FEATURE_DIM
    ema_copy_buffers = ("joint_anchor",)

    def __init__(
        self, anchor_dir: Union[str, Path],
        texts_path: Union[str, Path] = DEFAULT_TEXTS_PATH,
    ) -> None:
        """加载不进入优化器的联合条件表。

        Args:
            anchor_dir: 770行联合缓存及 manifest 所在目录。
            texts_path: 用于核验缓存编号与来源的官方属性文本 JSON。
        """
        super().__init__()
        self.anchor_dir = str(Path(anchor_dir).resolve())
        self.texts_path = str(Path(texts_path).resolve())
        self.template = JOINT_CONDITION_TEMPLATE
        self.register_buffer("joint_anchor", load_frozen_joint_clip_table(anchor_dir, texts_path))

    def _validate_conditions(self, conditions: Mapping[str, torch.Tensor]) -> None:
        """核验三离散条件的 batch、设备、类型和原始编号范围。

        Args:
            conditions: 三字段各 [B]、int64；额外字段不被读取。

        Raises:
            ValueError: 字段缺失、batch 为空或张量/编号不合法。
        """
        if not isinstance(conditions, Mapping):
            raise ValueError("冻结 CLIP 条件必须是三离散属性的张量字典。")
        for field in self.required_fields:
            if field not in conditions or not isinstance(conditions[field], torch.Tensor):
                raise ValueError(f"冻结 CLIP 条件缺少张量字段：{field}。")
        weather = conditions["weather"]
        if weather.ndim != 1 or weather.shape[0] == 0:
            raise ValueError("weather 必须是非空 [B] 张量。")
        if self.joint_anchor.dtype != torch.float32:
            raise ValueError("冻结 CLIP 表必须保持 float32；请使用 AMP 而非 model.half()。")
        batch, device = weather.shape[0], self.joint_anchor.device
        for field in self.required_fields:
            value, count = conditions[field], JOINT_COUNTS[field]
            if value.shape != (batch,) or value.dtype != torch.int64 or value.device != device:
                raise ValueError(f"{field}需要形状[{batch}]、类型int64、设备{device}。")
            if ((value < 0) | (value >= count)).any():
                raise ValueError(f"{field}原始编号必须在0到{count - 1}之间。")

    @torch.no_grad()
    def forward(self, conditions: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """读取整段联合模板文本对应的原始512维冻结向量。

        Args:
            conditions: 三类别字段各 [B]、int64，与缓存同设备。

        Returns:
            c [B,512]、float32、requires_grad=False。无残差、归一化或
            学习层；相同三属性组合的向量在 train/eval 下完全相同。
            后续 CM-CME/CG-DMF 的条件响应仍可学习。
        """
        self._validate_conditions(conditions)
        index = (conditions["weather"] * JOINT_COUNTS["illumination"]
                 + conditions["illumination"]) * JOINT_COUNTS["location"] + conditions["location"]
        return self.joint_anchor.index_select(0, index)
