"""F组：完整六条件原文→离线冻结CLIP表→检测器条件向量。

模型只接收图像级condition_text列表并查表，不运行CLIP或读取XML目标标注。
字符串保留在CPU；查表索引在buffer同设备构造，输出固定512维浮点向量。
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Union

import torch
import torch.nn as nn

from ..data.condition_texts import (
    CONDITION_COUNTS, SIX_CONDITION_TEMPLATE, SIX_XML_FIELDS,
    build_six_condition_text, load_condition_descriptions,
)
from .frozen_clip_condition import (
    CLIP_FEATURE_DIM, CLIP_MODEL_ID, DEFAULT_TEXTS_PATH, joint_texts_sha256,
)


SIX_CACHE_KIND = "joint_six_attribute_clip"
SIX_FEATURE_FILENAME = "six_condition_anchor.pt"


def load_frozen_six_clip_table(
    anchor_dir: Union[str, Path], texts_path: Union[str, Path] = DEFAULT_TEXTS_PATH,
) -> Tuple[torch.Tensor, List[str]]:
    """严格核验原文、模板、CLIP来源与缓存，再读取完整六条件特征。

    Args:
        anchor_dir: build_six_condition_anchors.py生成的新缓存目录。
        texts_path: 官方三属性原始描述表，须与生成时完全相同。

    Returns:
        (CPU float32特征[N,512], N条完整文本)，N为实际缓存的不同文本数。

    Raises:
        ValueError: 模板、字段、原文、形状、哈希或冻结CLIP来源不符。
        FileNotFoundError: 缓存或源文本缺失；不改用其他缓存。
    """
    directory, source = Path(anchor_dir), Path(texts_path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("完整六条件缓存需要schema_version=1的manifest。")
    if (manifest.get("cache_kind") != SIX_CACHE_KIND
            or manifest.get("template") != SIX_CONDITION_TEMPLATE
            or manifest.get("xml_fields") != list(SIX_XML_FIELDS)
            or manifest.get("condition_counts") != CONDITION_COUNTS
            or manifest.get("condition_dim") != CLIP_FEATURE_DIM
            or manifest.get("feature_transform") != "none"
            or manifest.get("time_representation") != "verbatim_xml_including_date"):
        raise ValueError("六条件模板/字段/维度记录不符，不能使用旧三条件缓存。")
    if manifest.get("texts_file_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("原始属性码表与六条件缓存的校验值不一致。")
    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("model_id") != CLIP_MODEL_ID:
        raise ValueError("六条件组需要与B相同的CLIP ViT-B/16。")
    descriptions = load_condition_descriptions(source)
    raw_rows, texts = manifest.get("raw_conditions_by_index"), manifest.get("condition_texts_by_index")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("六条件缓存没有原始元数据行。")
    if not all(isinstance(row, dict) for row in raw_rows):
        raise ValueError("六条件缓存的原始元数据必须为字典。")
    reconstructed = [build_six_condition_text(row, descriptions) for row in raw_rows]
    if (texts != reconstructed or len(set(reconstructed)) != len(reconstructed)
            or manifest.get("row_count") != len(reconstructed)
            or manifest.get("texts_sha256") != joint_texts_sha256(reconstructed)):
        raise ValueError("六条件缓存的文本、编号顺序或哈希不一致。")
    shape = (len(reconstructed), CLIP_FEATURE_DIM)
    artifacts = manifest.get("artifacts")
    record = artifacts.get(SIX_FEATURE_FILENAME) if isinstance(artifacts, dict) else None
    if not isinstance(record, dict) or record.get("shape") != list(shape):
        raise ValueError("六条件CLIP特征记录形状不符。")
    path = directory / SIX_FEATURE_FILENAME
    if record.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("六条件特征缓存哈希不符。")
    features = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(features, torch.Tensor) or features.shape != shape
            or features.dtype != torch.float32 or not torch.isfinite(features).all()
            or (torch.linalg.vector_norm(features, dim=1) <= 1e-7).any()):
        raise ValueError("六条件特征需要非零、有限的float32 [N,512]张量。")
    return features.detach().clone(), reconstructed


class FrozenSixCLIPConditionPrior(nn.Module):
    """完整六条件文本查表，无可学习条件编码参数。

    Attributes:
        required_fields: 只白名单接收图像级condition_text，不接收框/类别。
        joint_anchor: [N,512] float32持久冻结buffer。
        text_to_index: 从精确原文到缓存行号的索引，不使用图像ID。
        ema_copy_buffers: 明确要求EMA精确复制文本表。
    """

    required_fields = ("condition_text",)
    output_dim = CLIP_FEATURE_DIM
    ema_copy_buffers = ("joint_anchor",)

    def __init__(
        self, anchor_dir: Union[str, Path],
        texts_path: Union[str, Path] = DEFAULT_TEXTS_PATH,
    ) -> None:
        """建立精确文本索引并注册冻结特征表。

        Args:
            anchor_dir: 本轮六条件离线缓存目录。
            texts_path: 与缓存一致的原始属性码表。
        """
        super().__init__()
        features, texts = load_frozen_six_clip_table(anchor_dir, texts_path)
        self.anchor_dir = str(Path(anchor_dir).resolve())
        self.texts_path = str(Path(texts_path).resolve())
        self.condition_texts = texts
        self.text_to_index = {text: index for index, text in enumerate(texts)}
        self.template = SIX_CONDITION_TEMPLATE
        self.register_buffer("joint_anchor", features)

    def profiling_conditions(self) -> Dict[str, List[str]]:
        """提供一条已缓存的合法文本，仅用于计算量统计。

        Returns:
            condition_text列表长度1；不构造虚假的日期/默认条件。
        """
        return {"condition_text": [self.condition_texts[0]]}

    @torch.no_grad()
    def forward(self, conditions: Mapping[str, Any]) -> torch.Tensor:
        """对batch完整文本查表，未缓存文本明确失败。

        Args:
            conditions: condition_text为B条完整字符串；文本留在CPU。

        Returns:
            同缓存设备的c [B,512] float32，无梯度，无额外变换。

        Raises:
            ValueError: 输入不完整、缓存被转成half或文本不在缓存中。
        """
        if not isinstance(conditions, Mapping):
            raise ValueError("六条件组需要显式condition_text字典。")
        texts = conditions.get("condition_text")
        if not isinstance(texts, (list, tuple)) or not texts:
            raise ValueError("condition_text必须是非空batch字符串列表。")
        if self.joint_anchor.dtype != torch.float32:
            raise ValueError("冻结六条件表须保持float32；使用AMP，不使用model.half()。")
        indices: List[int] = []
        for position, text in enumerate(texts):
            if not isinstance(text, str) or text not in self.text_to_index:
                raise ValueError(
                    f"batch第{position}条六条件文本未缓存或非法：{text!r}；"
                    "请用真实元数据生成对应缓存，不默认查第0行。"
                )
            indices.append(self.text_to_index[text])
        index = torch.tensor(indices, dtype=torch.int64, device=self.joint_anchor.device)
        return self.joint_anchor.index_select(0, index)
