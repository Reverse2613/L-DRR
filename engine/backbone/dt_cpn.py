"""DT-CPN：把六种显式条件编码为128维向量，供后续双模态融合使用。

阅读顺序：load_anchor_tables（核对缓存）→ BoundedAnchorEmbedding
（冻结锚点与有界残差）→ DTCPN.forward（几何轨、环境轨和最终合成）。

这里只编码图像级条件，不读取目标类别/框，也不预测条件标签。
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXTS_PATH = REPO_ROOT / "configs/dataset/atrumod_condition_texts.json"
ANCHOR_SPECS = {
    "weather": (10, "weather_anchor.pt"),
    "illumination": (7, "illum_anchor.pt"),
    "location": (11, "loc_anchor.pt"),
}


def load_anchor_tables(
    anchor_dir: Union[str, Path], texts_path: Union[str, Path] = DEFAULT_TEXTS_PATH
) -> Dict[str, torch.Tensor]:
    r"""加载并核对类别编号、来源和数值，拒绝混用或缺失的缓存。

    Args:
        anchor_dir: build_condition_anchors.py 生成的完整目录。
        texts_path: 与生成锚点时完全相同的原始条件文本JSON。

    Returns:
        CPU float32表：weather[10,64]、illumination[7,64]、location[11,64]。

    Raises:
        FileNotFoundError: 缺少源文件、manifest或锚点，不自动生成替代值。
        ValueError: 文本编号/校验值、模型来源或张量规格不符。
    """
    directory, source = Path(anchor_dir), Path(texts_path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    texts = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("锚点manifest必须是schema_version=1。")
    if not isinstance(texts, dict) or texts.get("schema_version") != 1:
        raise ValueError("原始条件文本必须是schema_version=1。")
    if manifest.get("texts_file_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("原始条件文本已改变或缓存来源不符；请核对后重新生成锚点。")
    if manifest.get("model", {}).get("model_id") != "openai/clip-vit-base-patch16":
        raise ValueError("本版本DT-CPN需要CLIP ViT-B/16文本锚点。")
    if manifest.get("projection", {}).get("shape") != [64, 512]:
        raise ValueError("锚点需要共享的64×512投影。")
    tables: Dict[str, torch.Tensor] = {}
    for field, (count, filename) in ANCHOR_SPECS.items():
        rows = texts.get(field)
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError(f"{field}原始文本应有{count}行。")
        if any(not isinstance(row, dict) or type(row.get("id")) is not int
               or row["id"] != index or not isinstance(row.get("text"), str)
               or not row["text"].strip() for index, row in enumerate(rows)):
            raise ValueError(f"{field}必须按原始整数编号排列且文本非空。")
        if manifest.get("texts_by_id", {}).get(field) != [row["text"] for row in rows]:
            raise ValueError(f"{field}缓存行与原始类别文本不对应。")
        path = directory / filename
        record = manifest.get("artifacts", {}).get(filename, {})
        if record.get("shape") != [count, 64]:
            raise ValueError(f"{filename}的manifest形状不符。")
        if record.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"{filename}校验值不符，拒绝加载。")
        table = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(table, torch.Tensor) or table.shape != (count, 64):
            raise ValueError(f"{filename}必须是[{count},64]张量。")
        if table.dtype != torch.float32 or not torch.isfinite(table).all():
            raise ValueError(f"{filename}必须是有限float32值。")
        if not torch.allclose(table.norm(dim=1), torch.ones(count), atol=1e-5, rtol=0):
            raise ValueError(f"{filename}每行必须是L2单位向量。")
        tables[field] = table.detach().clone()
    return tables


class BoundedAnchorEmbedding(nn.Module):
    r"""冻结文本锚点，并学习其周围半径不超过0.2的类别残差。

    Attributes:
        anchor: 冻结buffer [类别数,64]，随模型移动和保存，但不进入优化器。
        residual: 可学习Embedding [类别数,64]，初始化为全0。
        radius: 有效残差的最大L2长度；约束作用于前向值，不是原始参数。
    """

    def __init__(self, anchor: torch.Tensor, radius: float = 0.2) -> None:
        r"""建立冻结锚点与零残差表。

        Args:
            anchor: 已核验的CPU float32单位锚点 [类别数,64]。
            radius: 有效残差球半径，默认0.2。
        """
        super().__init__()
        if anchor.ndim != 2 or anchor.shape[1] != 64:
            raise ValueError("文本锚点应为[类别数,64]。")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("残差半径必须是有限正数。")
        self.radius = radius
        self.register_buffer("anchor", anchor.detach().clone())
        self.residual = nn.Embedding(anchor.shape[0], 64)
        nn.init.zeros_(self.residual.weight)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        r"""查对应类别行，并把有效残差限制到指定半径。

        Args:
            indices: 合法原始类别编号 [B]，int64，与模型在同一设备。

        Returns:
            条件表示 [B,64]；与原锚点的距离不超过radius。
            相加后不再L2归一化，以保持network1.md的残差球定义。
        """
        # float32计算残差范数，避免AMP将小半径和稳定项变成低精度运算。
        delta = self.residual(indices).float()
        norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        scale = (self.radius / (norm + 1e-7)).clamp(max=1.0)
        return F.embedding(indices, self.anchor) + delta * scale


class DTCPN(nn.Module):
    r"""双轨显式条件编码器：几何64维、环境64维，合成128维。

    输入规范化与时间周期编码在数据端完成；本模块不重复归一化。
    """

    def __init__(
        self, anchor_dir: Union[str, Path],
        texts_path: Union[str, Path] = DEFAULT_TEXTS_PATH,
    ) -> None:
        r"""读取固定锚点并建立两条编码轨。

        Args:
            anchor_dir: 三个真实CLIP锚点及manifest所在目录。
            texts_path: 用于核对原始类别编号与文本的JSON路径。
        """
        super().__init__()
        tables = load_anchor_tables(anchor_dir, texts_path)
        self.weather_embedding = BoundedAnchorEmbedding(tables["weather"])
        self.illumination_embedding = BoundedAnchorEmbedding(tables["illumination"])
        self.location_embedding = BoundedAnchorEmbedding(tables["location"])
        # K=8：π、2π、4π、...、128π；每个连续量输出8对sin/cos。
        self.register_buffer("frequencies", math.pi * 2.0 ** torch.arange(8, dtype=torch.float32))
        self.geo_projection = nn.Linear(32, 64)
        self.geo_activation = nn.GELU()
        self.time_projection = nn.Linear(2, 64)
        self.env_norm = nn.LayerNorm(64)
        self.condition_projection = nn.Linear(128, 128)
        # Linear使用PyTorch常规非零初始化，Norm权重1/偏置0。
        # 只有类别残差为零，不能把整个条件链路全部初始化成零。

    def _validate_conditions(self, conditions: Mapping[str, torch.Tensor]) -> None:
        r"""核验batch、设备、类型、范围，拒绝默认填值和静默裁剪。

        Args:
            conditions: 六条件字典；允许额外的原始几何量，但本步不读取。

        Raises:
            ValueError: 任意字段缺失或不符合约定。
        """
        required = ("height", "angle", "time", "weather", "illumination", "location")
        for field in required:
            if field not in conditions or not isinstance(conditions[field], torch.Tensor):
                raise ValueError(f"DT-CPN缺少张量条件：{field}。")
        height = conditions["height"]
        if height.ndim != 2 or height.shape[1] != 1 or height.shape[0] == 0:
            raise ValueError("height应为非空[B,1]。")
        batch, device = height.shape[0], self.geo_projection.weight.device
        for field in required:
            value = conditions[field]
            continuous = field in ("height", "angle", "time")
            shape = (batch, 2 if field == "time" else 1) if continuous else (batch,)
            dtype = torch.float32 if continuous else torch.int64
            if value.shape != shape or value.dtype != dtype or value.device != device:
                raise ValueError(f"{field}需要形状{shape}、类型{dtype}、设备{device}。")
            if not torch.isfinite(value).all():
                raise ValueError(f"{field}含NaN/Inf。")
        for field in ("height", "angle"):
            if ((conditions[field] < 0) | (conditions[field] > 1)).any():
                raise ValueError(f"{field}必须已规范化到[0,1]，不自动裁剪。")
        if not torch.allclose(conditions["time"].norm(dim=1), torch.ones(batch, device=device),
                              atol=1e-5, rtol=0):
            raise ValueError("time应是一天周期的[sin,cos]，每行范数约为1。")
        for field, (count, _) in ANCHOR_SPECS.items():
            if ((conditions[field] < 0) | (conditions[field] >= count)).any():
                raise ValueError(f"{field}原始编号必须在0到{count - 1}之间。")

    def _fourier_encode(self, value: torch.Tensor) -> torch.Tensor:
        r"""交错排列一个连续量的多频sin/cos特征。

        Args:
            value: 已核验的规范化连续量 [B,1]，float32。

        Returns:
            [B,16]，顺序为sin(πv)、cos(πv)、...、sin(128πv)、cos(128πv)。
        """
        phase = value * self.frequencies
        return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(start_dim=1)

    def forward(self, conditions: Mapping[str, torch.Tensor]) -> torch.Tensor:
        r"""计算两轨表示并输出融合模块使用的条件向量。

        Args:
            conditions: height/angle各[B,1]，time[B,2]，均float32；
                weather/illumination/location各[B]、int64。张量与模型同设备。

        Returns:
            c [B,128]。后续按尺度生成FiLM和融合打分，不含条件预测输出。
        """
        self._validate_conditions(conditions)
        # 几何轨：[B,16] + [B,16] → [B,32] → [B,64]。
        geometry = torch.cat((self._fourier_encode(conditions["height"]),
                              self._fourier_encode(conditions["angle"])), dim=-1)
        e_geo = self.geo_activation(self.geo_projection(geometry))
        # 环境轨：三个64维类别表示与64维时间投影相加，再做LayerNorm。
        environment = (self.weather_embedding(conditions["weather"])
                       + self.illumination_embedding(conditions["illumination"])
                       + self.location_embedding(conditions["location"])
                       + self.time_projection(conditions["time"]))
        e_env = self.env_norm(environment)
        # 两轨按geo→env顺序拼接，只用Linear合成，不增加方案外的激活或损失。
        return self.condition_projection(torch.cat((e_geo, e_env), dim=-1))
