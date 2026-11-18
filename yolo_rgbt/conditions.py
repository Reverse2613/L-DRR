"""Environment CLIP lookup and deterministic geometry have separate roles.

This ports the existing three-condition cache and five-dimensional geometric
descriptor. It does not claim statistical independence, use a CLIP image
encoder, predict conditions, or read object annotations.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Dict, Mapping, Tuple

import torch
from torch import nn

from .data import CONDITION_COUNTS


TEMPLATE = "weather: {weather}; illumination: {illumination}; location: {location}."
FIELDS = ("weather", "illumination", "location")
FEATURE_FILE = "joint_condition_anchor.pt"
RHO_MAX = 1.0 + math.sin(math.radians(75.0))


def load_environment_cache(directory: Path, texts_path: Path) -> Tuple[torch.Tensor, str]:
    """Validate the original 770-row raw CLIP ViT-B/16 cache.

    Args:
        directory: Existing cache directory; no weights are downloaded.
        texts_path: Original ATR-UMOD attribute descriptions.

    Returns:
        Frozen float32 [770,512] table and its artifact SHA256.

    Raises:
        ValueError: Source, template, feature transform, IDs or checksum differ.
    """
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    descriptions = json.loads(texts_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1, "cache_kind": "joint_three_attribute_clip",
        "template": TEMPLATE, "condition_counts": CONDITION_COUNTS,
        "condition_dim": 512, "id_order": list(FIELDS),
        "index_formula": "(weather * 7 + illumination) * 11 + location",
        "feature_transform": "none",
        "texts_file_sha256": hashlib.sha256(texts_path.read_bytes()).hexdigest(),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"CLIP cache manifest mismatch: {key}.")
    if manifest.get("model", {}).get("model_id") != "openai/clip-vit-base-patch16":
        raise ValueError("Expected the existing CLIP ViT-B/16 text cache.")
    if descriptions.get("schema_version") != 1:
        raise ValueError("Unsupported condition descriptions schema.")
    texts: Dict[str, list[str]] = {}
    for field, count in CONDITION_COUNTS.items():
        rows = descriptions.get(field)
        if not isinstance(rows, list) or len(rows) != count or any(
            row.get("id") != index or not isinstance(row.get("text"), str) or not row["text"].strip()
            for index, row in enumerate(rows)
        ):
            raise ValueError(f"Invalid original description rows for {field}.")
        texts[field] = [row["text"] for row in rows]
    tuples = [[w, i, l] for w in range(10) for i in range(7) for l in range(11)]
    prompts = [TEMPLATE.format(weather=texts["weather"][w], illumination=texts["illumination"][i],
                               location=texts["location"][l]) for w, i, l in tuples]
    prompt_hash = hashlib.sha256(json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    if (manifest.get("texts_by_id") != texts or manifest.get("combination_ids") != tuples
            or manifest.get("joint_texts_by_index") != prompts or manifest.get("joint_texts_sha256") != prompt_hash):
        raise ValueError("CLIP cache rows do not correspond to the original attribute IDs.")
    path = directory / FEATURE_FILE
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact = manifest.get("artifacts", {}).get(FEATURE_FILE, {})
    if artifact.get("shape") != [770, 512] or artifact.get("sha256") != checksum:
        raise ValueError("CLIP feature file checksum/shape mismatch.")
    table = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(table, torch.Tensor) or table.shape != (770, 512) or table.dtype != torch.float32
            or not torch.isfinite(table).all() or (table.norm(dim=1) <= 1e-7).any()):
        raise ValueError("CLIP table must contain finite, nonzero raw float32 [770,512] vectors.")
    return table.detach().clone(), checksum


class FrozenEnvironmentPrior(nn.Module):
    """Look up real or constant environment prompts while retaining FP32.

    The table is a nonpersistent buffer: native EMA updates only state_dict
    tensors, so it cannot average the frozen vectors. Native checkpoints pickle
    the entire model, including this buffer. _apply additionally preserves FP32
    when native validation/checkpoint code calls half()/float().
    """

    def __init__(self, directory: Path, texts_path: Path, mode: str = "real") -> None:
        """Load the cache once.

        Args:
            directory: Existing original three-attribute CLIP cache.
            texts_path: Original attribute descriptions.
            mode: real per-image lookup or constant row-zero control.
        """
        super().__init__()
        if mode not in {"real", "constant"}:
            raise ValueError("Environment mode must be real or constant.")
        table, checksum = load_environment_cache(directory, texts_path)
        self.mode, self.cache_sha256 = mode, checksum
        self.register_buffer("joint_anchor", table, persistent=False)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> FrozenEnvironmentPrior:
        """Move the cache without irreversibly quantizing the stored vectors.

        Args:
            fn: PyTorch device/dtype conversion.
            recurse: Standard Module conversion flag.

        Returns:
            This module with its cache still float32 on the requested device.
        """
        anchor = self._buffers.pop("joint_anchor")
        try:
            super()._apply(fn, recurse=recurse)
            probe = fn(anchor)
            self._buffers["joint_anchor"] = (
                probe if probe.dtype == torch.float32 else anchor.to(device=probe.device)
            )
        except Exception:
            self._buffers["joint_anchor"] = anchor
            raise
        return self

    @torch.no_grad()
    def forward(self, conditions: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Read only environment IDs, never geometry or target information.

        Args:
            conditions: weather/illumination/location int64 [B] tensors.
                Extra height/angle fields are intentionally not encoded here.

        Returns:
            Frozen float32 [B,512], without normalization or a learned adapter.
        """
        if self.joint_anchor.dtype != torch.float32:
            raise ValueError("The frozen environment cache lost FP32 precision.")
        batch = None
        for field, count in CONDITION_COUNTS.items():
            value = conditions.get(field)
            if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() == 0:
                raise ValueError(f"{field}: require a nonempty [B] tensor.")
            if value.dtype != torch.int64 or value.device != self.joint_anchor.device:
                raise ValueError(f"{field}: require int64 on the model device.")
            if batch is None:
                batch = len(value)
            if len(value) != batch or ((value < 0) | (value >= count)).any():
                raise ValueError(f"{field}: inconsistent batch or invalid original ID.")
        index = (conditions["weather"] * 7 + conditions["illumination"]) * 11 + conditions["location"]
        if self.mode == "constant":
            index = torch.zeros_like(index)
        return self.joint_anchor.index_select(0, index)


def geometry_descriptor(
    conditions: Mapping[str, torch.Tensor], batch: int, device: torch.device, mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Produce geometry only for the micro-alignment branch.

    Args:
        conditions: height/angle float32 [B,1], normalized as in data.py.
        batch: Image batch size.
        device: Feature device.
        mode: geometry uses real telemetry. visual uses an explicitly absent
            descriptor and global bound; it does not fabricate height/angle.

    Returns:
        geometry float32 [B,5]: log height, inverse height, sin, cos, tan.
        radius_fraction float32 [B,1], multiplying the input-pixel shift bound.
        This is the old empirical envelope, not a calibrated parallax bound.
    """
    if mode == "visual":
        return torch.zeros(batch, 5, device=device), torch.ones(batch, 1, device=device)
    if mode != "geometry":
        raise ValueError("Geometry descriptor mode must be geometry or visual.")
    values = []
    for field in ("height", "angle"):
        value = conditions.get(field)
        if (not isinstance(value, torch.Tensor) or value.shape != (batch, 1)
                or value.dtype != torch.float32 or value.device != device
                or not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any()):
            raise ValueError(f"{field}: require real normalized finite float32 [B,1] in [0,1].")
        values.append(value)
    with torch.autocast(device_type=device.type, enabled=False):
        height_m = 80.0 + 220.0 * values[0]
        angle = values[1] * math.radians(75.0)
        inverse = 80.0 / height_m
        sine, cosine = angle.sin(), angle.cos()
        geometry = torch.cat((torch.log(height_m / 80.0) / math.log(300.0 / 80.0),
                              inverse, sine, cosine, angle.tan()), dim=1)
        radius = inverse * (1.0 + sine) / RHO_MAX
    return geometry, radius
