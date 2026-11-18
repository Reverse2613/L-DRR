"""Port the tested micro-alignment, FiLM, and CG-DMF formulas.

These modules do not add the old cross-modality message exchange, a joint gain
gate, a complement branch, a geometry MLP, or an auxiliary condition loss.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class RGBIRMicroAlignment(nn.Module):
    """One-scale RGB resampling on the unchanged IR output grid."""

    def __init__(self, channels: int, stride: int, max_shift_px: float) -> None:
        """Build the zero-output offset predictor.

        Args:
            channels: Common feature width C.
            stride: Input-to-feature stride, 8/16/32.
            max_shift_px: Maximum x/y component in input pixels.
        """
        super().__init__()
        self.channels, self.stride, self.max_shift_px = channels, stride, float(max_shift_px)
        width = 2 * channels + 5
        self.spatial_context = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate", groups=width, bias=False),
            nn.GELU(),
        )
        self.offset_head = nn.Conv2d(width, 2, 1)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    def compute_offsets(
        self, rgb: torch.Tensor, ir: torch.Tensor, geometry: torch.Tensor, radius: torch.Tensor,
    ) -> torch.Tensor:
        """Predict bounded sampling offsets, without reading target boxes.

        Args:
            rgb: RGB working features [B,C,h,w].
            ir: IR working features with the same shape.
            geometry: float32 [B,5].
            radius: float32 [B,1], in (0,1].

        Returns:
            float32 [B,2,h,w] input-pixel x/y offsets. Positive x samples
            from the RGB right-hand side, moving content left in the output.
        """
        if rgb.shape != ir.shape or rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError("Micro-alignment requires matching RGB/IR [B,C,h,w].")
        if geometry.shape != (len(rgb), 5) or radius.shape != (len(rgb), 1):
            raise ValueError("Invalid geometry or radius shape.")
        broadcast = geometry.to(rgb.dtype)[:, :, None, None].expand(-1, -1, *rgb.shape[-2:])
        raw = self.offset_head(self.spatial_context(torch.cat((rgb, ir, broadcast), dim=1)))
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            return raw.float().tanh() * (self.max_shift_px * radius[:, :, None, None])

    def forward(
        self, rgb: torch.Tensor, ir: torch.Tensor, geometry: torch.Tensor, radius: torch.Tensor,
    ) -> torch.Tensor:
        """Resample only RGB; exact identity and a live gradient at zero offset.

        Args:
            rgb: RGB working features [B,C,h,w].
            ir: IR features [B,C,h,w], not modified.
            geometry: float32 [B,5].
            radius: float32 [B,1].

        Returns:
            Calibrated RGB [B,C,h,w] in the original feature dtype.
            Two FP32 bilinear/border samples use align_corners=False:
            RGB + Warp_delta(RGB) - Warp_zero(RGB).
        """
        offsets = self.compute_offsets(rgb, ir, geometry, radius)
        batch, _, height, width = rgb.shape
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            x = 2.0 * (torch.arange(width, device=rgb.device, dtype=torch.float32) + 0.5) / width - 1.0
            y = 2.0 * (torch.arange(height, device=rgb.device, dtype=torch.float32) + 0.5) / height - 1.0
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            base = torch.stack((xx, yy), dim=-1)[None].expand(batch, -1, -1, -1)
            feature_offsets = offsets / float(self.stride)
            shifted_grid = base + feature_offsets.permute(0, 2, 3, 1) * offsets.new_tensor((2.0 / width, 2.0 / height))
            source = rgb.float()
            zero = F.grid_sample(source, base, mode="bilinear", padding_mode="border", align_corners=False)
            shifted = F.grid_sample(source, shifted_grid, mode="bilinear", padding_mode="border", align_corners=False)
            return (source + (shifted - zero)).to(rgb.dtype)


class LocalModalityScore(nn.Module):
    """Ordered visual query plus condition bias and query-key compatibility."""

    def __init__(self, channels: int, condition_dim: int = 512, query_dim: int = 32) -> None:
        """Construct the same scoring terms as the existing CG-DMF.

        Args:
            channels: Working feature width C.
            condition_dim: Frozen CLIP width, 512.
            query_dim: Local query/key width, 32.
        """
        super().__init__()
        self.query = nn.Conv2d(2 * channels, query_dim, 1)
        self.key = nn.Linear(condition_dim, query_dim)
        self.condition_bias = nn.Linear(condition_dim, 1)
        self.visual_score = nn.Sequential(nn.GELU(), nn.Conv2d(query_dim, 1, 1))
        self.scale = math.sqrt(query_dim)
        for layer in (self.key, self.condition_bias, self.visual_score[1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self, branch: torch.Tensor, other: torch.Tensor, condition: torch.Tensor,
        use_condition: bool = True,
    ) -> torch.Tensor:
        """Score a branch, preserving FP32 accumulation under AMP.

        Args:
            branch: Evaluated branch [B,C,h,w].
            other: Other modality [B,C,h,w].
            condition: Frozen environment vector [B,512].
            use_condition: False isolates direct condition terms in scoring;
                FiLM can remain enabled in the enclosing block.

        Returns:
            float32 score [B,1,h,w]. It is not a calibrated quality or
            foreground probability.
        """
        query = self.query(torch.cat((branch, other), dim=1))
        visual = self.visual_score(query).float()
        if not use_condition:
            return visual
        c = condition.to(self.key.weight.dtype)
        key = self.key(c)[:, :, None, None]
        bias = self.condition_bias(c).float()[:, :, None, None]
        compatibility = (query.float() * key.float()).sum(1, keepdim=True) / self.scale
        return visual + bias + compatibility


class ConditionFusion(nn.Module):
    """Working projection -> micro-align -> FiLM -> CG-DMF -> projection.

    Controls retain all parameters, including bypassed ones, so changing one
    switch does not change later initialization or state_dict structure.
    """

    def __init__(
        self, channels: int, stride: int, max_shift_px: float,
        alignment: str = "geometry", enable_film: bool = True,
        dynamic_fusion: bool = True, score_condition: bool = True,
    ) -> None:
        """Build one independent P3/P4/P5 fusion block.

        Args:
            channels: Native YOLO11s backbone width, 256/256/512.
            stride: 8/16/32.
            max_shift_px: Maximum input-pixel offset component.
            alignment: off, visual, or geometry.
            enable_film: Enable environment affine modulation.
            dynamic_fusion: False uses fixed equal weights after FiLM.
            score_condition: Include direct environment terms in scoring.
        """
        super().__init__()
        if channels % 32:
            raise ValueError("The port uses 32-group normalization; channels must be divisible by 32.")
        self.alignment = alignment
        self.enable_film, self.dynamic_fusion, self.score_condition = enable_film, dynamic_fusion, score_condition
        self.rgb_work = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.GroupNorm(32, channels))
        self.ir_work = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.GroupNorm(32, channels))
        self.micro_alignment = RGBIRMicroAlignment(channels, stride, max_shift_px)
        self.rgb_film = nn.Linear(512, 2 * channels)
        self.ir_film = nn.Linear(512, 2 * channels)
        for layer in (self.rgb_film, self.ir_film):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.rgb_score = LocalModalityScore(channels)
        self.ir_score = LocalModalityScore(channels)
        self.out_projection = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.GroupNorm(32, channels))

    @staticmethod
    def _film(features: torch.Tensor, layer: nn.Linear, condition: torch.Tensor) -> torch.Tensor:
        """Apply the original unbounded affine FiLM formula.

        Args:
            features: Working features [B,C,h,w].
            layer: Linear map 512 -> 2C.
            condition: Environment vector [B,512].

        Returns:
            (1+gamma)*features+beta, shape [B,C,h,w].
        """
        gamma, beta = layer(condition.to(layer.weight.dtype)).chunk(2, dim=1)
        return (1 + gamma[:, :, None, None]) * features + beta[:, :, None, None]

    def compute_weights(
        self, rgb: torch.Tensor, ir: torch.Tensor, condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute complementary per-location modality weights.

        Args:
            rgb: Modulated RGB [B,C,h,w].
            ir: Modulated IR [B,C,h,w].
            condition: Environment [B,512].

        Returns:
            RGB and IR weights [B,1,h,w], adding to one.
        """
        if not self.dynamic_fusion:
            weight = rgb.new_full((len(rgb), 1, *rgb.shape[-2:]), 0.5, dtype=torch.float32)
        else:
            rgb_score = self.rgb_score(rgb, ir, condition, self.score_condition)
            ir_score = self.ir_score(ir, rgb, condition, self.score_condition)
            weight = (rgb_score - ir_score).sigmoid()
        return weight, 1.0 - weight

    def forward(
        self, rgb: torch.Tensor, ir: torch.Tensor, condition: torch.Tensor,
        geometry: torch.Tensor, radius: torch.Tensor,
    ) -> torch.Tensor:
        """Execute the selected path without conditioning on object GT.

        Args:
            rgb: Raw native backbone RGB [B,C,h,w].
            ir: Raw native backbone IR [B,C,h,w].
            condition: Frozen environment [B,512].
            geometry: Deterministic descriptor [B,5].
            radius: Empirical fraction [B,1].

        Returns:
            One fused native-width tensor [B,C,h,w].
        """
        rgb, ir = self.rgb_work(rgb), self.ir_work(ir)
        if self.alignment != "off":
            rgb = self.micro_alignment(rgb, ir, geometry, radius)
        if self.enable_film:
            rgb = self._film(rgb, self.rgb_film, condition)
            ir = self._film(ir, self.ir_film, condition)
        rgb_weight, ir_weight = self.compute_weights(rgb, ir, condition)
        fused = (rgb_weight * rgb + ir_weight * ir).to(rgb.dtype)
        return self.out_projection(fused)


class ConcatFusion(nn.Module):
    """Simple 2C-to-C projection for the independently trained dual baseline."""

    def __init__(self, channels: int) -> None:
        """Initialize the projection to an equal average of both modalities.

        Args:
            channels: Native backbone feature width C.
        """
        super().__init__()
        self.projection = nn.Conv2d(2 * channels, channels, 1, bias=False)
        with torch.no_grad():
            self.projection.weight.zero_()
            eye = torch.eye(channels)[:, :, None, None] * 0.5
            self.projection.weight[:, :channels].copy_(eye)
            self.projection.weight[:, channels:].copy_(eye)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Fuse the raw backbone features without condition information.

        Args:
            rgb: RGB [B,C,h,w].
            ir: IR [B,C,h,w].

        Returns:
            Fused [B,C,h,w].
        """
        return self.projection(torch.cat((rgb, ir), dim=1))
