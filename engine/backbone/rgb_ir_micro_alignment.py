"""RGB→IR微校准：在FiLM前对RGB工作特征做小幅、可学习的空间重采样。

高度/倾角只提供经验包络，不把它解释为经相机标定证明的视差界限。
偏移以模型输入像素计量；除以本尺度步长后才进入grid_sample。
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


HEIGHT_MIN_M = 80.0
HEIGHT_MAX_M = 300.0
ANGLE_MAX_DEG = 75.0
GEOMETRY_DIM = 5
RHO_MAX = 1.0 + math.sin(math.radians(ANGLE_MAX_DEG))


def build_alignment_geometry(
    height: torch.Tensor, angle: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""从已有规范化元数据构造确定性几何描述符，不增加条件预测器。

    Args:
        height: float32 [B,1]，已有加载器的(H_m-80)/220，范围[0,1]。
        angle: float32 [B,1]，已有加载器的UAVangle_deg/75，范围[0,1]。
            与height同设备；这里不读取目标框或目标朝向。

    Returns:
        geometry: float32 [B,5]，依次为规范化log高度、80/H、
            sin(alpha)、cos(alpha)、tan(alpha)。第一原型直接使用这5维。
        radius_fraction: float32 [B,1]，(80/H)*(1+sin(alpha))/RHO_MAX，
            大于0且不超过1；乘显式像素上限后得到本图的经验偏移包络。

    Raises:
        ValueError: 元数据缺失、设备/形状/类型不符，或含非有限/越界值。
            不补默认高度、倾角，也不静默裁剪。
    """
    if not all(isinstance(value, torch.Tensor) for value in (height, angle)):
        raise ValueError("微校准必须显式提供height和angle张量。")
    if (height.ndim != 2 or height.shape[1] != 1 or height.shape[0] <= 0
            or angle.shape != height.shape):
        raise ValueError("微校准height/angle必须同为非空[B,1]。")
    if height.dtype != torch.float32 or angle.dtype != torch.float32:
        raise ValueError("微校准height/angle必须是加载器输出的float32。")
    if height.device != angle.device:
        raise ValueError("微校准height和angle必须同设备。")
    for name, value in (("height", height), ("angle", angle)):
        if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
            raise ValueError(f"微校准{name}必须有限且在[0,1]内，不补值或裁剪。")
    # 即使外层处于AMP，三角/对数与经验包络仍明确使用FP32。
    with torch.autocast(device_type=height.device.type, enabled=False):
        height_m = HEIGHT_MIN_M + (HEIGHT_MAX_M - HEIGHT_MIN_M) * height
        alpha = angle * math.radians(ANGLE_MAX_DEG)
        inverse_height = HEIGHT_MIN_M / height_m
        sine, cosine = torch.sin(alpha), torch.cos(alpha)
        log_height = torch.log(height_m / HEIGHT_MIN_M) / math.log(
            HEIGHT_MAX_M / HEIGHT_MIN_M,
        )
        geometry = torch.cat((log_height, inverse_height, sine, cosine,
                              torch.tan(alpha)), dim=1)
        radius_fraction = inverse_height * (1.0 + sine) / RHO_MAX
    return geometry, radius_fraction


class RGBIRMicroAlignment(nn.Module):
    r"""一个尺度的单向空间微校准；IR仅参与偏移预测，不被重采样。

    Attributes:
        stride: 本尺度相对模型输入的步长，B配置为8/16/32。
        max_shift_px: 每个坐标分量的最大模型输入像素偏移，默认8像素。
            它是L-infinity分量界限，不是欧氏距离半径。
        offset_head: 严格零初始化的2通道读出，可获得首步检测梯度。
    """

    def __init__(self, channels: int, stride: int, max_shift_px: float) -> None:
        r"""建立DW3×3→GELU→PW1×1偏移预测器。

        Args:
            channels: 两路共同工作特征的通道C。
            stride: 本尺度正整数步长。
            max_shift_px: 显式提供的正有限输入像素上限；三尺度使用同一值。

        Raises:
            ValueError: 通道、步长或像素上限非法。
        """
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (channels, stride)):
            raise ValueError("微校准channels/stride必须是正整数。")
        if (isinstance(max_shift_px, bool) or not isinstance(max_shift_px, (int, float))
                or not math.isfinite(max_shift_px) or max_shift_px <= 0):
            raise ValueError("微校准max_shift_px必须是正有限像素数。")
        self.channels = channels
        self.stride = stride
        self.max_shift_px = float(max_shift_px)
        context_channels = 2 * channels + GEOMETRY_DIM
        self.spatial_context = nn.Sequential(
            nn.Conv2d(context_channels, context_channels, 3, padding=1,
                      padding_mode="replicate", groups=context_channels, bias=False),
            nn.GELU(),
        )
        self.offset_head = nn.Conv2d(context_channels, 2, 1, bias=True)
        # 隐藏卷积保持非零常规初始化；只置零读出，避免整个分支首步无梯度。
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

    def _validate_inputs(
        self, rgb: torch.Tensor, ir: torch.Tensor, geometry: torch.Tensor,
        radius_fraction: torch.Tensor,
    ) -> None:
        r"""严格核验两路空间对应及几何包络契约。

        Args:
            rgb: S_rgb [B,C,H_l,W_l]，float32/float16/bfloat16。
            ir: S_ir，与rgb同形状、dtype、设备。
            geometry: float32 [B,5]，确定性描述符。
            radius_fraction: float32 [B,1]，范围(0,1]。
        """
        values = (rgb, ir, geometry, radius_fraction)
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise ValueError("微校准两路特征、几何描述符和包络必须是张量。")
        if (rgb.ndim != 4 or rgb.shape[1] != self.channels
                or any(size <= 0 for size in rgb.shape) or ir.shape != rgb.shape):
            raise ValueError(f"微校准两路特征必须同为非空[B,{self.channels},H_l,W_l]。")
        if rgb.dtype not in (torch.float32, torch.float16, torch.bfloat16) or ir.dtype != rgb.dtype:
            raise ValueError("微校准两路特征必须使用相同的受支持浮点类型。")
        if geometry.shape != (rgb.shape[0], GEOMETRY_DIM) or radius_fraction.shape != (rgb.shape[0], 1):
            raise ValueError("微校准geometry/radius_fraction需要[B,5]/[B,1]。")
        if geometry.dtype != torch.float32 or radius_fraction.dtype != torch.float32:
            raise ValueError("微校准几何描述符与包络必须使用float32。")
        device = self.offset_head.weight.device
        if any(value.device != device for value in values):
            raise ValueError(f"微校准输入必须与模型同在{device}。")
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError("微校准输入含NaN/Inf，不自动替换。")
        if ((radius_fraction <= 0) | (radius_fraction > 1)).any():
            raise ValueError("微校准经验包络比例必须在(0,1]内。")

    def compute_offsets(
        self, rgb: torch.Tensor, ir: torch.Tensor, geometry: torch.Tensor,
        radius_fraction: torch.Tensor,
    ) -> torch.Tensor:
        r"""预测有界、正向采样偏移，保留完整检测梯度。

        Args:
            rgb: 未经FiLM的S_rgb [B,C,H_l,W_l]。
            ir: 未经FiLM的S_ir [B,C,H_l,W_l]，不会被本模块原地修改。
            geometry: float32 [B,5]，沿空间广播作为预测器输入。
            radius_fraction: float32 [B,1]，由真实高度/倾角确定。

        Returns:
            float32 [B,2,H_l,W_l]，x/y模型输入像素采样偏移。
            正x表示从RGB右侧采样，所以RGB内容向输出左侧移动。
            各分量绝对值≤max_shift_px*radius_fraction。
        """
        self._validate_inputs(rgb, ir, geometry, radius_fraction)
        height, width = rgb.shape[-2:]
        broadcast = geometry.to(dtype=rgb.dtype)[:, :, None, None].expand(
            -1, -1, height, width,
        )
        context = torch.cat((rgb, ir, broadcast), dim=1)
        raw_offsets = self.offset_head(self.spatial_context(context))
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            offsets = torch.tanh(raw_offsets.float()) * (
                self.max_shift_px * radius_fraction[:, :, None, None]
            )
        if not torch.isfinite(offsets).all():
            raise ValueError("微校准预测偏移含NaN/Inf，不自动置零。")
        return offsets

    def forward(
        self, rgb: torch.Tensor, ir: torch.Tensor, geometry: torch.Tensor,
        radius_fraction: torch.Tensor,
    ) -> torch.Tensor:
        r"""按IR输出网格重采样RGB，零偏移时数值精确保持原RGB。

        Args:
            rgb: S_rgb [B,C,H_l,W_l]，浮点工作特征。
            ir: S_ir，同形状；只用于偏移预测，直接IR主路径保留原张量。
            geometry: float32 [B,5]。
            radius_fraction: float32 [B,1]。

        Returns:
            与rgb同形状/dtype的校准特征。使用rgb+Warp_delta(rgb)-Warp_0(rgb)
            的差分重采样形式，消除零偏移时grid_sample的浮点插值误差。
            两次采样均为bilinear/border/align_corners=False；不使用会切断
            首步偏移梯度的“检测到零偏移就直接返回”分支。
        """
        offsets_px = self.compute_offsets(rgb, ir, geometry, radius_fraction)
        batch, _, height, width = rgb.shape
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            # align_corners=False的基础网格必须落在像素中心，而非linspace端点。
            grid_x = 2.0 * (torch.arange(width, device=rgb.device, dtype=torch.float32) + 0.5) / width - 1.0
            grid_y = 2.0 * (torch.arange(height, device=rgb.device, dtype=torch.float32) + 0.5) / height - 1.0
            y, x = torch.meshgrid(grid_y, grid_x, indexing="ij")
            base = torch.stack((x, y), dim=-1)[None].expand(batch, -1, -1, -1)
            feature_offsets = offsets_px / float(self.stride)
            normalizer = feature_offsets.new_tensor((2.0 / width, 2.0 / height))
            shifted_grid = base + feature_offsets.permute(0, 2, 3, 1) * normalizer
            source = rgb.float()
            identity = F.grid_sample(source, base, mode="bilinear", padding_mode="border", align_corners=False)
            shifted = F.grid_sample(source, shifted_grid, mode="bilinear", padding_mode="border", align_corners=False)
            result = (source + (shifted - identity)).to(dtype=rgb.dtype)
        if not torch.isfinite(result).all():
            raise ValueError("微校准重采样输出含NaN/Inf，不自动替换。")
        return result
