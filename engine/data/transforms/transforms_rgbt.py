"""
engine/data/transforms/transforms_rgbt.py

双模态 (RGB + TIR) 旋转目标检测专用的锁步同步数据增强流水线.
通过显式重写 __call__ 彻底屏蔽父类单模态逻辑，实现双模态绝对空间几何锁步.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL import Image

from ...core import register
from .transforms_obb import (
    ResizeOBB,
    SanitizeBoundingBoxesOBB,
    ConvertOBB,
    get_spatial_size,
    obb2poly,
    poly2obb,
)
from ._transforms import ConvertPILImage, RandomPhotometricDistort


@register()
class PairedResizeOBB(T.Transform):
    r"""双模态同步尺寸缩放算子."""

    def __init__(
        self,
        size: Union[int, List[int], Tuple[int, int]],
        interpolation=T.InterpolationMode.BILINEAR,
        keep_ratio: bool = False,
        pad_to_size: bool = False,
        fill: int = 0,
    ) -> None:
        super().__init__()
        self.resize_op = ResizeOBB(
            size=size,
            interpolation=interpolation,
            keep_ratio=keep_ratio,
            pad_to_size=pad_to_size,
            fill=fill,
        )

    def __call__(self, *inputs: Any) -> Any:
        sample = inputs[0] if len(inputs) == 1 else inputs
        paired_imgs, target = sample[0], sample[1]
        extra_args = sample[2:] if len(sample) > 2 else ()

        # 1. 优先对 IR 图像与 target 协同缩放
        img_ir_resized, target_scaled = self.resize_op(
            paired_imgs["ir"], target
        )
        # 2. 对 RGB 图像使用完全相同的尺寸目标缩放
        img_rgb_resized, _ = self.resize_op(paired_imgs["rgb"], None)

        paired_imgs_out = {"ir": img_ir_resized, "rgb": img_rgb_resized}
        return (paired_imgs_out, target_scaled) + extra_args


@register()
class PairedRandomHorizontalFlipOBB(T.Transform):
    r"""双模态严格锁步水平翻转算子."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p: float = p

    def __call__(self, *inputs: Any) -> Any:
        sample = inputs[0] if len(inputs) == 1 else inputs
        paired_imgs, target = sample[0], sample[1]
        extra_args = sample[2:] if len(sample) > 2 else ()

        if torch.rand(1).item() < self.p:
            img_ir = paired_imgs["ir"]
            img_rgb = paired_imgs["rgb"]
            h, w = get_spatial_size(img_ir)

            # 双模态图像同步水平镜像
            img_ir_flipped = F.hflip(img_ir)
            img_rgb_flipped = F.hflip(img_rgb)

            # 旋转框几何翻转
            if target is not None and "boxes" in target:
                boxes = target["boxes"].clone()
                if boxes.shape[0] > 0:
                    polys = obb2poly(boxes)
                    polys[:, :, 0] = w - polys[:, :, 0]
                    boxes = poly2obb(polys, device=boxes.device)
                target = target.copy()
                target["boxes"] = boxes

            paired_imgs_out = {"ir": img_ir_flipped, "rgb": img_rgb_flipped}
            return (paired_imgs_out, target) + extra_args

        return (paired_imgs, target) + extra_args


@register()
class PairedPhotometricDistortRGBOnly(T.Transform):
    r"""物理安全色彩扰动算子 (仅作用于 RGB 分支)."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p: float = p
        self.distort_op = RandomPhotometricDistort(p=1.0)

    def __call__(self, *inputs: Any) -> Any:
        sample = inputs[0] if len(inputs) == 1 else inputs
        paired_imgs, target = sample[0], sample[1]
        extra_args = sample[2:] if len(sample) > 2 else ()

        img_rgb = paired_imgs["rgb"]
        img_ir = paired_imgs["ir"]

        if torch.rand(1).item() < self.p:
            img_rgb = self.distort_op(img_rgb)

        paired_imgs_out = {"ir": img_ir, "rgb": img_rgb}
        return (paired_imgs_out, target) + extra_args


@register()
class PairedSanitizeBoundingBoxesOBB(T.Transform):
    r"""双模态旋转框有效性校验算子 (自适应兼容 Dict 与 Tensor)."""

    def __init__(self, min_size: float = 1.0) -> None:
        super().__init__()
        self.sanitizer = SanitizeBoundingBoxesOBB(min_size=min_size)

    def __call__(self, *inputs: Any) -> Any:
        sample = inputs[0] if len(inputs) == 1 else inputs
        inpt, target = sample[0], sample[1]
        extra_args = sample[2:] if len(sample) > 2 else ()

        # 智能判定：兼容字典 {'ir': ...} 与已转换的 Tensor [6, H, W]
        img_for_size = inpt["ir"] if isinstance(inpt, dict) else inpt

        res = self.sanitizer(img_for_size, target)
        if isinstance(res, tuple):
            _, target_clean = res[0], res[1]
        else:
            target_clean = target

        return (inpt, target_clean) + extra_args


@register()
class PairedConvertPILToTensor(T.Transform):
    r"""成对 PIL 图像转 6 通道 Tensor 算子."""

    def __init__(self, dtype: str = "float32", scale: bool = True) -> None:
        super().__init__()
        self.converter = ConvertPILImage(dtype=dtype, scale=scale)

    def __call__(self, *inputs: Any) -> Any:
        sample = inputs[0] if len(inputs) == 1 else inputs
        paired_imgs, target = sample[0], sample[1]
        extra_args = sample[2:] if len(sample) > 2 else ()

        tensor_ir = self.converter(paired_imgs["ir"])
        tensor_rgb = self.converter(paired_imgs["rgb"])

        # 沿通道维度拼接为 [6, H, W]
        tensor_6ch = torch.cat([tensor_ir, tensor_rgb], dim=0)

        return (tensor_6ch, target) + extra_args


@register()
class PairedConvertOBB(T.Transform):
    r"""双模态旋转框归一化算子 (自适应兼容 Dict 与 Tensor)."""

    def __init__(self, normalize: bool = False) -> None:
        super().__init__()
        self.converter = ConvertOBB(normalize=normalize)

    def __call__(self, *inputs: Any) -> Any:
        sample = inputs[0] if len(inputs) == 1 else inputs
        inpt, target = sample[0], sample[1]
        extra_args = sample[2:] if len(sample) > 2 else ()

        # 智能判定：兼容字典 {'ir': ...} 与已转换的 Tensor [6, H, W]
        img_for_size = inpt["ir"] if isinstance(inpt, dict) else inpt

        res = self.converter(img_for_size, target)
        if isinstance(res, tuple):
            _, target_norm = res[0], res[1]
        else:
            target_norm = target

        return (inpt, target_norm) + extra_args



@register()
class PairedMosaicOBB(T.Transform):
    r"""双模态 (RGB + TIR) 严格几何锁步的四图拼接旋转框增强算子."""

    def __init__(
        self,
        output_size: int = 640,
        max_size: Optional[int] = None,
        rotation_range: float = 0.0,
        translation_range: Tuple[float, float] = (0.1, 0.1),
        scaling_range: Tuple[float, float] = (0.5, 1.5),
        probability: float = 1.0,
        fill_value: int = 0,
    ) -> None:
        super().__init__()
        self.output_size: int = output_size
        self.probability: float = probability
        self.resize_op = PairedResizeOBB(size=output_size)
        self.rotation_range: float = rotation_range
        self.translation_range: Tuple[float, float] = translation_range
        self.scaling_range: Tuple[float, float] = scaling_range
        self.fill_value: int = fill_value

    def _get_affine_matrix(
        self, img_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, float, Tuple[int, int], float]:
        h, w = img_size
        angle = (
            float(
                torch.empty(1)
                .uniform_(-self.rotation_range, self.rotation_range)
                .item()
            )
            if self.rotation_range > 0
            else 0.0
        )
        scale = float(
            torch.empty(1)
            .uniform_(self.scaling_range[0], self.scaling_range[1])
            .item()
        )

        max_dx = float(self.translation_range[0] * w)
        max_dy = float(self.translation_range[1] * h)
        tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
        ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))

        center = (w * 0.5, h * 0.5)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty

        return matrix, angle, (tx, ty), scale

    def __call__(self, *inputs: Any) -> Any:
        sample = inputs[0] if len(inputs) == 1 else inputs
        paired_imgs, target, dataset = sample[0], sample[1], sample[2]
        extra_args = sample[3:] if len(sample) > 3 else ()

        if self.probability < 1.0 and random.random() > self.probability:
            return (paired_imgs, target, dataset) + extra_args

        # 缩放基准首张成对样本
        (p_img0, target0) = self.resize_op(paired_imgs, target)
        ir0, rgb0 = p_img0["ir"], p_img0["rgb"]
        w, h = ir0.size

        # 随机采样另外 3 个成对样本
        sample_indices = random.choices(range(len(dataset)), k=3)
        paired_samples = [(ir0, rgb0, target0)]

        for s_idx in sample_indices:
            s_paired, s_target = dataset.load_item(s_idx)
            (s_p_img, s_target_res) = self.resize_op(s_paired, s_target)
            paired_samples.append(
                (s_p_img["ir"], s_p_img["rgb"], s_target_res)
            )

        merged_ir = Image.new(mode=ir0.mode, size=(w * 2, h * 2), color=0)
        merged_rgb = Image.new(mode=rgb0.mode, size=(w * 2, h * 2), color=0)

        placement_offsets = [[0, 0], [w, 0], [0, h], [w, h]]
        combined_boxes = []
        combined_labels = []

        for i, (img_ir_i, img_rgb_i, tgt_i) in enumerate(paired_samples):
            dx, dy = placement_offsets[i]
            merged_ir.paste(img_ir_i, (dx, dy))
            merged_rgb.paste(img_rgb_i, (dx, dy))

            if "boxes" in tgt_i and tgt_i["boxes"].shape[0] > 0:
                boxes_i = tgt_i["boxes"].clone()
                boxes_i[:, 0] += dx
                boxes_i[:, 1] += dy
                combined_boxes.append(boxes_i)
                combined_labels.append(tgt_i["labels"])

        merged_target = target0.copy()
        if len(combined_boxes) > 0:
            merged_target["boxes"] = torch.cat(combined_boxes, dim=0)
            merged_target["labels"] = torch.cat(combined_labels, dim=0)
        else:
            merged_target["boxes"] = torch.zeros((0, 5), dtype=torch.float32)
            merged_target["labels"] = torch.zeros((0,), dtype=torch.int64)

        affine_mat, angle, (tx, ty), scale = self._get_affine_matrix(
            (h * 2, w * 2)
        )

        final_ir = F.affine(
            merged_ir,
            angle=angle,
            translate=(tx, ty),
            scale=scale,
            shear=0.0,
            interpolation=T.InterpolationMode.BILINEAR,
            fill=0,
        )
        final_rgb = F.affine(
            merged_rgb,
            angle=angle,
            translate=(tx, ty),
            scale=scale,
            shear=0.0,
            interpolation=T.InterpolationMode.BILINEAR,
            fill=0,
        )

        if (
            "boxes" in merged_target
            and merged_target["boxes"].shape[0] > 0
        ):
            boxes_all = merged_target["boxes"]
            polys = obb2poly(boxes_all)
            n_pts = polys.shape[0]
            flat_pts = polys.reshape(-1, 2)
            ones = np.ones((flat_pts.shape[0], 1), dtype=flat_pts.dtype)
            homo_pts = np.hstack([flat_pts, ones])

            trans_pts = (affine_mat @ homo_pts.T).T.reshape(n_pts, 4, 2)
            trans_pts = np.nan_to_num(
                trans_pts, nan=0.0, posinf=1e7, neginf=-1e7
            )
            new_obbs = poly2obb(trans_pts, device=boxes_all.device)
            merged_target["boxes"] = new_obbs

        paired_output = {"ir": final_ir, "rgb": final_rgb}
        return (paired_output, merged_target, dataset) + extra_args


@register()
class PairedRandomZoomOutOBB(T.Transform):
    # 物理意义：模拟无人机拉高、小目标稀疏场景。
    # IR 与 RGB 共用同一缩放比与画布落点，仅平移框中心，不改角度，
    # 热成像填充 0 即低温背景，可见光填充 0 即黑边，双模态几何严格对齐。
    def __init__(self, fill: int = 0, side_range: Tuple[float, float] = (1.0, 4.0), p: float = 0.5) -> None:
        super().__init__()
        self.fill = fill
        self.side_range = side_range
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        sample = inputs[0] if len(inputs) == 1 else inputs
        paired_imgs, target = sample[0], sample[1]
        extra_args = sample[2:] if len(sample) > 2 else ()

        if torch.rand(1).item() >= self.p:
            return (paired_imgs, target) + extra_args

        # 以 IR 尺寸为基准采样同一组随机参数，保证双模态锁步
        ref_img = paired_imgs["ir"] if isinstance(paired_imgs, dict) else paired_imgs
        h, w = get_spatial_size(ref_img)
        scale = float(torch.empty(1).uniform_(self.side_range[0], self.side_range[1]).item())
        new_h, new_w = int(h * scale), int(w * scale)
        left = int(torch.randint(0, new_w - w + 1, (1,)).item())
        top = int(torch.randint(0, new_h - h + 1, (1,)).item())
        padding = [left, top, new_w - w - left, new_h - h - top]

        if isinstance(paired_imgs, dict):
            img_ir_pad = F.pad(paired_imgs["ir"], padding, fill=self.fill)
            img_rgb_pad = F.pad(paired_imgs["rgb"], padding, fill=self.fill)
            paired_out = {"ir": img_ir_pad, "rgb": img_rgb_pad}
        else:
            paired_out = F.pad(paired_imgs, padding, fill=self.fill)

        if target is not None and "boxes" in target:
            boxes = target["boxes"].clone()
            if boxes.shape[0] > 0:
                boxes[:, 0] += left
                boxes[:, 1] += top
            target = target.copy()
            target["boxes"] = boxes

        return (paired_out, target) + extra_args


@register()
class PairedRandomIoUCropOBB(T.Transform):
    # 物理意义：模拟无人机低空、大目标截断场景。
    # IR 与 RGB 共用同一裁剪窗口，仅保留中心落在窗口内的旋转框并平移，
    # 经 obb2poly/poly2obb 保角重算，避免单模态各自裁剪导致的模态错位。
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
        p: float = 1.0,
    ) -> None:
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.sampler_options = sampler_options or [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.trials = trials
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        sample = inputs[0] if len(inputs) == 1 else inputs
        paired_imgs, target = sample[0], sample[1]
        extra_args = sample[2:] if len(sample) > 2 else ()

        if torch.rand(1).item() >= self.p:
            return (paired_imgs, target) + extra_args
        if target is None or "boxes" not in target:
            return (paired_imgs, target) + extra_args

        ref_img = paired_imgs["ir"] if isinstance(paired_imgs, dict) else paired_imgs
        h, w = get_spatial_size(ref_img)

        for _ in range(self.trials):
            scale = float(self.min_scale + torch.rand(1).item() * (self.max_scale - self.min_scale))
            aspect = float(self.min_aspect_ratio + torch.rand(1).item() * (self.max_aspect_ratio - self.min_aspect_ratio))
            crop_h = int(h * scale / (aspect ** 0.5))
            crop_w = int(w * scale * (aspect ** 0.5))
            if crop_h > h or crop_w > w or crop_h <= 0 or crop_w <= 0:
                continue
            top = int(torch.randint(0, h - crop_h + 1, (1,)).item())
            left = int(torch.randint(0, w - crop_w + 1, (1,)).item())

            boxes = target["boxes"]
            if boxes.shape[0] == 0:
                continue
            keep = (boxes[:, 0] >= left) & (boxes[:, 0] < left + crop_w) & (boxes[:, 1] >= top) & (boxes[:, 1] < top + crop_h)
            if not bool(keep.any()):
                continue

            # 双模态同窗口裁剪
            if isinstance(paired_imgs, dict):
                img_ir_crop = F.crop(paired_imgs["ir"], top, left, crop_h, crop_w)
                img_rgb_crop = F.crop(paired_imgs["rgb"], top, left, crop_h, crop_w)
                paired_out = {"ir": img_ir_crop, "rgb": img_rgb_crop}
            else:
                paired_out = F.crop(paired_imgs, top, left, crop_h, crop_w)

            new_boxes = boxes[keep].clone()
            new_boxes[:, 0] -= left
            new_boxes[:, 1] -= top
            new_target = target.copy()
            new_target["boxes"] = new_boxes
            if "labels" in new_target:
                new_target["labels"] = new_target["labels"][keep]

            return (paired_out, new_target) + extra_args

        return (paired_imgs, target) + extra_args