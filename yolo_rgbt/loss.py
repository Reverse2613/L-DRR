# Adapted from Ultralytics 8.3.0 v8OBBLoss, AGPL-3.0.
"""Keep the native ProbIoU/BCE/DFL objective, correcting rectangular GT units."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn

from .runtime import require_framework

require_framework()
from ultralytics.utils.loss import v8OBBLoss
from ultralytics.utils.tal import make_anchors


class ATRUMODOBBLoss(v8OBBLoss):
    """Native OBB loss with audited pixel filtering and border-safe GT masks.

    The upstream 8.3.0 tiny-box check multiplies normalized width by image
    height and normalized height by image width. Here width uses W, height
    uses H. Valid GT is identified by positive sides rather than a coordinate
    sum, because genuine border boxes can have negative centers.
    """

    def __init__(self, model: nn.Module, min_side_px: float = 2.0) -> None:
        """Initialize the unchanged native assigner and three loss terms.

        Args:
            model: Paired model with the native OBB head and loss gains.
            min_side_px: Explicit minimum side in model-input pixels.
        """
        super().__init__(model)
        self.min_side_px = float(min_side_px)

    def __call__(self, preds: Any, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute box/cls/DFL losses with the native rotated assigner.

        Args:
            preds: Native training (feature-list, angle) or evaluation tuple.
            batch: img [B,6,H,W], bboxes [N,5] normalized xywhr,
                cls [N,1], batch_idx [N].

        Returns:
            Scalar total loss and detached [3] component vector.
        """
        loss = torch.zeros(3, device=self.device)
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]
        pred_distri, pred_scores = torch.cat(
            [feature.view(batch_size, self.no, -1) for feature in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # 1. Recover actual pixel side lengths. Do not clip border centers.
        if batch["bboxes"].ndim != 2 or batch["bboxes"].shape[1] != 5:
            raise ValueError("OBB targets must have shape [N,5].")
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1),
                             batch["bboxes"]), 1)
        pixel_width = targets[:, 4] * imgsz[1]
        pixel_height = targets[:, 5] * imgsz[0]
        targets = targets[(pixel_width >= self.min_side_px) & (pixel_height >= self.min_side_px)]
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 5), 2)
        mask_gt = (gt_bboxes[..., 2:4] > 0).all(2, keepdim=True)

        # 2. The native rotated task-aligned assignment is unchanged.
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)
        assigned_boxes = pred_bboxes.detach().clone()
        assigned_boxes[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), assigned_boxes.type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        # 3. Native BCE classification, ProbIoU regression, and DFL.
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask,
            )
        else:
            loss[0] += (pred_angle * 0).sum()
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()
