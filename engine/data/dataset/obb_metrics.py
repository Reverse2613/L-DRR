"""旋转检测指标的阈值与命名规则；不改变现有 VOC07 AP 算法。

AP50/AP75 指类别平均 AP；AP50_95 指 0.50:0.05:0.95 十个 AP 的平均。
这里仍采用本仓库 VOC2007 11 点插值，不能将结果称为 COCO 默认 AP。
"""

from collections import OrderedDict
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple


OBB_FULL_IOU_THRESHOLDS: Tuple[float, ...] = tuple(
    round(0.50 + 0.05 * index, 2) for index in range(10)
)


def resolve_iou_thresholds(
    iou_thrs: Optional[Sequence[float]], report_full_metrics: bool = False,
) -> Tuple[float, ...]:
    """校验阈值，保留旧默认 AP50，显式启用十阈值评测。

    Args:
        iou_thrs: 指定的 IoU 阈值；None 表示使用相应默认值。
        report_full_metrics: True 时必须评测完整的十阈值序列。

    Returns:
        排序后、无重复的有效阈值。

    Raises:
        ValueError: 空序列、重复、非有限、越界或不完整的十阈值配置。
    """
    if iou_thrs is None:
        return OBB_FULL_IOU_THRESHOLDS if report_full_metrics else (0.5,)
    values = tuple(float(value) for value in iou_thrs)
    if not values or any(not math.isfinite(value) or not 0 < value <= 1 for value in values):
        raise ValueError(f"IoU thresholds must be finite and in (0, 1]: {values}")
    if len(set(round(value, 8) for value in values)) != len(values):
        raise ValueError(f"Duplicate IoU thresholds: {values}")
    values = tuple(sorted(values))
    if report_full_metrics and not is_full_iou_grid(values):
        raise ValueError("report_full_metrics=True requires 0.50:0.05:0.95 (10 thresholds)")
    return values


def is_full_iou_grid(values: Sequence[float]) -> bool:
    """判断是否恰好包含十阈值，防止将两阈值均值误称 AP50_95。

    Args:
        values: 待检查的 IoU 阈值序列。

    Returns:
        是否与标准十阈值序列一致，容许浮点计算误差。
    """
    return len(values) == 10 and all(
        math.isclose(left, right, abs_tol=1e-8)
        for left, right in zip(sorted(values), OBB_FULL_IOU_THRESHOLDS)
    )


def metrics_from_iou_maps(
    per_iou_ap: Mapping[float, float],
) -> Dict[str, float]:
    """汇总指标，兼容旧 mAP 键；mAP 始终优先指 AP50。

    Args:
        per_iou_ap: IoU → 类别平均 AP，值为 [0,1] 比例。

    Returns:
        单阈值保留旧的 {'mAP': value}；多阈值包含可用的 AP50/AP75。
        仅完整十阈值才输出 AP50_95，否则输出 AP_mean_configured。

    Raises:
        ValueError: 没有指标、AP 非有限或不在 [0,1] 内。
    """
    if not per_iou_ap:
        raise ValueError("Cannot summarize an empty IoU/AP mapping")
    values = OrderedDict((float(key), float(value)) for key, value in per_iou_ap.items())
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
        raise ValueError(f"Invalid AP values: {values}")
    if len(values) == 1:
        return {"mAP": next(iter(values.values()))}
    mean_ap = sum(values.values()) / len(values)
    result: Dict[str, float] = {}
    for target, name in ((0.50, "AP50"), (0.75, "AP75")):
        matches = [value for key, value in values.items() if math.isclose(key, target, abs_tol=1e-8)]
        if matches:
            result[name] = matches[0]
    # 旧 solver 使用 mAP 选 checkpoint；新增严格 IoU 指标不应改变选模目标。
    result = {"mAP": result.get("AP50", mean_ap), **result}
    result["AP50_95" if is_full_iou_grid(tuple(values)) else "AP_mean_configured"] = mean_ap
    return result
