"""
engine/data/dataset/atrumod_condition_evaluator.py

ATR-UMOD 物理条件细分多指标评估器.
"""

import os
import json
from collections import OrderedDict
from typing import Dict, List, Optional
import numpy as np
import torch

from .dota_evaluator import DotaEvaluator
from .mmeval import eval_rbbox_map
from .atrumod_condition_parser import ATRUMODConditionParser
from ...core import register


@register()
class ATRUMODConditionEvaluator(DotaEvaluator):
    r"""专为 ATR-UMOD 数据集打造的多环境细分条件旋转目标评测器."""

    def __init__(
        self,
        dataset,
        xml_folder: Optional[str] = None,
        output_dir: Optional[str] = None,
        iou_types: Optional[List[str]] = None,
        iou_thrs: Optional[List[float]] = None,
    ) -> None:
        super().__init__(dataset=dataset, iou_types=iou_types, iou_thrs=iou_thrs)
        
        # 自动推导 XML 标注文件夹路径
        if xml_folder is not None:
            self.xml_folder = xml_folder
        elif hasattr(dataset, 'ann_folder'):
            self.xml_folder = dataset.ann_folder
        else:
            self.xml_folder = "./data/ATR-UMOD/test/labels_ir"

        self.output_dir = output_dir or "./outputs/eval_conditions"
        os.makedirs(self.output_dir, exist_ok=True)
        self.parser = ATRUMODConditionParser()

    def summarize(self):
        self.accumulate()
        if not self.img_ids:
            raise ValueError("未收集到预测，无法执行 ATR-UMOD 条件评估。")
        print("\n" + "=" * 80)
        print("🎯 开始执行 DOTA 旋转目标全局指标与 6 大物理条件细分基准评估...")
        print("=" * 80)

        # 1. 组装所有样本的基础 dets 与 gts
        dets = []
        gts = []
        img_condition_map: List[Dict[str, str]] = []

        for i, img_id in enumerate(self.img_ids):
            if isinstance(img_id, torch.Tensor):
                img_id = img_id.item()
            idx = int(img_id)
            target = self._get_target(idx)

            gt_bboxes = self._to_numpy(target['boxes']).copy()
            gt_labels = self._to_numpy(target['labels'])
            gt_bboxes_ignore = self._to_numpy(
                target.get('boxes_ignore', np.zeros((0, 5), dtype=np.float32))
            ).copy()
            gt_labels_ignore = self._to_numpy(
                target.get('labels_ignore', np.zeros((0,), dtype=np.int64))
            )

            # 角度归一化转弧度
            if gt_bboxes.size > 0:
                gt_bboxes[:, 4] = gt_bboxes[:, 4] * np.pi
            if gt_bboxes_ignore.size > 0:
                gt_bboxes_ignore[:, 4] = gt_bboxes_ignore[:, 4] * np.pi

            gts.append({
                'bboxes': gt_bboxes,
                'labels': gt_labels,
                'bboxes_ignore': gt_bboxes_ignore,
                'labels_ignore': gt_labels_ignore
            })

            pred = self.predictions[img_id]
            pred_bboxes = pred['bboxes']
            pred_scores = pred['scores']
            pred_labels = pred['labels']

            img_dets = []
            for cls_id in range(len(self.classes)):
                cls_mask = pred_labels == cls_id
                if np.any(cls_mask):
                    cls_bbox = pred_bboxes[cls_mask]
                    cls_score = pred_scores[cls_mask]
                    res = np.concatenate([cls_bbox, cls_score[:, None]], axis=1)
                    img_dets.append(res)
                else:
                    img_dets.append(np.zeros((0, 6), dtype=np.float32))
            dets.append(img_dets)

            # 提取当前样本对应的 XML 条件元数据
            # 尝试通过 img_id 获取原始真实文件名
            if hasattr(self.dataset, 'img_ids'):
                raw_filename = f"{self.dataset.img_ids[idx]}.xml"
                xml_path = os.path.join(self.xml_folder, raw_filename)
            else:
                xml_path = os.path.join(self.xml_folder, f"{idx:05d}.xml")
                if not os.path.exists(xml_path):
                    xml_path = os.path.join(self.xml_folder, f"{idx}.xml")

            # 已知真实文件名时不回退到索引命名，以免读取另一张图的条件。
            conds = self.parser.parse_xml_conditions(xml_path)
            img_condition_map.append(conds)

        unknown_counts = OrderedDict(
            (dim_key, sum(cond[dim_key] == self.parser.UNKNOWN for cond in img_condition_map))
            for dim_key in self.parser.CONDITIONS_SCHEMA
        )
        num_unknown_images = sum(
            any(value == self.parser.UNKNOWN for value in cond.values())
            for cond in img_condition_map
        )
        unknown_summary = ', '.join(f"{key}={count}" for key, count in unknown_counts.items())
        print(f"Unknown 条件图像: {num_unknown_images}; 按维度计数: {unknown_summary}")

        # 2. 首先计算全局整体指标
        overall_ap_by_iou, eval_results_by_iou = self._evaluate_maps(
            dets, gts, show_classes=True
        )
        mean_ap = float(np.mean(list(overall_ap_by_iou.values())))
        self.stats = {'mAP': mean_ap}
        self.eval_results = (
            next(iter(eval_results_by_iou.values()))
            if len(eval_results_by_iou) == 1 else eval_results_by_iou
        )
        for iou_thr, overall_ap in overall_ap_by_iou.items():
            print(f"\n🌟 全局总评测结果 Overall mAP@{iou_thr:.2f}: {overall_ap:.4f}")
        if len(overall_ap_by_iou) > 1:
            print(f"配置 IoU 阈值 {self.iou_thrs} 的平均 mAP: {mean_ap:.4f}")

        # 3. 按 6 大维度 21 个子条件进行切片求值
        print("\n" + "=" * 80)
        print("📊 正在切片计算 ATR-UMOD 物理环境细分条件指标...")
        print("=" * 80)

        condition_results = OrderedDict()

        for dim_key, sub_conds in self.parser.CONDITIONS_SCHEMA.items():
            condition_results[dim_key] = OrderedDict()
            for sub_name in sub_conds:
                # 筛选属于当前子条件的样本索引
                matched_indices = [
                    i for i, cond in enumerate(img_condition_map)
                    if cond[dim_key] == sub_name
                ]

                if len(matched_indices) == 0:
                    sub_ap_by_iou = OrderedDict((thr, 0.0) for thr in self.iou_thrs)
                else:
                    sub_dets = [dets[i] for i in matched_indices]
                    sub_gts = [gts[i] for i in matched_indices]

                    # Unknown 只排除缺失维度的分组，图像仍参与 Overall 和其余维度。
                    sub_ap_by_iou, _ = self._evaluate_maps(sub_dets, sub_gts)
                condition_results[dim_key][sub_name] = {
                    'mAP': float(np.mean(list(sub_ap_by_iou.values()))),
                    'num_images': len(matched_indices),
                    'mAP_by_iou': sub_ap_by_iou,
                }

        # 4. 打印格式化学术大表 (与原论文 Table 严格一致)
        for iou_thr, overall_ap in overall_ap_by_iou.items():
            self._print_paper_style_table(condition_results, overall_ap, iou_thr=iou_thr)

        # 5. 保存 JSON 文件便于画图与撰写实验章节
        save_path = os.path.join(self.output_dir, "condition_eval_results.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                'overall_mAP': mean_ap,
                'conditions': condition_results,
                'iou_thrs': self.iou_thrs,
                'overall_mAP_by_iou': overall_ap_by_iou,
                'unknown_counts': unknown_counts,
                'num_unknown_images': num_unknown_images,
            }, f, indent=4, ensure_ascii=False)
        print(f"📁 细分条件评测指标已成功导出至: {save_path}\n")

    def _evaluate_maps(self, dets, gts, show_classes=False):
        """按配置阈值计算 VOC07 AP；多阈值时 mAP 为这些阈值的平均。"""
        per_iou_ap = OrderedDict()
        per_iou_results = OrderedDict()
        for iou_thr in self.iou_thrs:
            mean_ap, eval_results = eval_rbbox_map(
                dets, gts, scale_ranges=None, iou_thr=iou_thr, use_07_metric=True,
                dataset=self.classes if show_classes and len(self.iou_thrs) == 1 else None,
            )
            per_iou_ap[iou_thr] = float(mean_ap)
            per_iou_results[iou_thr] = eval_results
        return per_iou_ap, per_iou_results

    def _print_paper_style_table(self, res: OrderedDict, overall_map: float, iou_thr: float = 0.5):
        r"""打印学术期刊级排版对比表格."""
        dim_names = {
            'AL': 'Altitude (m)',
            'AN': 'Angle (°)',
            'TI': 'Time',
            'WE': 'Weather',
            'IL': 'Illumination',
            'SC': 'Scenario'
        }

        print("\n" + "┌" + "─" * 22 + "┬" + "─" * 24 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┐")
        metric_name = f"mAP@{iou_thr:.2f}"
        print(f"│ {'Dimension':<20} │ {'Condition Subset':<22} │ {'Images':>10} │ {metric_name:>10} │")
        print("├" + "─" * 22 + "┼" + "─" * 24 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┤")

        for dim_key, sub_dict in res.items():
            first = True
            for sub_name, data in sub_dict.items():
                dim_str = dim_names[dim_key] if first else ""
                first = False
                sub_map = data['mAP_by_iou'][iou_thr]
                map_str = f"{sub_map * 100:.2f}%" if data['num_images'] > 0 else "N/A"
                print(f"│ {dim_str:<20} │ {sub_name:<22} │ {data['num_images']:>10d} │ {map_str:>10} │")
            print("├" + "─" * 22 + "┼" + "─" * 24 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┤")

        print(f"│ {'OVERALL':<20} │ {'All Test Images':<22} │ {len(self.img_ids):>10d} │ {overall_map * 100:>9.2f}% │")
        print("└" + "─" * 22 + "┴" + "─" * 24 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┘\n")
