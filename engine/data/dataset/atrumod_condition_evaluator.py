"""
engine/data/dataset/atrumod_condition_evaluator.py

ATR-UMOD 物理条件细分多指标评估器.
"""

import os
import json
from collections import defaultdict, OrderedDict
from typing import Dict, List, Any, Optional
import numpy as np
import torch

from .dota_evaluator import DotaEvaluator
from .mmeval import eval_rbbox_map, warmup_numba
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
            raw_filename = f"{idx:05d}.xml"
            if hasattr(self.dataset, 'img_ids') and idx < len(self.dataset.img_ids):
                raw_filename = f"{self.dataset.img_ids[idx]}.xml"

            xml_path = os.path.join(self.xml_folder, raw_filename)
            if not os.path.exists(xml_path):
                # 兼容性检索同名 xml
                xml_path = os.path.join(self.xml_folder, f"{idx}.xml")

            if os.path.exists(xml_path):
                conds = self.parser.parse_xml_conditions(xml_path)
            else:
                # 若找不到 xml，填充安全默认标签
                conds = {'AL': '[0, 120]', 'AN': '[0, 30]', 'TI': 'Afternoon', 'WE': 'Sunny', 'IL': 'Normal', 'SC': 'Road'}
            img_condition_map.append(conds)

        # 2. 首先计算全局整体指标
        mean_ap, eval_results = eval_rbbox_map(
            dets, gts, scale_ranges=None, iou_thr=0.5, use_07_metric=True, dataset=self.classes
        )
        self.stats = {'mAP': float(mean_ap)}
        print(f"\n🌟 全局总评测结果 Overall mAP@0.50: {mean_ap:.4f}")

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
                    condition_results[dim_key][sub_name] = {'mAP': 0.0, 'num_images': 0}
                    continue

                sub_dets = [dets[i] for i in matched_indices]
                sub_gts = [gts[i] for i in matched_indices]

                # 计算该子条件下的专用 mAP (关闭类别打印以保持界面整洁)
                sub_map, _ = eval_rbbox_map(
                    sub_dets, sub_gts, scale_ranges=None, iou_thr=0.5, use_07_metric=True, dataset=None
                )
                condition_results[dim_key][sub_name] = {
                    'mAP': float(sub_map),
                    'num_images': len(matched_indices)
                }

        # 4. 打印格式化学术大表 (与原论文 Table 严格一致)
        self._print_paper_style_table(condition_results, mean_ap)

        # 5. 保存 JSON 文件便于画图与撰写实验章节
        save_path = os.path.join(self.output_dir, "condition_eval_results.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                'overall_mAP': mean_ap,
                'conditions': condition_results
            }, f, indent=4, ensure_ascii=False)
        print(f"📁 细分条件评测指标已成功导出至: {save_path}\n")

    def _print_paper_style_table(self, res: OrderedDict, overall_map: float):
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
        print(f"│ {'Dimension':<20} │ {'Condition Subset':<22} │ {'Images':>10} │ {'mAP@0.50':>10} │")
        print("├" + "─" * 22 + "┼" + "─" * 24 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┤")

        for dim_key, sub_dict in res.items():
            first = True
            for sub_name, data in sub_dict.items():
                dim_str = dim_names[dim_key] if first else ""
                first = False
                map_str = f"{data['mAP'] * 100:.2f}%" if data['num_images'] > 0 else "N/A"
                print(f"│ {dim_str:<20} │ {sub_name:<22} │ {data['num_images']:>10d} │ {map_str:>10} │")
            print("├" + "─" * 22 + "┼" + "─" * 24 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┤")

        print(f"│ {'OVERALL':<20} │ {'All Test Images':<22} │ {len(self.img_ids):>10d} │ {overall_map * 100:>9.2f}% │")
        print("└" + "─" * 22 + "┴" + "─" * 24 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┘\n")