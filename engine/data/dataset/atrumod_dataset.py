"""
engine/data/dataset/atrumod_dataset.py

ATR-UMOD 多模态成对 (RGB + TIR) 目标检测数据集加载器 .
"""

import os
import re
import math
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile

from ._dataset import DetDataset
from ...core import register

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


@register()
class ATRUMODPairDataset(DetDataset):
    r"""ATR-UMOD 成对多模态旋转目标检测数据集加载器."""

    __inject__ = ["transforms"]

    # 经 data_pre/inspect_atrumod.py 全量验证确认的 11 类严格字符串？tuple，list，arr，dict，optional区别
    CLASSES: Tuple[str, ...] = (
        "car",
        "suv",
        "van",
        "bus",
        "freight_car",
        "truck",
        "motorcycle",
        "trailer",
        "excavator",
        "crane",
        "tank_truck",
    )

    def __init__(
        self,
        img_folder: str,
        ann_folder: str,
        img_folder_rgb: Optional[str] = None,
        transforms: Optional[Any] = None,
        return_masks: bool = False,
        max_retry: int = 5,
        strict_conditions: bool = False,
        strict_loading: bool = False,
    ) -> None:
        r"""初始化图像配对与条件解析策略。

        Args:
            img_folder: IR图像目录。
            ann_folder: IR参考框与六条件XML目录。
            img_folder_rgb: RGB图像目录；未设置时按现有目录规则推导。
            transforms: 成对图像/框的预处理。
            return_masks: 保留现有接口。
            max_retry: 旧模式的随机重试次数。
            strict_conditions: PG模式严格核验六条件，缺失/非法立即报错。
            strict_loading: 读取失败立即报错，禁止随机换图影响评估ID。
        """
        super().__init__()
        self.img_folder_ir: str = os.path.abspath(img_folder)
        self.ann_folder: str = os.path.abspath(ann_folder)
        self._transforms = transforms
        self.return_masks: bool = return_masks
        self.max_retry: int = max_retry
        self.strict_conditions: bool = strict_conditions
        self.strict_loading: bool = strict_loading

        if img_folder_rgb is not None and os.path.exists(img_folder_rgb):
            self.img_folder_rgb: str = os.path.abspath(img_folder_rgb)
        else:
            self.img_folder_rgb = self._deduce_rgb_folder(self.img_folder_ir)

        self.category2label: Dict[str, int] = {
            cls_name: idx for idx, cls_name in enumerate(self.CLASSES)
        }

        self.img_ids: List[str] = self._load_and_verify_triplets()
        logger.info(f"ATR-UMOD 数据集就绪，健康有效配对数: {len(self.img_ids)}")

    @staticmethod
    def _deduce_rgb_folder(ir_folder: str) -> str:
        parent_dir = os.path.dirname(ir_folder)
        base_name = os.path.basename(ir_folder)

        candidate_names = ["images", "images_rgb", "images_color", "rgb"]
        if base_name.endswith("_ir"):
            candidate_names.insert(0, base_name[:-3])#？

        for candidate in candidate_names:
            rgb_path = os.path.join(parent_dir, candidate)
            if os.path.exists(rgb_path):
                return rgb_path

        raise FileNotFoundError(f"无法推导可见光目录，请显式传入 img_folder_rgb: {ir_folder}")

    def _find_file(self, folder: str, file_id: str, extensions: List[str]) -> Optional[str]:
        for ext in extensions:
            file_path = os.path.join(folder, f"{file_id}{ext}")
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                return file_path
        return None

    def _load_and_verify_triplets(self) -> List[str]:
        r"""核对同名IR/RGB/XML配对，严格模式缺配对即报错。

        Returns:
            按文件名排序的有效样本ID，不重新编号文件内容。
        """
        if not os.path.exists(self.img_folder_ir):
            raise FileNotFoundError(f"红外目录不存在: {self.img_folder_ir}")

        candidates = [
            os.path.splitext(f)[0]#？
            for f in os.listdir(self.img_folder_ir)
            if f.lower().endswith((".jpg", ".png", ".bmp", ".jpeg"))
        ]
        if self.strict_loading and len(candidates) != len(set(candidates)):
            raise ValueError(f"{self.img_folder_ir}: 同一ID存在多个图像扩展名，评估ID不唯一。")

        valid_ids: List[str] = []
        image_exts = [".jpg", ".png", ".bmp", ".jpeg"]
        xml_exts = [".xml"]

        for file_id in sorted(candidates):
            ir_p = self._find_file(self.img_folder_ir, file_id, image_exts)
            rgb_p = self._find_file(self.img_folder_rgb, file_id, image_exts)
            ann_p = self._find_file(self.ann_folder, file_id, xml_exts)

            if ir_p and rgb_p and ann_p:
                valid_ids.append(file_id)
            elif self.strict_loading:
                raise FileNotFoundError(
                    f"样本{file_id}配对不完整：IR={ir_p}, RGB={rgb_p}, XML={ann_p}。"
                )

        return valid_ids

    def __len__(self) -> int:
        return len(self.img_ids)

    @staticmethod
    def _parse_time_harmonic(time_str: str) -> Tuple[float, float]:
        r"""将时间戳解析为圆周拓扑 (sin, cos) 谐波特征，消除午夜断点.

        物理映射关系:
            - 00:00 (午夜) -> phi = 0     -> sin =  0.0, cos = +1.0
            - 06:00 (清晨) -> phi = pi/2  -> sin = +1.0, cos =  0.0
            - 12:00 (正午) -> phi = pi    -> sin =  0.0, cos = -1.0
            - 18:00 (日落) -> phi = 3pi/2 -> sin = -1.0, cos =  0.0
        """
        try:
            match = re.search(r"T?(\d{1,2}):(\d{1,2})", time_str)
            if match:
                hour = float(match.group(1))
                minute = float(match.group(2))
                angle_rad = 2.0 * math.pi * ((hour + minute / 60.0) / 24.0)
                # 使用 round 避免浮点数精度漂移 (如 1e-16)
                return float(math.sin(angle_rad)), float(math.cos(angle_rad))
        except Exception as e:
            logger.error(f"时间格式解析异常: '{time_str}', 原因: {e}")

        # 异常默认回退为正午 12:00 (phi = pi -> sin=0.0, cos=-1.0)
        return 0.0, -1.0

    def _parse_conditions(self, root: ET.Element, xml_path: str) -> Dict[str, torch.Tensor]:
        r"""解析图像级六条件，PG严格模式不补默认值或裁剪。

        Args:
            root: 一张图像的XML根节点。
            xml_path: 用于错误定位的XML路径。

        Returns:
            height/angle各[1]、time[2] float32，三个类别为int64标量。
            PG严格模式另保留height_m/angle_deg各[1]，供后续GC读取原始单位值。

        Raises:
            ValueError: 严格模式下字段缺失、非有限、越界或时间格式非法。
        """
        if self.strict_conditions:
            fields = ("UAVheight", "UAVangle", "time", "weather", "illumination", "location")
            raw = {field: (root.findtext(field) or "").strip() for field in fields}
            missing = [field for field, value in raw.items() if not value]
            if missing:
                raise ValueError(f"{xml_path}: 缺少显式条件{', '.join(missing)}。")
            try:
                raw_height, raw_angle = float(raw["UAVheight"]), float(raw["UAVangle"])
                weather_id, illum_id, loc_id = (int(raw[field]) for field in fields[3:])
            except ValueError as error:
                raise ValueError(f"{xml_path}: 高度/倾角或类别编号不是合法数值。") from error
            if not math.isfinite(raw_height) or not 80.0 <= raw_height <= 300.0:
                raise ValueError(f"{xml_path}: UAVheight={raw_height}不在80–300 m。")
            if not math.isfinite(raw_angle) or not 0.0 <= raw_angle <= 75.0:
                raise ValueError(f"{xml_path}: UAVangle={raw_angle}不在0–75°。")
            for field, value, count in (("weather", weather_id, 10),
                                        ("illumination", illum_id, 7), ("location", loc_id, 11)):
                if not 0 <= value < count:
                    raise ValueError(f"{xml_path}: {field}={value}不在0–{count - 1}。")
            match = re.search(r"(?:T|^)(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$", raw["time"])
            if match is None:
                raise ValueError(f"{xml_path}: time={raw['time']!r}不是合法时刻。")
            hour, minute = int(match.group(1)), int(match.group(2))
            second = int(match.group(3) or 0)
            if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
                raise ValueError(f"{xml_path}: time={raw['time']!r}时分秒越界。")
            phase = 2.0 * math.pi * (hour + minute / 60.0 + second / 3600.0) / 24.0
            time_sin, time_cos = math.sin(phase), math.cos(phase)
            height_norm, angle_norm = (raw_height - 80.0) / 220.0, raw_angle / 75.0
        else:
            # 保留历史模型的数据行为；新PG配置显式启用严格模式。
            raw_height = float(root.findtext("UAVheight", default="120.0"))
            raw_angle = float(root.findtext("UAVangle", default="45.0"))
            height_norm = float(np.clip((raw_height - 80.0) / 220.0, 0.0, 1.0))
            angle_norm = float(np.clip(raw_angle / 75.0, 0.0, 1.0))
            time_sin, time_cos = self._parse_time_harmonic(root.findtext("time", default="12:00"))
            weather_id = int(np.clip(int(root.findtext("weather", default="0")), 0, 9))
            illum_id = int(np.clip(int(root.findtext("illumination", default="1")), 0, 6))
            loc_id = int(np.clip(int(root.findtext("location", default="0")), 0, 10))
        conditions = {
            "height": torch.tensor([height_norm], dtype=torch.float32),
            "angle": torch.tensor([angle_norm], dtype=torch.float32),
            "time": torch.tensor([time_sin, time_cos], dtype=torch.float32),
            "weather": torch.tensor(weather_id, dtype=torch.int64),
            "illumination": torch.tensor(illum_id, dtype=torch.int64),
            "location": torch.tensor(loc_id, dtype=torch.int64),
        }
        if self.strict_conditions:
            conditions["height_m"] = torch.tensor([raw_height], dtype=torch.float32)
            conditions["angle_deg"] = torch.tensor([raw_angle], dtype=torch.float32)
        return conditions

    def _parse_xml(self, xml_path: str, orig_w: int, orig_h: int) -> Dict[str, Any]:
        r"""解析 XML 中的旋转目标框及 6 维物理属性."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        conditions = self._parse_conditions(root, xml_path)

        # 2. 目标旋转框解析
        boxes: List[List[float]] = []
        labels: List[int] = []

        for obj in root.findall("object"):
            cls_name = obj.findtext("name", "").strip().lower()

            # 强类型拦截防御：如果出现未知类别立即报警，绝不静默跳过
            if cls_name not in self.category2label:
                logger.error(f"在文件 {xml_path} 中检测到未注册的未知类别: '{cls_name}'！")
                if self.strict_loading:
                    raise ValueError(f"{xml_path}: 未注册目标类别{cls_name!r}，不跳过该目标。")
                continue

            robndbox = obj.find("robndbox")
            polygon = obj.find("polygon")

            if robndbox is not None:
                cx = float(robndbox.findtext("cx", "0"))
                cy = float(robndbox.findtext("cy", "0"))
                w_box = float(robndbox.findtext("w", "0"))
                h_box = float(robndbox.findtext("h", "0"))
                angle_deg = float(robndbox.findtext("angle", "0"))
                #？
                if w_box <= 1.0 or h_box <= 1.0:
                    continue

                if w_box < h_box:
                    w_box, h_box = h_box, w_box
                    angle_deg += 90.0

                angle_deg = angle_deg % 180.0
                angle_norm_val = angle_deg / 180.0  # 映射到 [0, 1)

                boxes.append([cx, cy, w_box, h_box, angle_norm_val])
                labels.append(self.category2label[cls_name])

            elif polygon is not None:
                pts = [
                    [float(polygon.findtext("x1")), float(polygon.findtext("y1"))],
                    [float(polygon.findtext("x2")), float(polygon.findtext("y2"))],
                    [float(polygon.findtext("x3")), float(polygon.findtext("y3"))],
                    [float(polygon.findtext("x4")), float(polygon.findtext("y4"))],
                ]
                rect = cv2.minAreaRect(np.array(pts, dtype=np.float32))
                (cx, cy), (w_box, h_box), angle_deg = rect

                if w_box <= 1.0 or h_box <= 1.0:
                    continue
                if w_box < h_box:
                    w_box, h_box = h_box, w_box
                    angle_deg += 90.0

                angle_deg = angle_deg % 180.0
                angle_norm_val = angle_deg / 180.0

                boxes.append([cx, cy, w_box, h_box, angle_norm_val])
                labels.append(self.category2label[cls_name])

        target: Dict[str, Any] = {
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            "size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
            "conditions": conditions,
        }

        if len(boxes) > 0:
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 5), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        return target

    def _load_single_pair(self, idx: int) -> Tuple[Dict[str, Image.Image], Dict[str, Any]]:
        file_id: str = self.img_ids[idx]
        image_exts = [".jpg", ".png", ".bmp", ".jpeg"]

        ir_path = self._find_file(self.img_folder_ir, file_id, image_exts)
        rgb_path = self._find_file(self.img_folder_rgb, file_id, image_exts)
        xml_path = os.path.join(self.ann_folder, f"{file_id}.xml")

        img_ir: Image.Image = Image.open(ir_path).convert("RGB")
        img_rgb: Image.Image = Image.open(rgb_path).convert("RGB")

        if img_ir.size != img_rgb.size:
            img_rgb = img_rgb.resize(img_ir.size, Image.BILINEAR)

        w, h = img_ir.size
        target = self._parse_xml(xml_path, orig_w=w, orig_h=h)
        target["image_id"] = torch.tensor([idx])
        target["idx"] = torch.tensor([idx])

        return {"ir": img_ir, "rgb": img_rgb}, target

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        r"""读取一个固定ID的样本，并执行配置的成对预处理。

        Args:
            idx: img_ids中的位置。

        Returns:
            成对图像（预处理后可为[6,H,W]张量）及含条件的target。
        """
        if self.strict_loading:
            # 同一个评估image_id必须对应同一张图；失败时不能随机替换。
            try:
                paired_imgs, target = self._load_single_pair(idx)
                if self._transforms is not None:
                    result = self._transforms(paired_imgs, target, self)
                    paired_imgs, target = result[0], result[1]
                return paired_imgs, target
            except Exception:
                logger.exception("严格读取样本%s失败，保留原ID并中止。", self.img_ids[idx])
                raise
        for attempt in range(self.max_retry):
            try:
                paired_imgs, target = self._load_single_pair(idx)
                if self._transforms is not None:
                    res = self._transforms(paired_imgs, target, self)
                    paired_imgs, target = res[0], res[1]
                return paired_imgs, target
            except Exception as e:
                if self.strict_conditions:
                    logger.exception("显式条件样本%s读取失败，不进行随机替换。", self.img_ids[idx])
                    raise
                logger.error(f"读取样本 {self.img_ids[idx]} 异常: {e}. 启动重采样...")
                idx = np.random.randint(0, len(self.img_ids))
        raise RuntimeError(f"样本读取重试 {self.max_retry} 次仍失败！")

    def load_item(self, idx: int) -> Tuple[Dict[str, Image.Image], Dict[str, Any]]:
        return self._load_single_pair(idx)

    def get_ann_info(self, idx: int) -> Dict[str, Any]:
        file_id: str = self.img_ids[idx]
        xml_path = os.path.join(self.ann_folder, f"{file_id}.xml")
        target = self._parse_xml(xml_path, orig_w=640, orig_h=512)
        target["image_id"] = torch.tensor([idx])
        target["idx"] = torch.tensor([idx])
        return target

# # ==============================================================================
# # 本地单文件测试入口 
# # ==============================================================================
# if __name__ == "__main__":
#     import sys

#     print("\n" + "=" * 60)
#     print(">>> 启动 ATRUMODPairDataset 单元测试与张量维度检查...")
#     print("=" * 60)

#     # 替换为你本地真实的测试路径
#     test_ir = "/home/reverse/workspace/data/ATR-UMOD/test/images_ir"
#     test_ann = "/home/reverse/workspace/data/ATR-UMOD/test/labels_ir"

#     if not os.path.exists(test_ir):
#         print(f"[提示] 未检测到指定路径: {test_ir}，请在脚本中修改为您本地实际路径后运行。")
#         sys.exit(0)

#     dataset = ATRUMODPairDataset(img_folder=test_ir, ann_folder=test_ann)
#     print(f"成功实例化数据集！有效样本数量: {len(dataset)}")

#     sample_imgs, sample_target = dataset[0]
#     print(f"\n[验证 1] 图像类型: IR={type(sample_imgs['ir'])}, RGB={type(sample_imgs['rgb'])}")
#     print(f"[验证 2] 旋转框形态: {sample_target['boxes'].shape} (cx, cy, w, h, angle_norm)")
#     print(f"[验证 3] 目标类别标签: {sample_target['labels']}")
#     print(f"[验证 4] 6 维条件先验值:")
#     for k, v in sample_target["conditions"].items():
#         print(f"         - {k:<15s}: {v}")

#     print("\n>>> ATRUMODPairDataset 单元测试全部通过！\n")
