import os
import torch
import torch.utils.data
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from PIL import Image

from ._dataset import DetDataset
from ...core import register

@register()
class DOTADataset(DetDataset):
    __inject__ = ['transforms']

    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
               'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
               'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter')

    def __init__(self, img_folder, ann_folder, transforms=None, return_masks=False):
        super().__init__()
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self._transforms = transforms
        self.return_masks = return_masks

        self.category2label = {k: i for i, k in enumerate(self.CLASSES)}
        self.img_ids = self._load_img_ids()

    def _load_img_ids(self):
        img_ids = []
        if not os.path.exists(self.img_folder):
            return img_ids

        for filename in os.listdir(self.img_folder):
            if filename.lower().endswith(('.jpg', '.png', '.bmp', '.tif', '.tiff')):
                img_ids.append(os.path.splitext(filename)[0])
        return sorted(img_ids)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        img_id = self.img_ids[idx]
        # Try different extensions
        img_path = None
        for ext in ['.png', '.jpg', '.bmp', '.tif', '.tiff']:
            p = os.path.join(self.img_folder, f'{img_id}{ext}')
            if os.path.exists(p):
                img_path = p
                break

        if img_path is None:
             raise FileNotFoundError(f"Image not found for id {img_id}")

        txt_path = os.path.join(self.ann_folder, f'{img_id}.txt')

        # Load image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Parse Text Annotation
        target = self._parse_txt(txt_path, w, h)
        target['image_id'] = torch.tensor([idx])
        target['idx'] = torch.tensor([idx])

        return img, target

    def get_ann_info(self, idx):
        img_id = self.img_ids[idx]
        txt_path = os.path.join(self.ann_folder, f'{img_id}.txt')
        target = self._parse_txt(txt_path, 0, 0)
        target['image_id'] = torch.tensor([idx])
        target['idx'] = torch.tensor([idx])
        return target

    def _parse_txt(self, txt_path, w, h):
        boxes = []
        labels = []

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    # DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
                    if len(parts) < 9:
                        continue

                    category = parts[8]
                    if category not in self.category2label:
                        continue

                    poly = list(map(float, parts[:8]))

                    # Convert poly to rotated rect (cx, cy, w, h, angle)
                    poly_np = np.array(poly, dtype=np.float32).reshape(4, 2)
                    rect = cv2.minAreaRect(poly_np)
                    (cx, cy), (w_box, h_box), angle = rect

                    # Filter invalid boxes
                    if w_box <= 1 or h_box <= 1:
                        continue

                    if w_box < h_box:
                        w_box, h_box = h_box, w_box
                        angle += 90.0
                    angle = angle % 180.0
                    angle_norm = angle / 180.0
                    boxes.append([cx, cy, w_box, h_box, angle_norm])
                    labels.append(self.category2label[category])

        if len(boxes) > 0:
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'orig_size': torch.tensor([h, w]),
                'size': torch.tensor([h, w])
            }
        else:
             target = {
                'boxes': torch.zeros((0, 5), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'orig_size': torch.tensor([h, w]),
                'size': torch.tensor([h, w])
            }

        return target





# ==============================================================================
# DroneVehicle 专属数据集加载器 (继承自 DOTADataset)
# ==============================================================================
@register()
class DroneVehicleDataset(DOTADataset):
    r"""DroneVehicle 红外/可见光旋转目标数据集加载器.

    继承自标准 DOTADataset, 覆写其 CLASSES 元组, 将 DOTA 原生的 15 类别替换为
    DroneVehicle 官方基准的 5 类机动车分类.

    类别映射关系:
        0: car (轿车)
        1: truck (卡车)
        2: bus (大巴/公交)
        3: van (面包车/厢式客车)
        4: freight_car (货运车/集装箱车)
    """

    CLASSES = ("car", "truck", "bus", "van", "freight_car")

    def __init__(
        self,
        img_folder: str,
        ann_folder: str,
        transforms=None,
        return_masks: bool = False,
    ) -> None:
        """初始化 DroneVehicleDataset.

        Args:
            img_folder (str): 图像文件存储目录绝对路径.
            ann_folder (str): DOTA 格式标签文件存储目录绝对路径.
            transforms (Optional[Callable]): 数据增强变换流水线.
            return_masks (bool): 是否返回实例分割掩码, 目标检测任务默认为 False.
        """
        # 调用父类初始化, 此时 self.CLASSES 已经被覆写为 5 类别,
        # self.category2label 会自动生成 0~4 的正确索引映射
        super().__init__(
            img_folder=img_folder,
            ann_folder=ann_folder,
            transforms=transforms,
            return_masks=return_masks,
        )    





# ==============================================================================
# DroneVehicle 多模态成对 (RGB + TIR) 高鲁棒性防御型数据加载器
# ==============================================================================
import logging
from PIL import Image, ImageFile

# 开启 PIL 截断图片容忍保护，防止偶发损坏图片导致训练中断
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


@register()
class DroneVehiclePairDataset(DetDataset):
    r"""DroneVehicle 成对 (RGB + 红外) 防御型目标检测数据集加载器.

    特点:
        1. 启动时执行三方 (IR图, RGB图, 文本标签) 严格存在性与非空校验;
        2. 运行期支持 try-except 容错与自愈式重采样, 避免训练中途崩溃;
        3. 自动校验并强制统一 RGB 与 IR 的几何物理尺寸;
        4. 严格解析并输出标准 5 类旋转目标标注.
    """
    __inject__ = ["transforms"]  # <--- 加上这一行！告诉框架必须实例化 Compose 对象！
    CLASSES = ("car", "truck", "bus", "van", "freight_car")

    def __init__(
        self,
        img_folder: str,
        ann_folder: str,
        transforms=None,
        return_masks: bool = False,
        max_retry: int = 5,
    ) -> None:
        """初始化成对数据集加载器.

        Args:
            img_folder (str): 红外图像所在目录路径 (如 /dev/shm/.../trainimgr).
            ann_folder (str): DOTA 文本标签文件所在目录路径.
            transforms (Optional[Callable]): 作用于图像与标签的数据增强流水线.
            return_masks (bool): 是否返回掩码, 目标检测默认为 False.
            max_retry (int): 单次样本加载失败时的最大自愈重试次数, 默认 5 次.
        """
        super().__init__()
        self.img_folder_ir: str = img_folder
        self.ann_folder: str = ann_folder
        self._transforms = transforms
        self.return_masks: bool = return_masks
        self.max_retry: int = max_retry

        # 规则化推导对应的可见光 (RGB) 图像目录
        clean_ir_path: str = self.img_folder_ir.rstrip("/")
        if clean_ir_path.endswith("r"):
            self.img_folder_rgb: str = clean_ir_path[:-1]
        else:
            raise ValueError(
                f"红外图像目录命名不合规范，预期以 'r' 结尾 (如 trainimgr), 实际为: {self.img_folder_ir}"
            )

        if not os.path.exists(self.img_folder_rgb):
            raise FileNotFoundError(f"推导的可见光目录不存在: {self.img_folder_rgb}")

        self.category2label: Dict[str, int] = {
            k: i for i, k in enumerate(self.CLASSES)
        }
        # 严格过滤出三方均有效存在的样本 ID 列表
        self.img_ids: List[str] = self._load_and_verify_img_ids()

    def _find_image_path(self, folder: str, img_id: str) -> Optional[str]:
        """安全检索指定目录下支持的所有图片扩展名."""
        for ext in [".jpg", ".png", ".bmp", ".jpeg"]:
            p = os.path.join(folder, f"{img_id}{ext}")
            # 必须存在且文件大小大于 0 字节
            if os.path.exists(p) and os.path.getsize(p) > 0:
                return p
        return None

    def _load_and_verify_img_ids(self) -> List[str]:
        """初始化时执行三方防御性快筛，排除任何成对缺失或损坏的文件."""
        valid_ids: List[str] = []
        if not os.path.exists(self.img_folder_ir):
            logger.error(f"红外目录不存在: {self.img_folder_ir}")
            return valid_ids

        all_candidates: List[str] = [
            os.path.splitext(f)[0]
            for f in os.listdir(self.img_folder_ir)
            if f.lower().endswith((".jpg", ".png", ".bmp", ".jpeg"))
        ]

        corrupted_count = 0
        for img_id in sorted(all_candidates):
            ir_path = self._find_image_path(self.img_folder_ir, img_id)
            rgb_path = self._find_image_path(self.img_folder_rgb, img_id)
            ann_path = os.path.join(self.ann_folder, f"{img_id}.txt")

            # 严格校验三方必须同时有效存在
            if (
                ir_path is not None
                and rgb_path is not None
                and os.path.exists(ann_path)
            ):
                valid_ids.append(img_id)
            else:
                corrupted_count += 1

        if corrupted_count > 0:
            logger.warning(
                f"[{self.img_folder_ir}] 防御机制过滤掉 {corrupted_count} 个缺失或损坏样本！"
            )

        logger.info(f"最终确认健康成对样本总数: {len(valid_ids)}")
        return valid_ids

    def __len__(self) -> int:
        return len(self.img_ids)

    def _load_single_pair(
        self, idx: int
    ) -> Tuple[Dict[str, Image.Image], dict]:
        """底层的单样本物理读取与安全对齐逻辑."""
        img_id: str = self.img_ids[idx]

        ir_path = self._find_image_path(self.img_folder_ir, img_id)
        rgb_path = self._find_image_path(self.img_folder_rgb, img_id)

        if ir_path is None or rgb_path is None:
            raise IOError(f"样本物理文件读取失败: ID={img_id}")

        # 使用 PIL 读取并统一转为标准 3 通道 RGB 模式
        img_ir: Image.Image = Image.open(ir_path).convert("RGB")
        img_rgb: Image.Image = Image.open(rgb_path).convert("RGB")

        # 几何尺寸安全对齐防御
        if img_ir.size != img_rgb.size:
            logger.warning(
                f"样本 {img_id} 尺寸不一致: IR={img_ir.size}, RGB={img_rgb.size}. 强制以 IR 尺寸重采样对齐!"
            )
            img_rgb = img_rgb.resize(img_ir.size, Image.BILINEAR)

        w, h = img_ir.size

        # 读取并解析 DOTA 格式标签
        txt_path: str = os.path.join(self.ann_folder, f"{img_id}.txt")
        target: dict = self._parse_txt(txt_path, w, h)
        target["image_id"] = torch.tensor([idx])
        target["idx"] = torch.tensor([idx])

        paired_imgs: Dict[str, Image.Image] = {"ir": img_ir, "rgb": img_rgb}
        return paired_imgs, target

    def __getitem__(self, idx: int):
        for attempt in range(self.max_retry):
            try:
                paired_imgs, target = self._load_single_pair(idx)

                if self._transforms is not None:
                    res = self._transforms(paired_imgs, target, self)
                    # 确保提取出处理后的 6 通道张量与目标标签
                    paired_imgs, target = res[0], res[1]

                return paired_imgs, target

            except Exception as e:
                logger.error(
                    f"读取索引 {idx} (ID: {self.img_ids[idx]}) 时发生异常: {str(e)}. "
                    f"正在启动自愈保护机制 (尝试 {attempt + 1}/{self.max_retry})..."
                )
                idx = np.random.randint(0, len(self.img_ids))

        raise RuntimeError(
            f"数据集自愈重试 {self.max_retry} 次全部失败，请检查数据完整性！"
        )

    def get_ann_info(self, idx: int) -> dict:
        """评估器专用快速标签获取接口."""
        img_id: str = self.img_ids[idx]
        txt_path: str = os.path.join(self.ann_folder, f"{img_id}.txt")
        target = self._parse_txt(txt_path, 0, 0)
        target["image_id"] = torch.tensor([idx])
        target["idx"] = torch.tensor([idx])
        return target

    def _parse_txt(self, txt_path: str, w: int, h: int) -> dict:
        """标准 10 列 DOTA 文本标签解析."""
        boxes: List[List[float]] = []
        labels: List[int] = []

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue

                    category = parts[8]
                    if category not in self.category2label:
                        continue

                    poly = list(map(float, parts[:8]))
                    poly_np = np.array(poly, dtype=np.float32).reshape(4, 2)
                    rect = cv2.minAreaRect(poly_np)
                    (cx, cy), (w_box, h_box), angle = rect

                    if w_box <= 1 or h_box <= 1:
                        continue

                    if w_box < h_box:
                        w_box, h_box = h_box, w_box
                        angle += 90.0
                    angle = angle % 180.0
                    angle_norm = angle / 180.0
                    boxes.append([cx, cy, w_box, h_box, angle_norm])
                    labels.append(self.category2label[category])

        if len(boxes) > 0:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "orig_size": torch.tensor([h, w]),
                "size": torch.tensor([h, w]),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 5), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "orig_size": torch.tensor([h, w]),
                "size": torch.tensor([h, w]),
            }

        return target


    def load_item(self, idx: int) -> Tuple[Dict[str, Image.Image], dict]:
        """供 Mosaic 算子动态调用的成对样本读取接口."""
        return self._load_single_pair(idx)