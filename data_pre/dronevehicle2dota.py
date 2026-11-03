"""DroneVehicle IR XML 标注转 DOTA 标准 OBB 格式转换器.

该脚本负责解析 DroneVehicle 数据集中的红外标注 (trainlabelr, vallabelr, testlabelr),
并将多边形旋转框 (polygon) 与退化水平框 (bndbox) 统一转换为遥感目标检测标准的 DOTA 文本格式。

输出格式:
    每行文本格式:
    x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
    - x1..y4: 旋转框 4 个顶点的【绝对像素坐标】 (浮点数保留 2 位小数)
    - class_name: 归一化后的标准类别字符串 (car, truck, bus, van, freight car)
    - difficulty: 难度系数, 统一填 0
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ==============================================================================
# 类别归一化映射字典 (吸收官方笔误变体)
# ==============================================================================
CLASS_MAPPING: Dict[str, str] = {
    # 1. 轿车
    "car": "car",
    # 2. 卡车
    "truck": "truck",
    # 3. 公交/大巴
    "bus": "bus",
    # 4. 面包车/厢式客车
    "van": "van",
    # 5. 货车/货运车 (统一彻底映射为无空格的 freight_car)
    "freight car": "freight_car",
    "freight_car": "freight_car",
    "freight": "freight_car",
    "feright car": "freight_car",
    "feright_car": "freight_car",
    "feright": "freight_car",
}

# DroneVehicle 各数据划分下的红外标签子目录名与图像子目录名
SPLIT_CONFIG: Dict[str, Dict[str, str]] = {
    "train": {"label_dir": "trainlabelr", "img_dir": "trainimgr"},
    "val": {"label_dir": "vallabelr", "img_dir": "valimgr"},
    "test": {"label_dir": "testlabelr", "img_dir": "testimgr"},
}


def normalize_class_name(raw_name: str) -> Optional[str]:
    """对 XML 中的原始类别名进行清洗，并映射到符合 DOTA 规范的无空格标准名称.

    Args:
        raw_name (str): 从 XML <name> 标签读取的原始字符串.

    Returns:
        Optional[str]: 映射后的标准类别名 (car, truck, bus, van, freight_car).
                       若未识别则返回 None.
    """
    cleaned: str = raw_name.strip().lower()
    # 统一将连续空格与下划线规范化
    cleaned = cleaned.replace("_", " ")
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")

    norm_name: Optional[str] = CLASS_MAPPING.get(cleaned, None)

    # 强迫症安全防御：确保生成的 DOTA 标签名绝无空格
    if norm_name is not None:
        assert (
            " " not in norm_name
        ), f"致命逻辑错误：DOTA 类别名不允许包含空格，当前为 '{norm_name}'"

    return norm_name


def parse_polygon_elements(polygon_elem: ET.Element) -> Optional[np.ndarray]:
    """从 XML <polygon> 节点提取多边形顶点坐标序列.

    Args:
        polygon_elem (ET.Element): XML 的 polygon 元素节点.

    Returns:
        Optional[np.ndarray]: 形状为 [P, 2] 的 float 数组, P 为顶点数。
                             若解析失败或点数不足返回 None.
    """
    xs: List[float] = []
    ys: List[float] = []
    for child in polygon_elem:
        tag: str = child.tag.lower()
        if tag.startswith("x") and child.text is not None:
            xs.append(float(child.text))
        elif tag.startswith("y") and child.text is not None:
            ys.append(float(child.text))

    if len(xs) == 0 or len(xs) != len(ys):
        return None
    return np.stack([np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)], axis=1)


def clip_and_validate_box(
    corners: np.ndarray, 
    img_w: float, 
    img_h: float
) -> Optional[np.ndarray]:
    """将旋转框顶点约束在图像分辨率边界内, 并剔除长宽小于 1 像素的退化框.

    Args:
        corners (np.ndarray): 形状为 [4, 2] 的顶点像素坐标数组.
        img_w (float): 图像的宽度 (如 840.0).
        img_h (float): 图像的高度 (如 712.0).

    Returns:
        Optional[np.ndarray]: 裁剪后的 [4, 2] 顶点坐标; 若退化则返回 None.
    """
    clipped: np.ndarray = corners.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, img_w)
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, img_h)

    # 最小外接边界检查
    x_min, x_max = clipped[:, 0].min(), clipped[:, 0].max()
    y_min, y_max = clipped[:, 1].min(), clipped[:, 1].max()

    if (x_max - x_min) < 1.0 or (y_max - y_min) < 1.0:
        return None
    return clipped


def convert_single_split(
    data_root: Path, 
    split: str, 
    output_dirname: str = "dota_labels_ir"
) -> Dict[str, Counter]:
    """转换指定划分 (train/val/test) 的所有红外 XML 为 DOTA 格式文本.

    Args:
        data_root (Path): DroneVehicle 数据集的根路径.
        split (str): 数据划分 ('train', 'val', 或 'test').
        output_dirname (str): 转换后生成的输出目录名, 默认为 'dota_labels_ir'.

    Returns:
        Dict[str, Counter]: 包含各类目标统计计数与跳过原因统计的字典.
    """
    cfg = SPLIT_CONFIG[split]
    label_dir: Path = data_root / split / cfg["label_dir"]
    out_dir: Path = data_root / split / output_dirname

    if not label_dir.exists():
        raise FileNotFoundError(f"未找到标签目录: {label_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    class_stats: Counter = Counter()
    skip_stats: Counter = Counter()
    total_xml_count: int = 0
    total_txt_count: int = 0

    xml_files = sorted(label_dir.glob("*.xml"))
    for xml_path in xml_files:
        total_xml_count += 1
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            skip_stats["xml_parse_error"] += 1
            continue

        # 读取图像物理尺寸
        try:
            img_w = float(root.findtext("size/width", "840"))
            img_h = float(root.findtext("size/height", "712"))
        except (ValueError, TypeError):
            skip_stats["invalid_size"] += 1
            continue

        dota_lines: List[str] = []

        for obj in root.iter("object"):
            raw_name: str = (obj.findtext("name") or "").strip()
            norm_name: Optional[str] = normalize_class_name(raw_name)

            if norm_name is None:
                skip_stats[f"unknown_class:{raw_name}"] += 1
                continue

            poly_elem = obj.find("polygon")
            bnd_elem = obj.find("bndbox")

            corners: Optional[np.ndarray] = None

            if poly_elem is not None:
                pts = parse_polygon_elements(poly_elem)
                if pts is None:
                    skip_stats["invalid_polygon"] += 1
                    continue
                if pts.shape[0] == 4:
                    corners = pts
                else:
                    # 顶点数超过 4 的极端情况, 计算最小外接矩形取 4 顶点
                    rect = cv2.minAreaRect(pts.astype(np.float32))
                    corners = cv2.boxPoints(rect)
            elif bnd_elem is not None:
                # 处理退化水平框
                try:
                    xmin = float(bnd_elem.findtext("xmin", "0"))
                    ymin = float(bnd_elem.findtext("ymin", "0"))
                    xmax = float(bnd_elem.findtext("xmax", "0"))
                    ymax = float(bnd_elem.findtext("ymax", "0"))
                    corners = np.array(
                        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                        dtype=np.float32,
                    )
                except ValueError:
                    skip_stats["invalid_bndbox"] += 1
                    continue
            else:
                skip_stats["no_box_annotation"] += 1
                continue

            # 裁剪并校验有效性
            valid_corners = clip_and_validate_box(corners, img_w, img_h)
            if valid_corners is None:
                skip_stats["degenerate_box"] += 1
                continue

            # 构造 DOTA 标准行: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
            flat_pts = valid_corners.reshape(-1)
            pts_str = " ".join(f"{coord:.2f}" for coord in flat_pts)
            # DOTA 格式: 绝对坐标 + 类别名 + 难度(0)
            line = f"{pts_str} {norm_name} 0"
            dota_lines.append(line)
            class_stats[norm_name] += 1

        # 保存为 .txt 文件 (与 xml 同名)
        target_txt = out_dir / f"{xml_path.stem}.txt"
        target_txt.write_text("\n".join(dota_lines) + "\n" if dota_lines else "", encoding="utf-8")
        total_txt_count += 1

    print(f"\n[{split.upper()}] 处理完毕: {total_xml_count} 个 XML -> {total_txt_count} 个 DOTA txt")
    print(f"各类别有效标注数量: {dict(class_stats)}")
    if skip_stats:
        print(f"过滤跳过情况: {dict(skip_stats)}")

    return {"classes": class_stats, "skipped": skip_stats}


def main() -> None:
    """命令行主执行函数."""
    parser = argparse.ArgumentParser(description="DroneVehicle IR XML 标签转 DOTA OBB 格式工具")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/reverse/workspace/data/DroneVehicle",
        help="DroneVehicle 数据集根目录绝对路径",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="待转换的数据划分, 以逗号分隔 (建议先转 train,val)",
    )
    parser.add_argument(
        "--output-dirname",
        type=str,
        default="dota_labels_ir",
        help="生成的 DOTA 标签保存目录名",
    )
    args = parser.parse_args()

    root_path = Path(args.data_root)
    if not root_path.exists():
        print(f"[错误] 数据集根路径不存在: {root_path}", file=sys.stderr)
        sys.exit(1)

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        convert_single_split(root_path, split, output_dirname=args.output_dirname)


if __name__ == "__main__":
    main()