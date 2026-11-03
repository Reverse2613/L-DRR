"""
data_pre/inspect_atrumod.py

ATR-UMOD 数据集元数据完整性体检脚本.
遍历全部 XML 标签文件，彻底查清:
1. XML 中实际存在的每一个类别名称 (精确到空格与下划线) 及其出现频次；
2. 6 维物理属性 (高度, 角度, 时间, 天气, 光照, 场景) 的真实取值范围与离散值集合；
3. 检测是否存在空目标 XML 或格式异常文件.
"""

import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Set, Dict, Any

def inspect_atrumod(xml_folder: str) -> None:
    r"""全面审查 ATR-UMOD XML 标注文件."""
    if not os.path.exists(xml_folder):
        print(f"[错误] 目录不存在: {xml_folder}")
        return

    xml_files = [f for f in os.listdir(xml_folder) if f.lower().endswith(".xml")]
    total_files = len(xml_files)
    print(f"\n{'='*70}")
    print(f">>> 开始审查 ATR-UMOD 标签目录: {xml_folder}")
    print(f">>> 检测到 XML 标注文件总数: {total_files}")
    print(f"{'='*70}\n")

    category_counter = Counter()
    #set集合，自动去重
    weather_set: Set[int] = set()
    illum_set: Set[int] = set()
    loc_set: Set[int] = set()

    height_list = []
    angle_list = []
    corrupted_files = []
    empty_object_files = 0

    for idx, filename in enumerate(xml_files):
        xml_path = os.path.join(xml_folder, filename)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 1. 统计属性值
            h_text = root.findtext("UAVheight")
            a_text = root.findtext("UAVangle")
            w_text = root.findtext("weather")
            i_text = root.findtext("illumination")
            l_text = root.findtext("location")

            if h_text is not None: height_list.append(float(h_text))
            if a_text is not None: angle_list.append(float(a_text))
            if w_text is not None: weather_set.add(int(w_text))
            if i_text is not None: illum_set.add(int(i_text))
            if l_text is not None: loc_set.add(int(l_text))

            # 2. 统计目标类别 (精确捕获原始字符串)
            objects = root.findall("object")
            if len(objects) == 0:
                empty_object_files += 1

            for obj in objects:
                raw_name = obj.findtext("name", "")
                # 记录原始大小写与字符形态
                category_counter[raw_name] += 1

        except Exception as e:
            corrupted_files.append((filename, str(e)))

        if (idx + 1) % 2000 == 0 or (idx + 1) == total_files:
            print(f"进度: [{idx+1}/{total_files}] 已处理...")

    print("\n" + "#"*70)
    print("                    【元数据探针解析报告】")
    print("#"*70)

    print("\n1. 真实目标类别清单与目标计数 (精确字符串):")
    print("-" * 50)
    #按（类别名，次数）降序排列
    for cat_name, count in sorted(category_counter.items(), key=lambda x: -x[1]):
        print(f"   - '{cat_name}': {count:>7d} 个")
    print(f"   [总计识别不同类别名称数]: {len(category_counter)}")

    print("\n2. 连续物理属性真实极值:")
    print("-" * 50)
    if height_list:
        print(f"   - UAVheight (高度): Min = {min(height_list):.1f}m, Max = {max(height_list):.1f}m")
    if angle_list:
        print(f"   - UAVangle  (俯仰): Min = {min(angle_list):.1f}°,  Max = {max(angle_list):.1f}°")

    print("\n3. 离散条件属性出现的所有实际整数集合:")
    print("-" * 50)
    print(f"   - weather      (天气标签): {sorted(list(weather_set))}")
    print(f"   - illumination (光照标签): {sorted(list(illum_set))}")
    print(f"   - location     (场景标签): {sorted(list(loc_set))}")

    print("\n4. 异常与边界样本统计:")
    print("-" * 50)
    print(f"   - 无目标样本 (全背景图): {empty_object_files} 个")
    print(f"   - 损坏解析失败文件    : {len(corrupted_files)} 个")
    if corrupted_files:
        for f, err in corrupted_files[:5]:
            print(f"     * {f}: {err}")
    print("#"*70 + "\n")

if __name__ == "__main__":
    # 传入本地解压出来的 train或者test 标签目录
    target_dir = "/home/reverse/workspace/data/ATR-UMOD/test/labels_ir"
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    inspect_atrumod(target_dir)