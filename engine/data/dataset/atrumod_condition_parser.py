"""
engine/data/dataset/atrumod_condition_parser.py

ATR-UMOD 数据集多环境物理条件解析器.
支持 6 大维度 (高度 AL, 俯仰角 AN, 时间 TI, 天气 WE, 照度 IL, 场景 SC) 的规则规约与样本分组.
"""

import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional, Any


class ATRUMODConditionParser:
    r"""ATR-UMOD 数据集物理环境条件标签规约与解析器."""

    # 1. 场景粗粒度合并映射 (11 类 -> 5 大类)
    SCENARIO_MAPPING = {
        '0': 'Road',               # Urban villages
        '1': 'Neighborhood',       # Urban villages
        '2': 'Construction site',  # Urban Suburb
        '3': 'Road',               # Urban Suburb
        '4': 'Neighborhood',       # School -> 归入生活/社区园区
        '5': 'Factory',            # Urban Suburb
        '6': 'Parking Lot',        # Urban Suburb
        '7': 'Neighborhood',       # Urban Suburb
        '8': 'Road',               # Urban Center
        '9': 'Construction site',  # Urban villages
        '10': 'Parking Lot'        # Urban villages
    }

    # 2. 光照合并映射 (7 类 -> 4 类)
    ILLUMINATION_MAPPING = {
        '0': 'Normal',      # Overexposure -> Normal
        '1': 'Normal',      # Normal
        '2': 'Dim',         # Pre-Twilight -> Dim
        '3': 'Dim',         # Post-Twilight -> Dim
        '4': 'Dim',         # Dim
        '5': 'Near Night',  # Near Night
        '6': 'Night'        # Night
    }

    # 3. 天气合并映射 (10 类 -> 5 类)
    WEATHER_MAPPING = {
        '0': 'Sunny',
        '1': 'Cloudy',
        '2': 'Cloudy',      # Rainy -> Cloudy
        '3': 'Cloudy',      # After Rain -> Cloudy
        '4': 'Foggy',
        '5': 'Snowy',
        '6': 'Night',       # Sunny_Night
        '7': 'Night',       # Cloudy_Night
        '8': 'Night',       # Sunny_Near_Night
        '9': 'Night'        # Cloudy_Near_Night
    }

    # 条件大类顺序与各子类顺序 (严格对应论文表格)
    CONDITIONS_SCHEMA = {
        'AL': ['[0, 120]', '(120, 300]'],
        'AN': ['[0, 30]', '(30, 75]'],
        'TI': ['Morning', 'Afternoon', 'Night'],
        'WE': ['Cloudy', 'Foggy', 'Snowy', 'Sunny', 'Night'],
        'IL': ['Normal', 'Dim', 'Near Night', 'Night'],
        'SC': ['Construction site', 'Factory', 'Neighborhood', 'Parking Lot', 'Road']
    }

    @staticmethod
    def get_altitude_group(alt: float) -> str:
        """高度分级: <= 120m 为低空, > 120m 为高空."""
        return '[0, 120]' if alt <= 120.0 else '(120, 300]'

    @staticmethod
    def get_angle_group(deg: float) -> str:
        """航拍俯仰角分级: <= 30度 为大俯视角(低倾角), > 30度 为倾斜斜视角."""
        return '[0, 30]' if deg <= 30.0 else '(30, 75]'

    @staticmethod
    def get_time_group(time_str: str) -> str:
        """解析时间戳字符串 (如 2022-10-13T18:38)."""
        try:
            # 兼容 ISO 格式 '2022-10-13T18:38'
            if 'T' in time_str:
                hour = int(time_str.split('T')[1].split(':')[0])
            else:
                # 尝试提取冒号前后的数字
                match = re.search(r'(\d{1,2}):\d{2}', time_str)
                if match:
                    hour = int(match.group(1))
                else:
                    return 'Afternoon'  # 默认回退
        except Exception:
            return 'Afternoon'

        if 7 <= hour < 14:
            return 'Morning'
        elif 14 <= hour < 19:
            return 'Afternoon'
        else:
            return 'Night'

    @classmethod
    def parse_xml_conditions(cls, xml_path: str) -> Dict[str, str]:
        """解析单张图像 XML 文件, 返回其所属的 6 个条件类别标签.

        Args:
            xml_path (str): XML 文件绝对路径.

        Returns:
            Dict[str, str]: 形如:
                {
                    'AL': '[0, 120]',
                    'AN': '(30, 75]',
                    'TI': 'Afternoon',
                    'WE': 'Night',
                    'IL': 'Night',
                    'SC': 'Road'
                }
        """
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"找不到指定的 XML 标注文件: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 1. 高度解析
        height_node = root.find('UAVheight')
        alt_val = float(height_node.text) if height_node is not None and height_node.text else 100.0
        al_group = cls.get_altitude_group(alt_val)

        # 2. 角度解析
        angle_node = root.find('UAVangle')
        ang_val = float(angle_node.text) if angle_node is not None and angle_node.text else 45.0
        an_group = cls.get_angle_group(ang_val)

        # 3. 采集时间解析
        time_node = root.find('time')
        time_str = time_node.text if time_node is not None and time_node.text else '12:00'
        ti_group = cls.get_time_group(time_str)

        # 4. 天气解析
        weather_node = root.find('weather')
        raw_we = str(weather_node.text).strip() if weather_node is not None and weather_node.text else '0'
        we_group = cls.WEATHER_MAPPING.get(raw_we, 'Sunny')

        # 5. 照度解析
        illum_node = root.find('illumination')
        raw_il = str(illum_node.text).strip() if illum_node is not None and illum_node.text else '1'
        il_group = cls.ILLUMINATION_MAPPING.get(raw_il, 'Normal')

        # 6. 场景解析
        loc_node = root.find('location')
        raw_sc = str(loc_node.text).strip() if loc_node is not None and loc_node.text else '0'
        sc_group = cls.SCENARIO_MAPPING.get(raw_sc, 'Road')

        return {
            'AL': al_group,
            'AN': an_group,
            'TI': ti_group,
            'WE': we_group,
            'IL': il_group,
            'SC': sc_group,
        }