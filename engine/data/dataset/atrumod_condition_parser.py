"""
engine/data/dataset/atrumod_condition_parser.py

ATR-UMOD 数据集多环境物理条件解析器.
支持 6 大维度 (高度 AL, 俯仰角 AN, 时间 TI, 天气 WE, 照度 IL, 场景 SC) 的规则规约与样本分组.
"""

import math
import re
import xml.etree.ElementTree as ET
from typing import Dict


class ATRUMODConditionParser:
    r"""ATR-UMOD 数据集物理环境条件标签规约与解析器."""

    UNKNOWN = 'Unknown'

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

    @classmethod
    def unknown_conditions(cls) -> Dict[str, str]:
        """无法读取条件 XML 时，六个维度均标为 Unknown。"""
        return {dim_key: cls.UNKNOWN for dim_key in cls.CONDITIONS_SCHEMA}

    @classmethod
    def get_altitude_group(cls, alt: float) -> str:
        """高度分级: <= 120m 为低空, > 120m 为高空."""
        try:
            alt = float(alt)
        except (TypeError, ValueError):
            return cls.UNKNOWN
        if not math.isfinite(alt) or not 0.0 <= alt <= 300.0:
            return cls.UNKNOWN
        return '[0, 120]' if alt <= 120.0 else '(120, 300]'

    @classmethod
    def get_angle_group(cls, deg: float) -> str:
        """航拍俯仰角分级: <= 30度 为大俯视角(低倾角), > 30度 为倾斜斜视角."""
        try:
            deg = float(deg)
        except (TypeError, ValueError):
            return cls.UNKNOWN
        if not math.isfinite(deg) or not 0.0 <= deg <= 75.0:
            return cls.UNKNOWN
        return '[0, 30]' if deg <= 30.0 else '(30, 75]'

    @classmethod
    def get_time_group(cls, time_str: str) -> str:
        """解析时间戳字符串 (如 2022-10-13T18:38)."""
        if not isinstance(time_str, str):
            return cls.UNKNOWN
        match = re.search(
            r'(?:T|^|\s)(\d{1,2}):(\d{2})(?::(\d{2}))?'
            r'(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$',
            time_str.strip(),
        )
        if match is None:
            return cls.UNKNOWN
        hour, minute = int(match.group(1)), int(match.group(2))
        second = int(match.group(3) or 0)
        if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
            return cls.UNKNOWN

        if 7 <= hour < 14:
            return 'Morning'
        elif 14 <= hour < 19:
            return 'Afternoon'
        else:
            return 'Night'

    @classmethod
    def parse_xml_conditions(cls, xml_path: str) -> Dict[str, str]:
        """解析单张图像 XML 文件, 返回其所属的 6 个条件类别标签.

        缺失或非法字段仅将对应维度标为 Unknown；XML 缺失或无法解析时六维均为 Unknown。

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
        try:
            tree = ET.parse(xml_path)
        except (OSError, ET.ParseError):
            return cls.unknown_conditions()
        root = tree.getroot()

        # 1. 高度解析
        al_group = cls.get_altitude_group(root.findtext('UAVheight'))

        # 2. 角度解析
        an_group = cls.get_angle_group(root.findtext('UAVangle'))

        # 3. 采集时间解析
        ti_group = cls.get_time_group(root.findtext('time'))

        # 4. 天气解析
        raw_we = (root.findtext('weather') or '').strip()
        we_group = cls.WEATHER_MAPPING.get(raw_we, cls.UNKNOWN)

        # 5. 照度解析
        raw_il = (root.findtext('illumination') or '').strip()
        il_group = cls.ILLUMINATION_MAPPING.get(raw_il, cls.UNKNOWN)

        # 6. 场景解析
        raw_sc = (root.findtext('location') or '').strip()
        sc_group = cls.SCENARIO_MAPPING.get(raw_sc, cls.UNKNOWN)

        return {
            'AL': al_group,
            'AN': an_group,
            'TI': ti_group,
            'WE': we_group,
            'IL': il_group,
            'SC': sc_group,
        }
