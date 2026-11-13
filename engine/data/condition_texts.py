"""完整六条件的原文模板；只使用XML顶层图像属性，不使用目标标注。

高度/倾角不做归一化，时间保留原始日期及格式；类别编号查官方描述。
这个函数只是确定性文本拼装，不是可学习条件编码器。
"""

import datetime
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Mapping, Union


SIX_XML_FIELDS = ("UAVheight", "UAVangle", "time", "weather", "illumination", "location")
CONDITION_COUNTS = {"weather": 10, "illumination": 7, "location": 11}
SIX_CONDITION_TEMPLATE = (
    "altitude:{altitude} metre;camera pitch angle:{angle} degrees;"
    "current time:{time};weather:{weather};illumination:{illumination};location:{location}."
)


def load_condition_descriptions(path: Union[str, Path]) -> Dict[str, List[str]]:
    """读取并核验三类官方原始编号与文本，禁止替换为评估分组码。

    Args:
        path: schema_version=1的现有atrumod_condition_texts.json。

    Returns:
        天气10条、光照7条、场景11条；每个列表的索引即原始编号。

    Raises:
        ValueError: 码表版本、编号顺序或文本不合法。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("六条件模板需要schema_version=1的官方属性码表。")
    result: Dict[str, List[str]] = {}
    for field, count in CONDITION_COUNTS.items():
        rows = data.get(field)
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError(f"{field}必须有{count}条原始属性描述。")
        if any(
            not isinstance(row, dict) or type(row.get("id")) is not int
            or row["id"] != index or not isinstance(row.get("text"), str)
            or not row["text"].strip()
            for index, row in enumerate(rows)
        ):
            raise ValueError(f"{field}编号必须连续有序，原文不能为空。")
        result[field] = [row["text"] for row in rows]
    return result


def read_raw_six_conditions(root: ET.Element, source: str) -> Dict[str, str]:
    """读取XML顶层六字段，保留数字与时间的原文表示。

    Args:
        root: XML根节点；不遍历object或读取框/目标类别。
        source: XML路径，仅用于定位缺失字段。

    Returns:
        六字段字符串字典，首尾空白去除，内容不重写。

    Raises:
        ValueError: 任一顶层字段缺失或为空。
    """
    raw = {field: (root.findtext(field) or "").strip() for field in SIX_XML_FIELDS}
    missing = [field for field, value in raw.items() if not value]
    if missing:
        raise ValueError(f"{source}: 缺少六条件原文{missing}，不补默认值。")
    return raw


def build_six_condition_text(
    raw: Mapping[str, str], descriptions: Mapping[str, List[str]],
    source: str = "six-condition metadata",
) -> str:
    """验证原始条件并按用户固定模板生成一整段文本。

    Args:
        raw: UAVheight/UAVangle/time和三类别编号的原始字符串。
        descriptions: 已核验的官方三属性描述表。
        source: 错误来源；不进入模板。

    Returns:
        六条件完整文本；日期、数字格式、描述中的下划线与拼写均保留。

    Raises:
        ValueError: 缺失、数值越界、时间/日期非法或类别编号非法。
    """
    if any(not isinstance(raw.get(field), str) or not raw[field].strip()
           for field in SIX_XML_FIELDS):
        raise ValueError(f"{source}: 六条件必须完整提供原始字符串。")
    values = {field: raw[field].strip() for field in SIX_XML_FIELDS}
    for field, lower, upper in (("UAVheight", 80.0, 300.0), ("UAVangle", 0.0, 75.0)):
        try:
            value = float(values[field])
        except ValueError as error:
            raise ValueError(f"{source}: {field}不是合法数值。") from error
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{source}: {field}不在[{lower},{upper}]，不裁剪。")
    match = re.fullmatch(
        r"(?:(\d{4})-(\d{1,2})-(\d{1,2})T)?(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?",
        values["time"],
    )
    if match is None:
        raise ValueError(f"{source}: time原文不是合法HH:MM[:SS]或日期T时刻。")
    year, month, day, hour, minute, second = match.groups()
    try:
        if year is not None:
            datetime.date(int(year), int(month), int(day))
        datetime.time(int(hour), int(minute), int(second or 0))
    except ValueError as error:
        raise ValueError(f"{source}: time原文日期或时分秒越界。") from error
    selected: Dict[str, str] = {}
    for field, count in CONDITION_COUNTS.items():
        try:
            index = int(values[field])
        except ValueError as error:
            raise ValueError(f"{source}: {field}不是整数原始编号。") from error
        if not 0 <= index < count or len(descriptions.get(field, [])) != count:
            raise ValueError(f"{source}: {field}编号或原始描述表非法。")
        selected[field] = descriptions[field][index]
    return SIX_CONDITION_TEMPLATE.format(
        altitude=values["UAVheight"], angle=values["UAVangle"], time=values["time"], **selected,
    )
