"""
tests/test_condition_parser.py
验证 XML 标注文件解析是否符合论文映射规则.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.data.dataset.atrumod_condition_parser import ATRUMODConditionParser

def test_xml_parsing():
    parser = ATRUMODConditionParser()
    
    # 假设测试文件为 tests/00355.xml (你可以把给我的内容存成这个文件)
    sample_xml = "/home/reverse/workspace/data/ATR-UMOD/test/labels_ir/00355.xml"
    if not os.path.exists(sample_xml):
        print(f"请先将 00355.xml 存放到 {sample_xml} 进行测试！")
        return

    res = parser.parse_xml_conditions(sample_xml)
    print("=" * 60)
    print("🚀 [00355.xml 解析结果验证]:")
    print("=" * 60)
    for k, v in res.items():
        print(f"  维度 [{k}]: {v}")
    
    # 理论预期:
    # UAVheight: 120 -> AL: '[0, 120]'
    # UAVangle: 45   -> AN: '(30, 75]'
    # time: 18:38    -> TI: 'Afternoon'
    # weather: 6     -> WE: 'Night'
    # illumination: 6-> IL: 'Night'
    # location: 3    -> SC: 'Road'
    assert res['AL'] == '[0, 120]', "高度映射错误！"
    assert res['AN'] == '(30, 75]', "角度映射错误！"
    assert res['TI'] == 'Afternoon', "时间映射错误！"
    assert res['WE'] == 'Night', "天气映射错误！"
    assert res['IL'] == 'Night', "照度映射错误！"
    assert res['SC'] == 'Road', "场景映射错误！"
    print("\n✅ 所有 6 大物理维度映射完全符合论文标准定义！")

if __name__ == '__main__':
    test_xml_parsing()