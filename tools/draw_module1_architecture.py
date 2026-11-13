"""绘制当前模块一的中文架构图，输出 PNG、SVG 和 PDF。

图示依据 dt_cpn.py、cm_cme.py、cg_dmf.py、dual_hgnetv2_pg.py；
这里只绘图，不导入模型、不加载检测权重，也不启动训练或评估。
SVG 将文字转换为轮廓，PDF 嵌入字体，便于跨电脑展示。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

# 将绘图缓存写入可写的临时目录，避免改变用户的全局字体配置。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rio_detr_module1_mpl")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from fontTools.ttLib import TTFont


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "ink": "#182C47", "muted": "#5D6D81", "border": "#D7E0EB",
    "rgb": "#15846C", "rgb_bg": "#EAF6F1",
    "ir": "#2468AA", "ir_bg": "#EBF3FC",
    "condition": "#B9770E", "condition_bg": "#FFF4DB",
    "exchange": "#B64F61", "exchange_bg": "#FFF0F2",
    "fusion": "#7158A6", "fusion_bg": "#F3EFFB",
    "feature": "#566B84", "panel": "#F7F9FC", "white": "#FFFFFF",
}

Point = tuple[float, float]


def resolve_fonts(font_path: Path | None = None) -> tuple[FontProperties, FontProperties]:
    """选择中文字体，必要时从 TTC 提取首个字体到临时目录。

    Args:
        font_path: 可选的本地中文 TTF/TTC 文件。

    Returns:
        正常与粗体两种字体属性；图形导出不依赖查看者安装该字体。

    Raises:
        FileNotFoundError: 没有可用中文字体，不静默替换成缺字字体。
    """
    candidates = [font_path] if font_path is not None else [
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/mnt/c/Windows/Fonts/simhei.ttf"),
    ]
    chosen = next((p for p in candidates if p is not None and p.is_file()), None)
    if chosen is None:
        raise FileNotFoundError("缺少中文字体；请用 --font 指定本地 TTF/TTC。")

    def prepare(path: Path) -> Path:
        """把 TTC 首字体保存成 PDF 后端可直接嵌入的 TTF。"""
        if path.suffix.lower() != ".ttc":
            return path
        directory = Path("/tmp/rio_detr_module1_fonts")
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{path.stem}-face0.ttf"
        font = TTFont(str(path), fontNumber=0)
        font.save(destination)
        font.close()
        return destination

    normal = prepare(chosen)
    bold_path = chosen.with_name("msyhbd.ttc") if chosen.name == "msyh.ttc" else chosen
    bold = prepare(bold_path) if bold_path.is_file() else normal
    return FontProperties(fname=str(normal)), FontProperties(fname=str(bold))


class Diagram:
    """使用以左上角为原点的固定画布绘制说明图。"""

    def __init__(
        self, width: int, height: int,
        fonts: tuple[FontProperties, FontProperties],
    ) -> None:
        """建立画布，坐标单位为100 DPI下的像素。

        Args:
            width: 画布宽度。
            height: 画布高度。
            fonts: 正常与粗体中文字体。
        """
        self.width, self.height = width, height
        self.normal, self.bold = fonts
        self.figure = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        self.axis = self.figure.add_axes((0, 0, 1, 1))
        self.axis.set_xlim(0, width)
        self.axis.set_ylim(height, 0)
        self.axis.axis("off")
        self.figure.patch.set_facecolor(COLORS["white"])

    def text(
        self, x: float, y: float, value: str, size: float = 15,
        color: str = "ink", bold: bool = False, align: str = "center",
    ) -> None:
        """在指定位置写单行文本。

        Args:
            x: 水平位置。
            y: 垂直位置。
            value: 文本；数学符号也可使用 Matplotlib mathtext。
            size: 字号，单位pt。
            color: COLORS 的键或颜色字符串。
            bold: 是否使用粗体。
            align: 水平对齐方式。
        """
        self.axis.text(
            x, y, value, fontsize=size, color=COLORS.get(color, color),
            fontproperties=self.bold if bold else self.normal,
            ha=align, va="center", zorder=5,
        )

    def rect(
        self, x: float, y: float, width: float, height: float,
        fill: str = "white", border: str = "border", dashed: bool = False,
    ) -> None:
        """绘制圆角矩形，坐标指向外框左上角。

        Args:
            x: 左侧位置。
            y: 顶部位置。
            width: 外框宽度。
            height: 外框高度。
            fill: 填充颜色。
            border: 边框颜色。
            dashed: 是否使用虚线边框。
        """
        self.axis.add_patch(FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0,rounding_size=14",
            facecolor=COLORS.get(fill, fill), edgecolor=COLORS.get(border, border),
            linewidth=1.3, linestyle=(0, (5, 4)) if dashed else "solid", zorder=1,
        ))

    def box(
        self, x: float, y: float, width: float, height: float,
        lines: Sequence[str], fill: str = "white", border: str = "border",
        title_size: float = 18, body_size: float = 14,
    ) -> None:
        """绘制标题和说明组成的节点。

        Args:
            x: 左侧位置。
            y: 顶部位置。
            width: 节点宽度。
            height: 节点高度。
            lines: 首行为标题，后续各行为说明。
            fill: 填充颜色。
            border: 边框颜色。
            title_size: 标题字号。
            body_size: 说明字号。
        """
        self.rect(x, y, width, height, fill, border)
        gap = min(31, (height - 34) / max(1, len(lines) - 1))
        start = y + height / 2 - gap * (len(lines) - 1) / 2
        for index, line in enumerate(lines):
            self.text(x + width / 2, start + index * gap, line,
                      title_size if index == 0 else body_size,
                      "ink" if index == 0 else "muted", bold=index == 0)

    def path(
        self, points: Sequence[Point], color: str = "feature",
        dashed: bool = False, arrow: bool = True,
    ) -> None:
        """绘制折线，末段可带箭头。

        Args:
            points: 按流向排列的折线顶点。
            color: 线条颜色。
            dashed: 条件控制使用虚线，特征使用实线。
            arrow: 是否在最后一段绘制箭头。
        """
        linestyle = (0, (4, 3)) if dashed else "solid"
        stroke = COLORS.get(color, color)
        if not arrow or len(points) > 2:
            previous = points[:-1] if arrow else points
            self.axis.plot([p[0] for p in previous], [p[1] for p in previous],
                           color=stroke, linewidth=1.8, linestyle=linestyle, zorder=2)
        if arrow:
            self.axis.add_patch(FancyArrowPatch(
                points[-2], points[-1], arrowstyle="-|>", mutation_scale=14,
                linewidth=1.8, color=stroke, linestyle=linestyle,
                shrinkA=0, shrinkB=2, zorder=2,
            ))

    def dot(self, x: float, y: float, color: str) -> None:
        """在同一支路的分叉或汇合处标记连接点。"""
        self.axis.add_patch(Circle((x, y), 3.7, facecolor=COLORS[color],
                                   edgecolor="none", zorder=3))

    def panel(self, y: float, height: float, title: str, number: str, color: str) -> None:
        """绘制细节页的一条横向模块说明面板。"""
        self.rect(55, y, self.width - 110, height, "panel")
        self.axis.add_patch(Circle((92, y + 37), 19, facecolor=COLORS[color],
                                   edgecolor="none", zorder=3))
        self.text(92, y + 37, number, 17, "white", True)
        self.text(126, y + 37, title, 21, bold=True, align="left")

    def export(self, output_dir: Path, name: str, dpi: int = 180) -> None:
        """输出高清位图与两个可缩放格式。

        Args:
            output_dir: 输出目录。
            name: 文件名主干。
            dpi: PNG 导出分辨率；SVG、PDF不受此限制。
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "svg", "pdf"):
            target = output_dir / f"{name}.{suffix}"
            self.figure.savefig(target, dpi=dpi, facecolor="white")
            print(target.resolve())
        plt.close(self.figure)


def draw_overview(fonts: tuple[FontProperties, FontProperties]) -> Diagram:
    """绘制总览：条件共享、双主干、三尺度融合及原检测器接口。

    Args:
        fonts: 正常与粗体中文字体。

    Returns:
        可直接导出的模块一总览画布。
    """
    d = Diagram(2000, 980, fonts)
    d.text(65, 65, "模块一：先编码条件，再双向互补，再动态选择", 27, bold=True, align="left")
    d.text(65, 110, "DT-CPN + CM-CME + CG-DMF  ·  ATR-UMOD 双光旋转目标检测", 16,
           "muted", align="left")
    d.path([(1510, 74), (1570, 74)], arrow=False)
    d.text(1584, 74, "图像 / 特征", 14, align="left")
    d.path([(1740, 74), (1800, 74)], "condition", True, False)
    d.text(1813, 74, "条件控制", 14, align="left")

    d.box(70, 175, 265, 100, ["六类图像级条件", "高度 · 倾角 · 时刻", "天气 · 光照 · 场景"],
          "condition_bg", "condition", 19, 15)
    d.box(410, 175, 300, 100, ["① DT-CPN", "几何轨 + 环境轨", "编码图像级条件"],
          "condition_bg", "condition", 20, 15)
    d.box(790, 175, 220, 100, ["条件向量 c", "[B,128]"],
          "condition_bg", "condition", 20, 18)
    d.path([(335, 225), (410, 225)], "condition", True)
    d.path([(710, 225), (790, 225)], "condition", True)

    d.box(70, 395, 175, 100, ["RGB 图像", "[B,3,512,640]"], "rgb_bg", "rgb", 20, 12)
    d.box(70, 645, 175, 100, ["IR 图像", "[B,3,512,640]"], "ir_bg", "ir", 20, 12)
    d.box(305, 395, 250, 100, ["HGNetv2-B0", "RGB 独立预训练主干"], "rgb_bg", "rgb", 19, 14)
    d.box(305, 645, 250, 100, ["HGNetv2-B0", "IR 独立预训练主干"], "ir_bg", "ir", 19, 14)
    d.path([(245, 445), (305, 445)], "rgb")
    d.path([(245, 695), (305, 695)], "ir")
    d.text(70, 830, "双主干参数独立", 15, "muted", align="left")

    d.rect(780, 310, 850, 570, "panel", dashed=True)
    d.text(810, 331, "S3 / S4 / S5：各自处理", 16, bold=True, align="left")
    d.text(1530, 331, "参数独立", 15, "muted")
    d.path([(555, 445), (620, 445)], "rgb", arrow=False)
    d.path([(620, 432), (620, 792)], "rgb", arrow=False)
    d.path([(555, 695), (675, 695)], "ir", arrow=False)
    d.path([(675, 458), (675, 818)], "ir", arrow=False)
    d.path([(1010, 225), (1120, 225), (1120, 720)], "condition", True, False)

    for index, (cy, shape) in enumerate([(445, "256×64×80"), (625, "512×32×40"),
                                         (805, "1024×16×20")], 3):
        d.path([(620, cy - 13), (815, cy - 13)], "rgb")
        d.path([(675, cy + 13), (815, cy + 13)], "ir")
        d.dot(620, cy - 13, "rgb")
        d.dot(675, cy + 13, "ir")
        d.text(722, cy - 31, f"RGB S{index}", 12, "rgb")
        d.text(740, cy + 32, f"IR S{index}", 12, "ir")
        d.box(815, cy - 60, 270, 120, ["② CM-CME", "条件调制 + 双向消息", "输出仍保留 RGB / IR 两路"],
              "exchange_bg", "exchange", 20, 14)
        d.box(1180, cy - 60, 230, 120, ["③ CG-DMF", "局部动态权重", "加权融合 + 通道重整"],
              "fusion_bg", "fusion", 20, 14)
        d.box(1455, cy - 45, 155, 90, [f"Y{index}", shape], "white", "feature", 22, 14)
        d.path([(1085, cy), (1180, cy)])
        d.path([(1410, cy), (1455, cy)])
        # 相同 c 在每个尺度同时控制 CM-CME 与 CG-DMF。
        branch_y = cy - 85
        d.dot(1120, branch_y, "condition")
        d.path([(1120, branch_y), (950, branch_y), (950, cy - 60)], "condition", True)
        d.path([(1120, branch_y), (1295, branch_y), (1295, cy - 60)], "condition", True)
        d.path([(1610, cy), (1675, cy)], arrow=False)
        d.dot(1675, cy, "feature")

    d.path([(1675, 445), (1675, 805)], arrow=False)
    d.rect(1720, 345, 240, 535, "panel")
    d.text(1840, 375, "原检测框架", 17, bold=True)
    d.box(1750, 415, 195, 120, ["HybridEncoder", "三尺度聚合", "输出224通道"], title_size=16, body_size=14)
    d.path([(1675, 475), (1750, 475)])
    d.dot(1675, 475, "feature")
    d.box(1750, 625, 195, 110, ["OBB Decoder", "原3层旋转解码器", "300个检测 Query"], title_size=17, body_size=13)
    d.path([(1847, 535), (1847, 625)])
    d.box(1750, 790, 195, 75, ["检测输出", "11类 + 旋转框"], title_size=18, body_size=14)
    d.path([(1847, 735), (1847, 790)])
    d.text(70, 925, "图像 / 特征为实线，条件为虚线；同一 c 跨尺度共享。", 15,
           "muted", align="left")
    d.text(70, 956, "Y 的尺寸省略 batch B；输出通道和空间尺寸与原编码器接口一致。", 14,
           "muted", align="left")
    d.text(1945, 934, "当前尚未接入模块二 GC-Decoder", 14,
           "muted", align="right")
    return d


def draw_details(fonts: tuple[FontProperties, FontProperties]) -> Diagram:
    """绘制三模块内部计算，突出消息源、受体与两条残差主路径。

    Args:
        fonts: 正常与粗体中文字体。

    Returns:
        DT-CPN、单尺度CM-CME及CG-DMF的细节画布。
    """
    d = Diagram(2000, 1855, fonts)
    d.text(65, 65, "模块一内部细节：条件如何控制互补与选择", 27, bold=True, align="left")
    d.text(65, 110, "下图展开一个尺度；C、h、w 随 S3 / S4 / S5 改变。橙色虚线表示条件控制。", 16,
           "muted", align="left")

    # DT-CPN：两轨先分别编码，再按geo→env拼接。
    d.panel(165, 340, "DT-CPN：六类条件 → 一个128维条件向量", "1", "condition")
    d.box(95, 240, 245, 90, ["高度 H / 倾角 θ", "数据端已规范化"],
          "condition_bg", "condition", 18, 14)
    d.box(390, 235, 460, 100, ["几何轨", "H / θ 各16维 Fourier → 拼接32维", "Linear 32→64 + GELU"],
          "condition_bg", "condition", 19, 15)
    d.box(920, 245, 165, 80, ["e_geo", "[B,64]"],
          "condition_bg", "condition", 21, 18)
    d.path([(340, 285), (390, 285)], "condition")
    d.path([(850, 285), (920, 285)], "condition")

    d.box(95, 360, 245, 120, ["环境条件", "天气 / 光照 / 场景", "时刻 sin/cos"],
          "condition_bg", "condition", 18, 14)
    d.box(390, 355, 460, 130, ["环境轨", "三个64维冻结文本锚点 + 有界残差",
                             "时刻 Linear 2→64；四项相加", "LayerNorm → 64维"],
          "condition_bg", "condition", 19, 14)
    d.box(920, 380, 165, 80, ["e_env", "[B,64]"],
          "condition_bg", "condition", 21, 18)
    d.path([(340, 420), (390, 420)], "condition")
    d.path([(850, 420), (920, 420)], "condition")

    d.box(1170, 285, 310, 140, ["拼接 + 线性合成", "e_geo ⊕ e_env → [B,128]", "Linear 128→128"],
          "condition_bg", "condition", 18, 15)
    d.path([(1085, 285), (1120, 285), (1120, 315), (1170, 315)], "condition")
    d.path([(1085, 420), (1120, 420), (1120, 395), (1170, 395)], "condition")
    d.box(1550, 305, 280, 100, ["条件向量 c", "[B,128]"],
          "condition_bg", "condition", 21, 18)
    d.path([(1480, 355), (1550, 355)], "condition")
    d.text(1690, 441, "每张图只编码一次", 15, "muted")
    d.text(1690, 474, "供所有尺度的 CM / CG 使用", 14, "muted")

    # CM-CME：FiLM保留主路径；局部状态及差分均从旧S读取。
    d.panel(550, 650, "CM-CME：先调制主路径，再向受体注入另一模态的消息", "2", "exchange")
    d.text(1900, 587, "输入 / 输出：两路 [B,C,h,w]", 14, "muted", align="right")
    d.box(95, 690, 165, 100, ["F_rgb", "RGB 特征"], "rgb_bg", "rgb", 22, 16)
    d.box(95, 990, 165, 100, ["F_ir", "IR 特征"], "ir_bg", "ir", 22, 16)
    d.box(320, 685, 280, 110, ["RGB 工作投影", "1×1 Conv + GN", "得到未调制的 S_rgb"],
          "rgb_bg", "rgb", 18, 15)
    d.box(320, 985, 280, 110, ["IR 工作投影", "1×1 Conv + GN", "得到未调制的 S_ir"],
          "ir_bg", "ir", 18, 15)
    d.path([(260, 740), (320, 740)], "rgb")
    d.path([(260, 1040), (320, 1040)], "ir")

    d.box(790, 690, 320, 100, ["FiLM 条件仿射调制", "Z = (1+γ(c)) ⊙ S + β(c)", "空间位置共享 γ / β"],
          "rgb_bg", "rgb", 18, 14)
    d.box(790, 990, 320, 100, ["FiLM 条件仿射调制", "Z = (1+γ(c)) ⊙ S + β(c)", "空间位置共享 γ / β"],
          "ir_bg", "ir", 18, 14)
    d.path([(600, 740), (790, 740)], "rgb")
    d.path([(600, 1040), (790, 1040)], "ir")
    d.box(860, 630, 160, 45, ["c [B,128]"], "condition_bg", "condition", 14)
    d.box(860, 1120, 160, 45, ["c [B,128]"], "condition_bg", "condition", 14)
    d.path([(940, 675), (940, 690)], "condition", True)
    d.path([(940, 1120), (940, 1090)], "condition", True)

    d.box(700, 805, 640, 170, ["双向消息：独立读取同一份旧 S", "局部状态 L → 支持 R / 需求 N / 有序视觉项 h",
                             "G = R_d × N_r × sigmoid[p(c)+h_d→r]",
                             "E = GELU(DWConv3×3(S_d−S_r))", "注入量 = G ⊙ E"],
          "exchange_bg", "exchange", 18, 15)
    d.path([(600, 740), (650, 740), (650, 865), (700, 865)], "rgb")
    d.path([(600, 1040), (665, 1040), (665, 945), (700, 945)], "ir")
    d.dot(650, 740, "rgb")
    d.dot(665, 1040, "ir")
    d.text(624, 826, "S_rgb", 13, "rgb")
    d.text(625, 963, "S_ir", 13, "ir")
    d.box(1385, 892, 115, 55, ["c"], "condition_bg", "condition", 20)
    d.path([(1385, 920), (1340, 920)], "condition", True)

    for cy, color in [(740, "rgb"), (1040, "ir")]:
        d.axis.add_patch(Circle((1475, cy), 21, facecolor="white", edgecolor=COLORS[color],
                               linewidth=1.6, zorder=3))
        d.text(1475, cy - 1, "+", 24, color)
        d.path([(1110, cy), (1454, cy)], color)
        d.path([(1496, cy), (1540, cy)], color)
    d.text(1235, 715, "Z_rgb 主路径", 14, "rgb")
    d.text(1235, 1070, "Z_ir 主路径", 14, "ir")
    d.path([(1340, 880), (1430, 880), (1430, 800), (1475, 800), (1475, 761)], "exchange")
    d.path([(1340, 955), (1430, 955), (1430, 985), (1475, 985), (1475, 1019)], "exchange")
    d.text(1580, 848, "IR → RGB 注入", 15, "exchange")
    d.text(1580, 964, "RGB → IR 注入", 15, "exchange")
    d.box(1540, 690, 350, 100, ["RGB 更新特征 Z′_rgb", "Z_rgb + 注入(IR→RGB)", "[B,C,h,w]"],
          "rgb_bg", "rgb", 18, 15)
    d.box(1540, 990, 350, 100, ["IR 更新特征 Z′_ir", "Z_ir + 注入(RGB→IR)", "[B,C,h,w]"],
          "ir_bg", "ir", 18, 15)
    d.text(95, 1176, "d 表示供体、r 表示受体；两方向同时更新，消息加到受体的 FiLM 主路径。", 14,
           "muted", align="left")

    # CG-DMF：有序Query评价互补后支路；权重只用于原两路特征的凸组合。
    d.panel(1240, 555, "CG-DMF：根据互补后的局部证据与条件，分配两路贡献", "3", "fusion")
    d.box(95, 1410, 260, 145, ["互补后的两路特征", "Z′_rgb / Z′_ir", "各 [B,C,h,w]"],
          "white", "feature", 16, 15)
    d.box(440, 1325, 350, 105, ["有序 RGB Query", "1×1 Conv([RGB,IR])", "Q_rgb：[B,32,h,w]"],
          "rgb_bg", "rgb", 18, 15)
    d.box(440, 1515, 350, 105, ["有序 IR Query", "1×1 Conv([IR,RGB])", "Q_ir：[B,32,h,w]"],
          "ir_bg", "ir", 18, 15)
    d.path([(355, 1450), (395, 1450), (395, 1377), (440, 1377)])
    d.path([(355, 1515), (395, 1515), (395, 1567), (440, 1567)])
    d.box(920, 1325, 420, 105, ["RGB 局部分数 s_rgb", "视觉项 + 条件底分", "+ Q_rgb · K_rgb(c) / √32"],
          "fusion_bg", "fusion", 18, 15)
    d.box(920, 1515, 420, 105, ["IR 局部分数 s_ir", "视觉项 + 条件底分", "+ Q_ir · K_ir(c) / √32"],
          "fusion_bg", "fusion", 18, 15)
    d.path([(790, 1377), (920, 1377)], "rgb")
    d.path([(790, 1567), (920, 1567)], "ir")
    d.box(990, 1440, 280, 60, ["同一条件 c", "生成两路 K 与条件底分"],
          "condition_bg", "condition", 15, 13)
    d.path([(1130, 1440), (1130, 1430)], "condition", True)
    d.path([(1130, 1500), (1130, 1515)], "condition", True)

    d.box(1420, 1405, 190, 170, ["两模态 Softmax", "M_rgb / M_ir", "逐位置权重", "[B,1,h,w]"],
          "fusion_bg", "fusion", 15, 14)
    d.path([(1340, 1377), (1385, 1377), (1385, 1450), (1420, 1450)], "fusion")
    d.path([(1340, 1567), (1385, 1567), (1385, 1530), (1420, 1530)], "fusion")
    d.text(1515, 1605, "M_rgb + M_ir = 1", 13, "fusion")
    d.box(1675, 1425, 240, 125, ["按位置加权", "M_rgb · Z′_rgb", "+ M_ir · Z′_ir"],
          "fusion_bg", "fusion", 18, 15)
    d.path([(1610, 1490), (1675, 1490)], "fusion")
    d.box(1675, 1595, 240, 75, ["1×1 Conv + GN", "通道重整"],
          "fusion_bg", "fusion", 17, 14)
    d.path([(1795, 1550), (1795, 1595)], "fusion")
    d.box(1675, 1710, 240, 65, ["Y：[B,C,h,w]"], "white", "feature", 18)
    d.path([(1795, 1670), (1795, 1710)])
    # 这是特征主路径：两个Query只负责打分，不能替代待融合特征。
    d.path([(225, 1555), (225, 1660), (1645, 1660), (1645, 1520), (1675, 1520)])
    d.text(920, 1640, "两路更新特征直接参与加权", 15, "muted")
    d.text(95, 1760, "空间权重在通道维广播；先融合，再做通道重整。", 14,
           "muted", align="left")
    d.text(65, 1830, "c 在图中重复标注，均为同一个条件向量；工作投影与双向消息均采用独立参数。", 14,
           "muted", align="left")
    return d


def main() -> None:
    """解析绘图参数并输出两张说明图，不执行任何模型运算。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plan")
    parser.add_argument("--font", type=Path, help="本地中文TTF/TTC；默认自动寻找微软雅黑或Noto CJK")
    parser.add_argument("--dpi", type=int, default=180, help="PNG分辨率；默认180")
    args = parser.parse_args()
    if not 80 <= args.dpi <= 600:
        parser.error("--dpi 必须在80到600之间。")
    matplotlib.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 42,
                                "ps.fonttype": 42, "axes.unicode_minus": False})
    fonts = resolve_fonts(args.font)
    draw_overview(fonts).export(args.output_dir, "module1_architecture", args.dpi)
    draw_details(fonts).export(args.output_dir, "module1_details", args.dpi)


if __name__ == "__main__":
    main()
