"""只看 RMS：修改下面的 path，然后运行 python tools/analyze_backbone_features.py。

输出 rms_models.png（四模型对照）、rms_fusion.png（双流分支与最终融合对照）。
每张热图独立归一化，用于看空间分布；颜色亮度不能跨图判断模型优劣。

修改记录（2026-11-08）：精简本文件；删除此前新增的五个辅助文件：
backbone_feature_plots.py、backbone_features.example.yml、test_backbone_features.py、
BACKBONE_FEATURES.md、BACKBONE_FEATURES_CHANGELOG.md。未修改训练或模型源码。
"""
from pathlib import Path
import copy
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ===== 只需改这里的路径（相对路径以运行时所在目录为基准） =====
IR_IMAGE = "/home/reverse/workspace/data/ATR-UMOD/test/images_ir/00187.jpg"
RGB_IMAGE = "/home/reverse/workspace/data/ATR-UMOD/test/images/00187.jpg"
OUTPUT_DIR = "./tools"
MODELS = [  # 名称、输入模态、实际训练配置、对应已训练权重
    ("IR only", "ir", "./configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_single_ir.yml", "./outputs/rtdetrv2_obb_hgnetv2_s_atrumod_single_ir/best_stg1.pth"),
    ("RGB only", "rgb", "./configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_single_rgb.yml", "./outputs/rtdetrv2_obb_hgnetv2_s_atrumod_single_rgb/best_stg1.pth"),
    ("Add", "paired", "./configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_add_baseline.yml", "./outputs/rtdetrv2_obb_hgnetv2_s_atrumod_add_baseline/best_stg1.pth"),
    ("Concat", "paired", "./configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_concat_baseline.yml", "./outputs/rtdetrv2_obb_hgnetv2_s_atrumod_concat_baseline/best_stg1.pth"),
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OVERLAY_ALPHA = 0.6  # 叠加强度，低响应区域保留原图。


def merge(base, extra):
    result = copy.deepcopy(base)
    for key, value in extra.items():
        result[key] = (merge(result[key], value)
                       if isinstance(value, dict) and isinstance(result.get(key), dict) else value)
    return result


def read_config(path, stack=()):
    path = Path(path).expanduser().resolve()
    if path in stack:
        raise ValueError(f"循环 include: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    config = {}
    for included in data.pop("__include__", []):
        config = merge(config, read_config(path.parent / included, (*stack, path)))
    return merge(config, data)


def load_backbone(config, checkpoint_path):
    from engine.core import create, merge_config

    config = copy.deepcopy(config)
    model = config["model"]
    detector = config[model] if isinstance(model, str) else merge(config.get(model["type"], {}), model)
    reference = detector["backbone"]
    name = reference if isinstance(reference, str) else reference["type"]
    if isinstance(reference, dict):
        config[name] = merge(config.get(name, {}), {k: v for k, v in reference.items() if k != "type"})
    config.setdefault(name, {})["pretrained"] = False
    if "type" in config[name]:
        config.setdefault(config[name]["type"], {})["pretrained"] = False
    backbone = create(name, merge_config(config, inplace=False))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("ema") is not None:
        state = checkpoint["ema"]["module"]
        source = "EMA"
    else:
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        source = "model/raw"
    weights = {}
    for key, value in state.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        if key in weights:
            raise ValueError(f"重复权重键: {key}")
        weights[key] = value
    if set(weights) != set(backbone.state_dict()) and any(k.startswith("backbone.") for k in weights):
        weights = {k[len("backbone."):]: v for k, v in weights.items() if k.startswith("backbone.")}
    backbone.load_state_dict(weights, strict=True)
    print(f"{name}: {source}")
    return backbone.to(DEVICE).eval()


def pixel_settings(config):
    ops = config["val_dataloader"]["dataset"]["transforms"]["ops"]
    ignored = {"PairedSanitizeBoundingBoxesOBB", "SanitizeBoundingBoxesOBB",
               "PairedConvertOBB", "ConvertOBB", "ConvertBox", "SanitizeBoundingBoxes"}
    resize, convert, normalize = None, None, None
    for op in ops:
        kind = op["type"]
        if kind in ("PairedResizeOBB", "ResizeOBB", "Resize"):
            if resize is not None or convert is not None:
                raise ValueError("验证预处理需要一次 resize，且在转 Tensor 前")
            resize = {k: v for k, v in op.items() if k != "type"}
        elif kind in ("PairedConvertPILToTensor", "ConvertPILImage"):
            if resize is None or convert is not None or op.get("dtype", "float32") != "float32":
                raise ValueError("验证预处理需要 resize 后转 float32 Tensor")
            convert = {"scale": op.get("scale", True)}
        elif kind == "Normalize":
            if convert is None or normalize is not None:
                raise ValueError("Normalize 应在转 Tensor 后")
            normalize = (op["mean"], op["std"])
        elif kind not in ignored:
            raise ValueError(f"暂不支持验证预处理 {kind}，请保持实际训练预处理")
    if resize is None or convert is None:
        raise ValueError("配置中缺少验证 resize/转 Tensor")
    return (merge({"keep_ratio": False, "pad_to_size": False, "fill": 0}, resize), convert, normalize)


def prepare_images(settings):
    from engine.data.transforms.transforms_rgbt import PairedResizeOBB

    with Image.open(IR_IMAGE) as image:
        ir = image.convert("RGB")
    with Image.open(RGB_IMAGE) as image:
        rgb = image.convert("RGB")
    if ir.size != rgb.size:
        raise ValueError("IR/RGB 必须是已经配准、尺寸相同的一对图像")
    originals = {"IR": np.array(ir), "RGB": np.array(rgb)}
    resize, convert, normalize = settings
    mask = Image.new("RGB", ir.size, "white")
    pair, _ = PairedResizeOBB(**resize)({"ir": ir, "rgb": rgb}, None)
    valid_pair, _ = PairedResizeOBB(**{**resize, "fill": 0})({"ir": mask, "rgb": mask}, None)
    valid = np.asarray(valid_pair["ir"])[:, :, 0] > 0
    tensors = [torch.from_numpy(np.array(pair[key]).transpose(2, 0, 1).copy()).float() for key in ("ir", "rgb")]
    if convert["scale"]:
        tensors = [value / 255 for value in tensors]
    if normalize is not None:
        mean, std = normalize
        if len(mean) not in (3, 6) or len(std) != len(mean) or any(x <= 0 for x in std):
            raise ValueError("mean/std 应为三或六通道参数，std 必须为正")
        for index in range(2):
            start = 3 * index if len(mean) == 6 else 0
            tensors[index] = ((tensors[index] - torch.tensor(mean[start:start + 3])[:, None, None])
                              / torch.tensor(std[start:start + 3])[:, None, None])
    ir, rgb = [value.unsqueeze(0).to(DEVICE) for value in tensors]
    return {"ir": ir, "rgb": rgb, "paired": torch.cat([ir, rgb], dim=1)}, valid, originals


def rms(feature):
    heatmap = feature.detach().float().square().mean(dim=1).sqrt()[0].cpu().numpy()
    if not np.isfinite(heatmap).all():
        raise ValueError("特征存在 NaN/Inf")
    return heatmap


def capture(backbone, tensor, modality):
    maps, handles = {}, []
    branches = ({"IR": backbone.backbone_ir, "RGB": backbone.backbone_rgb}
                if modality == "paired" else
                {modality.upper(): next(module for module in backbone.modules() if hasattr(module, "stages"))})
    try:
        for branch, network in branches.items():
            if len(network.stages) != 4 or list(network.return_idx) != [1, 2, 3]:
                raise ValueError("本脚本针对 HGNetv2 四个阶段及 S3/S4/S5 输出")
            for index, stage in enumerate(network.stages):
                def hook(_module, _inputs, output, key=(branch, f"S{index + 2}")):
                    maps[key] = rms(output)  # 直接聚合，不保留多通道张量。
                handles.append(stage.register_forward_hook(hook))
        with torch.inference_mode():
            output = backbone(tensor)
        if modality == "paired":
            if not isinstance(output, (list, tuple)) or len(output) != 3:
                raise ValueError("双流 backbone 应返回三个最终融合特征")
            maps.update({("Fused", f"S{index + 3}"): rms(value) for index, value in enumerate(output)})
        return maps
    finally:
        for handle in handles:
            handle.remove()


def normalize_rms(heatmap, valid):
    covered = np.array(Image.fromarray(valid.astype(np.float32)).resize(
        (heatmap.shape[1], heatmap.shape[0]), Image.Resampling.BOX)) > 0
    low, high = np.percentile(heatmap[covered], [1, 99])
    if high <= low:
        low, high = heatmap[covered].min(), heatmap[covered].max()
    scaled = np.clip((heatmap - low) / (high - low), 0, 1) if high > low else np.zeros_like(heatmap)
    return scaled, covered


def original_rms(heatmap, valid, image_shape):
    scaled, covered = normalize_rms(heatmap, valid)
    canvas_size = (valid.shape[1], valid.shape[0])
    def enlarge(value):
        return np.array(Image.fromarray(value.astype(np.float32)).resize(canvas_size, Image.Resampling.BILINEAR))
    weights = enlarge(covered)
    # 有效区域加权插值，避免纯 padding 单元的颜色渗入图像边缘。
    canvas = np.divide(enlarge(scaled * covered), weights, out=np.zeros_like(weights), where=weights > 1e-6)
    ys, xs = np.nonzero(valid)
    content = canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.clip(np.array(Image.fromarray(content).resize(
        (image_shape[1], image_shape[0]), Image.Resampling.BILINEAR)), 0, 1)


def save_plot(rows, valid, filename):
    fig, axes = plt.subplots(len(rows), 4, figsize=(12, 2.3 * len(rows)), squeeze=False, constrained_layout=True)
    for row, (name, maps) in enumerate(rows):
        for column, stage in enumerate(("S2", "S3", "S4", "S5")):
            ax = axes[row, column]
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(stage)
            if column == 0:
                ax.set_ylabel(name)
            if stage not in maps:
                ax.text(.5, .5, "No fusion at S2", ha="center", va="center", transform=ax.transAxes)
                continue
            heatmap = maps[stage]
            scaled, covered = normalize_rms(heatmap, valid)
            plotted = ax.imshow(np.ma.array(scaled, mask=~covered), cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    fig.suptitle("RMS | Each map normalized independently: compare spatial patterns, not brightness")
    fig.colorbar(plotted, ax=axes.ravel().tolist(), shrink=.7, label="Normalized RMS")
    try:
        fig.savefig(filename, dpi=160)
    finally:
        plt.close(fig)


def save_overlays(rows, valid, filename):
    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 3 * len(rows)), squeeze=False, constrained_layout=True)
    for row, (name, image, maps) in enumerate(rows):
        axes[row, 0].imshow(image)
        axes[row, 0].set_ylabel(name)
        for column, stage in enumerate(("S3", "S4", "S5"), start=1):
            response = original_rms(maps[stage], valid, image.shape)
            colors = plt.get_cmap("inferno")(response)[..., :3]
            alpha = OVERLAY_ALPHA * response[..., None]
            axes[row, column].imshow(np.clip(image / 255.0 * (1 - alpha) + colors * alpha, 0, 1))
        for column, title in enumerate(("Original", "S3 fused", "S4 fused", "S5 fused")):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(title)
    fig.suptitle("Final fusion RMS response | Per-map normalized | Overlaid on original images")
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="inferno"),
                 ax=axes.ravel().tolist(), shrink=.7, label="Normalized RMS")
    try:
        fig.savefig(filename, dpi=160)
    finally:
        plt.close(fig)


def main():
    paths = [IR_IMAGE, RGB_IMAGE, OUTPUT_DIR] + [path for _, _, config, weights in MODELS for path in (config, weights)]
    if any(not path or path == "path" for path in paths):
        raise ValueError("请先修改脚本顶部的 path 占位符")
    configs = [read_config(config) for _, _, config, _ in MODELS]
    settings = [pixel_settings(config) for config in configs]
    if any(value != settings[0] for value in settings[1:]):
        raise ValueError("四模型的验证像素预处理不同，不能直接比较")
    inputs, valid, originals = prepare_images(settings[0])
    model_rows, fusion_rows, overlay_rows = [], [], []
    for (name, modality, _, weights), config in zip(MODELS, configs):
        backbone = load_backbone(config, weights)
        maps = capture(backbone, inputs[modality], modality)
        primary = "Fused" if modality == "paired" else modality.upper()
        model_rows.append((name, {stage: value for (branch, stage), value in maps.items() if branch == primary}))
        if modality == "paired":
            for branch in ("IR", "RGB"):
                overlay_rows.append((f"{name} / {branch}", originals[branch], model_rows[-1][1]))
            for branch in ("IR", "RGB", "Fused"):
                fusion_rows.append((f"{name} / {branch}", {stage: value for (label, stage), value in maps.items() if label == branch}))
        del backbone
        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    save_plot(model_rows, valid, output / "rms_models.png")
    save_plot(fusion_rows, valid, output / "rms_fusion.png")
    save_overlays(overlay_rows, valid, output / "rms_fusion_overlay.png")
    print(f"完成：RMS 对照图及原图叠加图已保存到 {output}")


if __name__ == "__main__":
    main()
