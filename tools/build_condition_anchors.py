"""将 ATR-UMOD 原始属性文本转换为 DT-CPN 的固定 CLIP 锚点。

阅读顺序：load_condition_texts → encode_clip_texts →
make_orthogonal_projection → project_and_normalize → build_anchors。

只读取用户已下载的本地 CLIP ViT-B/16，不下载权重、不训练模型。
三个属性共用一个 P[64,512]；最终缓存每一行对应原始类别编号。
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
# Python 字典保持插入顺序：合并后前10行天气、接着7行光照、最后11行场景。
ATTRIBUTE_SPECS = {
    "weather": (10, "weather_anchor.pt"),
    "illumination": (7, "illum_anchor.pt"),
    "location": (11, "loc_anchor.pt"),
}
MODEL_ID = "openai/clip-vit-base-patch16"
PROJECTION_SEED = 20261109


def file_sha256(path: Path) -> str:
    """分块计算文件校验值，用于追踪文本、权重和缓存来源。

    Args:
        path: 已存在的文件路径。

    Returns:
        文件内容的 SHA-256 十六进制字符串。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_condition_texts(path: Path) -> Dict[str, List[str]]:
    """读取按原始编号排序的文本，不增加 prompt、不修正原文。

    Args:
        path: configs/dataset/atrumod_condition_texts.json 的路径。

    Returns:
        天气10条、光照7条、场景11条文本；各列表第k项对应编号k。

    Raises:
        ValueError: 版本、类别数量、编号顺序或文本不合法。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"{path}: 需要 schema_version=1 的条件文本文件。")
    result: Dict[str, List[str]] = {}
    for field, (count, _) in ATTRIBUTE_SPECS.items():
        rows = data.get(field)
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError(f"{path}: {field} 必须有 {count} 行。")
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or type(row.get("id")) is not int:
                raise ValueError(f"{field}[{index}]: id 必须是整数。")
            if row["id"] != index:
                raise ValueError(f"{field}[{index}]: 应为 id={index}，不自动重排。")
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"{field}[{index}]: 缺少非空 text。")
        result[field] = [row["text"] for row in rows]
    return result


def encode_clip_texts(
    model_dir: Path, texts: List[str], device: str
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """从本地官方格式 CLIP ViT-B/16 提取投影后的原始文本特征。

    Args:
        model_dir: 用户下载的 Hugging Face 格式模型目录，不接受远程模型ID。
        texts: N条原始文本，本任务N=28；超过77 token会报错，不截断。
        device: 显式选择 cpu 或 cuda，不自动回退设备。

    Returns:
        CPU float32特征 [N,512]，以及实际加载文件/版本等来源记录。

    Raises:
        FileNotFoundError: 本地文件不全。
        ValueError: 模型规格不符、文本过长或文本权重未完整加载。
    """
    # 延迟导入：数学检查无需加载 Transformers 或任何预训练模型。
    import transformers
    from transformers import CLIPConfig, CLIPTextModelWithProjection, CLIPTokenizer

    if not model_dir.is_dir():
        raise FileNotFoundError(f"本地模型目录不存在：{model_dir}；请按 flow.md 自行下载。")
    required = ["config.json", "vocab.json", "merges.txt",
                "tokenizer_config.json", "special_tokens_map.json"]
    # 两种格式都支持，若两者都有则明确优先使用 safetensors。
    weight_name = "model.safetensors" if (model_dir / "model.safetensors").is_file() else "pytorch_model.bin"
    required.append(weight_name)
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{model_dir} 缺少文件：{', '.join(missing)}；不会联网下载。")
    raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict) or raw_config.get("model_type") != "clip":
        raise ValueError("config.json 必须是官方完整CLIP配置，不能用其他模型配置自动补默认值。")
    config = CLIPConfig.from_pretrained(str(model_dir), local_files_only=True)
    if (config.projection_dim != 512 or config.text_config.hidden_size != 512
            or config.text_config.projection_dim != 512
            or config.text_config.num_hidden_layers != 12
            or config.text_config.num_attention_heads != 8
            or config.text_config.vocab_size != 49408
            or config.text_config.max_position_embeddings != 77
            or config.vision_config.patch_size != 16
            or config.vision_config.hidden_size != 768
            or config.vision_config.num_hidden_layers != 12):
        raise ValueError("模型配置不符合 CLIP ViT-B/16 的512维文本/77 token规格。")
    tokenizer = CLIPTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    tokens = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    lengths = tokens["attention_mask"].sum(dim=1)
    too_long = torch.nonzero(lengths > 77).flatten().tolist()
    if too_long:
        raise ValueError(f"文本行 {too_long} 超过77 token；请检查原文，未进行截断。")
    model, loading = CLIPTextModelWithProjection.from_pretrained(
        str(model_dir), local_files_only=True, torch_dtype=torch.float32,
        use_safetensors=(weight_name == "model.safetensors"), output_loading_info=True,
    )
    if loading["missing_keys"] or loading["mismatched_keys"] or loading["error_msgs"]:
        raise ValueError(f"CLIP文本权重加载不完整，拒绝使用随机补齐参数：{loading}")
    # 完整CLIP文件含视觉塔；本工具只提取文本，以下多余权重是预期情况。
    unexpected = loading["unexpected_keys"]
    invalid = [key for key in unexpected if not (
        key.startswith(("vision_model.", "visual_projection.")) or key == "logit_scale"
    )]
    if invalid:
        raise ValueError(f"发现非预期的未使用权重：{invalid}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了cuda，但当前CUDA不可用；未自动改用CPU。")
    model = model.to(device).eval().requires_grad_(False)
    with torch.inference_mode():
        # text_embeds已包含CLIP的text_projection，不是token平均或未经投影的隐藏状态。
        features = model(**{key: value.to(device) for key, value in tokens.items()}).text_embeds
    features = features.cpu().float()
    if features.shape != (len(texts), 512) or not torch.isfinite(features).all():
        raise ValueError("CLIP输出必须是有限的 [N,512] 特征。")
    return features, {
        "model_id": MODEL_ID, "local_model_dir": str(model_dir),
        "transformers_version": transformers.__version__, "device": device,
        "files_sha256": {name: file_sha256(model_dir / name) for name in required},
        "expected_unused_vision_keys": unexpected,
    }


def make_orthogonal_projection(seed: int = PROJECTION_SEED) -> torch.Tensor:
    """生成固定的64×512矩阵P，满足行正交 P @ P.T ≈ I。

    Args:
        seed: 本地CPU随机生成器种子，不改变训练使用的全局随机状态。

    Returns:
        CPU float32矩阵 [64,512]；三个属性必须共用这一份矩阵。
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # 对512×64高斯矩阵做QR，得到64个相互正交的512维列向量。
    gaussian = torch.randn(512, 64, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian, mode="reduced")
    # 固定QR的符号约定，消除同一分解中列向量任意取正负的歧义。
    signs = torch.where(torch.diagonal(r) >= 0, 1.0, -1.0)
    projection = (q * signs).T.contiguous().float()
    error = (projection @ projection.T - torch.eye(64)).abs().max().item()
    if error > 1e-5:
        raise ValueError(f"投影矩阵行正交检查失败，最大误差={error}。")
    return projection


def project_and_normalize(features: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """执行批量线性投影与单位长度归一化，遇到零向量明确报错。

    Args:
        features: CPU float32的CLIP原始文本特征 [N,512]。
        projection: 所有属性共享的CPU float32矩阵P [64,512]。

    Returns:
        CPU float32锚点 [N,64]；每行L2范数在浮点误差内等于1。

    Raises:
        ValueError: 输入形状/数值非法，或投影后某行长度接近0。
    """
    if features.ndim != 2 or features.shape[1] != 512 or projection.shape != (64, 512):
        raise ValueError("输入应为 features[N,512] 和 projection[64,512]。")
    if not torch.isfinite(features).all() or not torch.isfinite(projection).all():
        raise ValueError("输入特征或投影矩阵含NaN/Inf。")
    # 数学式用列向量 P @ t；这里每条文本是一行，等价写作 features @ P.T。
    projected = features @ projection.T
    lengths = torch.linalg.vector_norm(projected, dim=1, keepdim=True)
    bad_rows = (~torch.isfinite(lengths) | (lengths <= 1e-12)).flatten()
    if bad_rows.any():
        raise ValueError(f"无法归一化文本行：{torch.nonzero(bad_rows).flatten().tolist()}")
    return (projected / lengths).detach().contiguous()


def cosine_distortion(features: torch.Tensor, anchors: torch.Tensor) -> Dict[str, float]:
    """比较降维前后不同文本之间的余弦，不把正交误认为最小失真。

    Args:
        features: 有限非零的CLIP特征 [N,512]，本任务N=28。
        anchors: 对应的单位锚点 [N,64]。

    Returns:
        不同文本无序对的余弦绝对误差均值与最大值；28条文本共378对。
    """
    normalized = features / torch.linalg.vector_norm(features, dim=1, keepdim=True)
    before = normalized @ normalized.T
    after = anchors @ anchors.T
    rows, cols = torch.triu_indices(features.shape[0], features.shape[0], offset=1)
    error = (before[rows, cols] - after[rows, cols]).abs()
    return {"mean_absolute_error": error.mean().item(), "max_absolute_error": error.max().item()}


def build_anchors(texts_path: Path, model_dir: Path, output_dir: Path, seed: int, device: str) -> None:
    """按原始类别编号生成三个锚点表，并保存复现信息。

    Args:
        texts_path: 28条原始属性文本的JSON文件。
        model_dir: 已下载的本地CLIP ViT-B/16目录。
        output_dir: 新缓存目录；已有目录拒绝覆盖，避免混用不同版本。
        seed: 正交投影的随机种子。
        device: 文本编码设备cpu/cuda；QR和最终缓存固定在CPU。

    Returns:
        无返回值。保存 [10,64]/[7,64]/[11,64] 锚点、P、原始特征和manifest。
    """
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{output_dir}；请指定新目录，未覆盖缓存。")
    texts = load_condition_texts(texts_path)
    flattened = [text for field in ATTRIBUTE_SPECS for text in texts[field]]
    features, provenance = encode_clip_texts(model_dir, flattened, device)
    projection = make_orthogonal_projection(seed)
    anchors = project_and_normalize(features, projection)
    distortion = cosine_distortion(features, anchors)
    manifest: Dict[str, Any] = {
        "schema_version": 1, "text_policy": "verbatim; no prompt rewriting",
        "texts_file_sha256": file_sha256(texts_path), "texts_by_id": texts,
        "generator_script_sha256": file_sha256(Path(__file__)),
        "model": provenance, "torch_version": str(torch.__version__),
        "projection": {
            "seed": seed, "shape": [64, 512],
            "method": "CPU float64 Gaussian QR, sign-canonicalized, stored float32",
            "row_orthogonality_max_error": (projection @ projection.T - torch.eye(64)).abs().max().item(),
        },
        "cosine_distortion_378_pairs": distortion, "artifacts": {},
    }
    # 所有特征计算成功后才建立输出目录；manifest最后写入，标记完整缓存。
    output_dir.mkdir(parents=True, exist_ok=False)
    start = 0
    for field, (count, filename) in ATTRIBUTE_SPECS.items():
        # clone避免小切片在torch.save时连带保存整个28行的底层存储。
        table = anchors[start:start + count].clone()
        torch.save(table, output_dir / filename)
        manifest["artifacts"][filename] = {"shape": list(table.shape), "sha256": file_sha256(output_dir / filename)}
        print(f"{filename}: {list(table.shape)}，最大范数误差={(table.norm(dim=1) - 1).abs().max().item():.2e}")
        start += count
    for filename, tensor in (("projection.pt", projection), ("text_features_512.pt", features)):
        torch.save(tensor.clone(), output_dir / filename)
        manifest["artifacts"][filename] = {"shape": list(tensor.shape), "sha256": file_sha256(output_dir / filename)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"余弦绝对变化：均值={distortion['mean_absolute_error']:.6f}，最大={distortion['max_absolute_error']:.6f}")
    print(f"缓存已保存：{output_dir}（没有进行模型训练）")


def main() -> None:
    """解析命令行参数，运行本地离线锚点生成流程。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="用户下载的本地CLIP ViT-B/16目录")
    parser.add_argument("--texts", type=Path, default=REPO_ROOT / "configs/dataset/atrumod_condition_texts.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "pretrain/condition_anchors/clip_vit_b16_64")
    parser.add_argument("--seed", type=int, default=PROJECTION_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    build_anchors(args.texts.resolve(), args.model_dir.expanduser().resolve(),
                  args.output_dir.expanduser().resolve(), args.seed, args.device)


if __name__ == "__main__":
    main()
