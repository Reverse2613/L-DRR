"""离线生成770种三环境属性联合文本的原始512维 CLIP 缓存。

先按固定模板合成整段文本，再用冻结 CLIP ViT-B/16 文本塔批量编码。
不平均/相加单属性向量、不降维、不额外归一化、不训练或下载模型。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.backbone.frozen_clip_condition import (
    CLIP_FEATURE_DIM, CLIP_MODEL_ID, DEFAULT_TEXTS_PATH,
    JOINT_CACHE_KIND, JOINT_CONDITION_TEMPLATE, JOINT_COUNTS, JOINT_FEATURE_FILENAME,
    JOINT_FIELDS, JOINT_INDEX_FORMULA, JOINT_ROW_COUNT,
    build_joint_texts, joint_texts_sha256, load_joint_condition_texts,
)
from tools.build_condition_anchors import file_sha256


def encode_joint_clip_texts(
    model_dir: Path, texts: List[str], device: str, batch_size: int,
    expected_row_count: int = JOINT_ROW_COUNT,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """本地加载一次冻结 CLIP，分批编码全部联合模板文本。

    Args:
        model_dir: 用户已下载的 Hugging Face 格式 CLIP ViT-B/16 目录。
        texts: 按缓存行号排列的完整文本，超过77 token 明确报错。
        device: cpu 或 cuda；CUDA 不可用时不自动回退。
        batch_size: 正整数，每批文本数量；模型只加载一次。
        expected_row_count: 预期行数；默认770保持旧三条件接口，六条件
            缓存显式传实际不同文本数。不截断或丢弃任何文本行。

    Returns:
        (features, provenance)：CPU float32 [expected_row_count,512] 原始 text_embeds
        及 CLIP 来源、版本、批量与 token 长度记录。无额外特征变换。

    Raises:
        FileNotFoundError: 本地模型文件不完整。
        ValueError: 模型规格、输入或权重不符合契约，或文本超过77 token。
        RuntimeError: 指定 CUDA 但当前不可用。
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("device 必须显式为 cpu 或 cuda。")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数。")
    if type(expected_row_count) is not int or expected_row_count <= 0:
        raise ValueError("expected_row_count必须是正整数。")
    if len(texts) != expected_row_count or any(not isinstance(text, str) or not text for text in texts):
        raise ValueError(f"需要完整的{expected_row_count}段非空联合文本。")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"本地 CLIP 模型目录不存在：{model_dir}；不会联网下载。")
    required = ["config.json", "vocab.json", "merges.txt",
                "tokenizer_config.json", "special_tokens_map.json"]
    weight_name = "model.safetensors" if (model_dir / "model.safetensors").is_file() else "pytorch_model.bin"
    required.append(weight_name)
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{model_dir}缺少文件：{', '.join(missing)}；不会联网下载。")
    raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict) or raw_config.get("model_type") != "clip":
        raise ValueError("config.json 必须是官方完整 CLIP 配置。")

    # 延迟导入：模板/缓存的数学检查不需要 Transformers 或 CLIP 推理。
    import transformers
    from transformers import CLIPConfig, CLIPTextModelWithProjection, CLIPTokenizer

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
        raise ValueError("模型规格不符合 CLIP ViT-B/16 的512维文本/77 token 契约。")
    tokenizer = CLIPTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    # 先对全部文本核验长度，拒绝静默截断；这里只产生小型 CPU token 表。
    tokens = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    lengths = tokens["attention_mask"].sum(dim=1)
    too_long = torch.nonzero(lengths > 77).flatten().tolist()
    if too_long:
        raise ValueError(f"联合文本行{too_long}超过77 token；未截断，请核对模板与原文。")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 cuda，但当前 CUDA 不可用；未自动改用 CPU。")
    model, loading = CLIPTextModelWithProjection.from_pretrained(
        str(model_dir), local_files_only=True, torch_dtype=torch.float32,
        use_safetensors=(weight_name == "model.safetensors"), output_loading_info=True,
    )
    if loading["missing_keys"] or loading["mismatched_keys"] or loading["error_msgs"]:
        raise ValueError(f"CLIP 文本权重加载不完整，拒绝随机补齐：{loading}")
    unexpected = loading["unexpected_keys"]
    invalid = [key for key in unexpected if not (
        key.startswith(("vision_model.", "visual_projection.")) or key == "logit_scale"
    )]
    if invalid:
        raise ValueError(f"发现非预期的未使用 CLIP 权重：{invalid}")
    model = model.to(device).eval().requires_grad_(False)
    batches: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        with torch.inference_mode():
            # text_embeds 已含冻结 CLIP 内部的 text_projection；这里不再进行
            # L2归一化、512→64降维或任何可训练适配。
            features = model(**{key: value[start:end].to(device) for key, value in tokens.items()}).text_embeds
        batches.append(features.cpu().float())
        print(f"CLIP 联合文本编码：{end}/{len(texts)}")
    combined = torch.cat(batches, dim=0)
    if combined.shape != (expected_row_count, CLIP_FEATURE_DIM) or not torch.isfinite(combined).all():
        raise ValueError(f"CLIP 联合文本输出必须是有限 [{expected_row_count},512] 特征。")
    if (torch.linalg.vector_norm(combined, dim=1) <= 1e-7).any():
        raise ValueError("CLIP 联合文本输出含零向量或数值近零的行。")
    provenance: Dict[str, Any] = {
        "model_id": CLIP_MODEL_ID, "local_model_dir": str(model_dir),
        "transformers_version": transformers.__version__, "device": device,
        "batch_size": batch_size, "max_tokens": int(lengths.max().item()),
        "files_sha256": {name: file_sha256(model_dir / name) for name in required},
        "expected_unused_vision_keys": unexpected,
    }
    return combined, provenance


def build_joint_anchors(
    texts_path: Path, model_dir: Path, output_dir: Path, device: str, batch_size: int,
) -> None:
    """生成完整联合缓存并记录可核验的模板、编号与模型来源。

    Args:
        texts_path: 官方三属性原文 JSON。
        model_dir: 用户已下载的本地 CLIP ViT-B/16 目录。
        output_dir: 新缓存目录；已有目录拒绝覆盖。
        device: 编码设备 cpu 或 cuda。
        batch_size: 文本推理批量，默认 CLI 使用64。

    Returns:
        无返回值。保存 joint_condition_anchor.pt [770,512] 和 manifest.json。
        模型训练只读取这张冻结表，不需将 CLIP 放进检测模型。
    """
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在：{output_dir}；请换新目录，未覆盖缓存。")
    source_texts = load_joint_condition_texts(texts_path)
    joint_texts, combination_ids = build_joint_texts(source_texts)
    features, provenance = encode_joint_clip_texts(model_dir, joint_texts, device, batch_size)
    manifest: Dict[str, Any] = {
        "schema_version": 1, "cache_kind": JOINT_CACHE_KIND,
        "template": JOINT_CONDITION_TEMPLATE, "feature_transform": "none",
        "condition_dim": CLIP_FEATURE_DIM, "condition_counts": JOINT_COUNTS,
        "id_order": list(JOINT_FIELDS), "index_formula": JOINT_INDEX_FORMULA,
        "texts_file_sha256": file_sha256(texts_path), "texts_by_id": source_texts,
        "combination_ids": combination_ids, "joint_texts_by_index": joint_texts,
        "joint_texts_sha256": joint_texts_sha256(joint_texts),
        "generator_script_sha256": file_sha256(Path(__file__)),
        "model": provenance, "torch_version": str(torch.__version__), "artifacts": {},
    }
    # 文本/模型推理全部通过后才建立目录；manifest 最后写入标记完整缓存。
    output_dir.mkdir(parents=True, exist_ok=False)
    feature_path = output_dir / JOINT_FEATURE_FILENAME
    torch.save(features.detach().clone(), feature_path)
    manifest["artifacts"][JOINT_FEATURE_FILENAME] = {
        "shape": list(features.shape), "sha256": file_sha256(feature_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"联合条件缓存已保存：{output_dir}，形状[770,512]，无降维/额外归一化。")


def main() -> None:
    """解析本地模型与输出参数，并执行一次性离线缓存生成。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, default=DEFAULT_TEXTS_PATH)
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "pretrain/clip-vit-base-patch16")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "pretrain/condition_anchors/clip_vit_b16_joint_512")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    build_joint_anchors(
        args.texts.expanduser().resolve(), args.model_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve(), args.device, args.batch_size,
    )


if __name__ == "__main__":
    main()
