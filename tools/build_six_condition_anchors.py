"""离线编码完整六条件原文；使用现有本地冻结CLIP，不训练、不下载。

按用户模板拼装全部不同文本，一条文本对应一个512维向量。
XML只使用顶层六属性；完整时间原文含日期，保持原样。
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.backbone.frozen_clip_condition import (
    CLIP_FEATURE_DIM, DEFAULT_TEXTS_PATH, joint_texts_sha256,
)
from engine.backbone.frozen_six_condition import SIX_CACHE_KIND, SIX_FEATURE_FILENAME
from engine.data.condition_texts import (
    CONDITION_COUNTS, SIX_CONDITION_TEMPLATE, SIX_XML_FIELDS,
    build_six_condition_text, load_condition_descriptions, read_raw_six_conditions,
)
from tools.build_condition_anchors import file_sha256
from tools.build_joint_condition_anchors import encode_joint_clip_texts


def collect_six_condition_rows(
    ann_dirs: Sequence[Path], texts_path: Path,
) -> Tuple[List[str], List[Dict[str, str]], List[Dict[str, Any]]]:
    """收集实际出现的完整六条件文本，按文本排序建立确定性行号。

    Args:
        ann_dirs: 显式XML目录，默认本轮train/test；不读取目标类别或框。
        texts_path: 官方原始属性描述表。

    Returns:
        (N条文本, N份原始六条件, 各目录的元数据清单摘要)。相同文本仅
        编码一次；不按检测结果选文本，不穷举连续量的笛卡尔积。

    Raises:
        ValueError: 目录重复/为空或条件不合法。
        FileNotFoundError: XML目录不存在。
    """
    if not ann_dirs or len({str(path.resolve()) for path in ann_dirs}) != len(ann_dirs):
        raise ValueError("ann_dirs必须为非空、不重复的XML目录列表。")
    descriptions = load_condition_descriptions(texts_path)
    by_text: Dict[str, Dict[str, str]] = {}
    sources: List[Dict[str, Any]] = []
    for directory in ann_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"XML目录不存在：{directory}。")
        paths = sorted(directory.glob("*.xml"))
        if not paths:
            raise ValueError(f"XML目录为空：{directory}。")
        inventory: List[str] = []
        for path in paths:
            raw = read_raw_six_conditions(ET.parse(path).getroot(), str(path))
            text = build_six_condition_text(raw, descriptions, str(path))
            by_text.setdefault(text, raw)
            # 清单哈希只包含文件名和图像级六属性，不包含object标注。
            inventory.append(json.dumps({"xml": path.name, "conditions": raw},
                                        ensure_ascii=False, sort_keys=True))
        sources.append({"ann_dir": str(directory.resolve()), "xml_count": len(paths),
                        "metadata_sha256": joint_texts_sha256(inventory)})
    texts = sorted(by_text)
    return texts, [by_text[text] for text in texts], sources


def build_six_anchors(
    ann_dirs: Sequence[Path], texts_path: Path, model_dir: Path,
    output_dir: Path, device: str, batch_size: int,
) -> None:
    """生成原始512维六条件缓存；全部检查通过后才写新目录。

    Args:
        ann_dirs: 本轮需要缓存的图像级条件来源目录。
        texts_path: 官方原始属性文本JSON。
        model_dir: 已有本地CLIP ViT-B/16目录；缺文件报错，不下载。
        output_dir: 新缓存目录；已有目录拒绝覆盖。
        device: cpu或cuda；指定CUDA时不自动回退。
        batch_size: 文本编码批量。

    Returns:
        保存six_condition_anchor.pt [N,512]与可校验manifest.json。
    """
    if output_dir.exists():
        raise FileExistsError(f"六条件缓存目录已存在：{output_dir}；未覆盖，请换新目录。")
    texts, raw_rows, sources = collect_six_condition_rows(ann_dirs, texts_path)
    print(f"六条件原文：{sum(source['xml_count'] for source in sources)}份XML，{len(texts)}种完整文本。")
    features, provenance = encode_joint_clip_texts(
        model_dir, texts, device, batch_size, expected_row_count=len(texts),
    )
    manifest: Dict[str, Any] = {
        "schema_version": 1, "cache_kind": SIX_CACHE_KIND,
        "template": SIX_CONDITION_TEMPLATE, "xml_fields": list(SIX_XML_FIELDS),
        "condition_counts": CONDITION_COUNTS, "condition_dim": CLIP_FEATURE_DIM,
        "feature_transform": "none", "time_representation": "verbatim_xml_including_date",
        "row_count": len(texts), "condition_texts_by_index": texts,
        "raw_conditions_by_index": raw_rows, "texts_sha256": joint_texts_sha256(texts),
        "texts_file_sha256": file_sha256(texts_path), "source_metadata": sources,
        "generator_script_sha256": file_sha256(Path(__file__)),
        "model": provenance, "torch_version": str(torch.__version__), "artifacts": {},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    feature_path = output_dir / SIX_FEATURE_FILENAME
    torch.save(features.detach().clone(), feature_path)
    manifest["artifacts"][SIX_FEATURE_FILENAME] = {
        "shape": list(features.shape), "sha256": file_sha256(feature_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"六条件缓存已保存：{output_dir}，[{len(texts)},512]，最长{provenance['max_tokens']} token。")


def main() -> None:
    """解析本地路径，生成一次性六条件文本缓存。"""
    data_root = Path("/home/reverse/workspace/data/ATR-UMOD")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann-dirs", nargs="+", type=Path,
                        default=[data_root / split / "labels_ir" for split in ("train", "test")])
    parser.add_argument("--texts", type=Path, default=DEFAULT_TEXTS_PATH)
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "pretrain/clip-vit-base-patch16")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "pretrain/condition_anchors/clip_vit_b16_six_512")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    build_six_anchors(
        [path.expanduser().resolve() for path in args.ann_dirs],
        args.texts.expanduser().resolve(), args.model_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve(), args.device, args.batch_size,
    )


if __name__ == "__main__":
    main()
