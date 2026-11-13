"""C/D/E/F限时CUDA检查；不训练、不下载、不覆盖既有模型/缓存。

六条件临时缓存只写/tmp；检查真实样本推理、一次AMP反向、EMA与旧权重。
每个模型检查只取batch2，不计算整集AP，不调用optimizer.step。
"""

import argparse
import gc
import json
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def timeout_error(_signum: int, _frame: Any) -> None:
    """超时立即失败，不把未完成检查记录为通过。

    Args:
        _signum: POSIX alarm信号。
        _frame: 当前执行帧。

    Raises:
        TimeoutError: 总检查达到540秒。
    """
    raise TimeoutError("条件对照检查超过540秒；检查未完成，不自动续跑。")


def check_controls(device: torch.device) -> Dict[str, Any]:
    """检验常量/随机条件的含义和真实FiLM旁路。

    Args:
        device: 可访问CUDA的设备，不回退CPU。

    Returns:
        冻结、随机状态、范数与旁路检查摘要。
    """
    from engine.backbone.cm_cme import CMCME
    from engine.backbone.frozen_clip_condition import FrozenCLIPConditionPrior
    from engine.backbone.frozen_clip_controls import ConstantCLIPConditionPrior, RandomCLIPConditionPrior

    anchor = REPO_ROOT / "pretrain/condition_anchors/clip_vit_b16_joint_512"
    real = FrozenCLIPConditionPrior(anchor).to(device)
    constant = ConstantCLIPConditionPrior(anchor).to(device)
    before = torch.random.get_rng_state().clone()
    random1 = RandomCLIPConditionPrior(anchor).to(device)
    random2 = RandomCLIPConditionPrior(anchor).to(device)
    assert torch.equal(before, torch.random.get_rng_state())
    conditions = {field: torch.tensor(values, device=device, dtype=torch.int64)
                  for field, values in {"weather": [0, 6], "illumination": [0, 6], "location": [0, 1]}.items()}
    torch.testing.assert_close(constant(conditions)[0], constant(conditions)[1], rtol=0, atol=0)
    torch.testing.assert_close(random1(conditions), random2(conditions), rtol=0, atol=0)
    assert not torch.equal(random1(conditions)[0], random1(conditions)[1])
    torch.testing.assert_close(random1.random_anchor.norm(dim=1), real.joint_anchor.norm(dim=1),
                               atol=2e-6, rtol=2e-6)
    assert all(not prior(conditions).requires_grad for prior in (real, constant, random1))
    # 用非零FiLM偏置检验旁路，避免初态FiLM本来就是恒等而误判。
    film = CMCME(32, condition_dim=512).to(device)
    with torch.no_grad():
        film.rgb_film.bias.fill_(0.5)
        film.ir_film.bias.fill_(0.25)
    bypass = CMCME(32, condition_dim=512, enable_film=False).to(device)
    bypass.load_state_dict(film.state_dict(), strict=True)
    rgb, ir = (torch.randn(2, 32, 4, 5, device=device) for _ in range(2))
    c = real(conditions)
    output = bypass(rgb, ir, c)
    torch.testing.assert_close(output[0], bypass.rgb_work(rgb), rtol=0, atol=0)
    torch.testing.assert_close(output[1], bypass.ir_work(ir), rtol=0, atol=0)
    assert not torch.allclose(output[0], film(rgb, ir, c)[0])
    return {"constant_same_across_images": True, "random_reproducible": True,
            "random_norms_match": True, "global_rng_unchanged": True, "nonzero_film_bypassed": True}


def check_legacy_add(device: torch.device) -> Dict[str, Any]:
    """严格加载已完成Add的EMA权重，检查旧图像输入仍能推理。

    Args:
        device: CUDA设备。

    Returns:
        权重兼容与单样本输出形状；不重跑旧实验或计算AP。
    """
    from engine.core import YAMLConfig

    cfg = YAMLConfig(str(REPO_ROOT / "configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_add_baseline.yml"))
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = 1
    model = cfg.model.to(device).eval()
    path = REPO_ROOT / "outputs/rtdetrv2_obb_hgnetv2_s_atrumod_add_baseline/best_stg1.pth"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["ema"]["module"], strict=True)
    images, _ = next(iter(cfg.val_dataloader))
    with torch.inference_mode():
        output = model(images.to(device))
    assert output["pred_logits"].shape == (1, 300, 11)
    assert output["pred_boxes"].shape == (1, 300, 5)
    assert all(torch.isfinite(output[key]).all() for key in ("pred_logits", "pred_boxes"))
    return {"strict_ema_load": True, "image_only_forward": True}


def run_checks(device: torch.device) -> Dict[str, Any]:
    """按顺序检查四组新配置，六条件缓存始终保留在临时目录。

    Args:
        device: CUDA设备；模型逐个释放，避免并行争用显存。

    Returns:
        控制语义、缓存以及新旧模型的完整检查摘要。
    """
    from engine.backbone.frozen_six_condition import FrozenSixCLIPConditionPrior
    from engine.data.condition_texts import build_six_condition_text, load_condition_descriptions
    from tools.build_six_condition_anchors import build_six_anchors
    from tools.check_module1 import check_module1

    result: Dict[str, Any] = {"controls": check_controls(device)}
    gc.collect()
    torch.cuda.empty_cache()
    data_root = Path("/home/reverse/workspace/data/ATR-UMOD")
    texts_path = REPO_ROOT / "configs/dataset/atrumod_condition_texts.json"
    with tempfile.TemporaryDirectory(prefix="rio_condition_ablation_") as temporary:
        cache = Path(temporary) / "six"
        build_six_anchors(
            [data_root / split / "labels_ir" for split in ("train", "test")],
            texts_path, REPO_ROOT / "pretrain/clip-vit-base-patch16", cache, "cuda", 64,
        )
        prior = FrozenSixCLIPConditionPrior(cache).to(device)
        assert not prior({"condition_text": [prior.condition_texts[0]]}).requires_grad
        try:
            prior({"condition_text": ["uncached six-condition text"]})
        except ValueError:
            pass
        else:
            raise AssertionError("未缓存六条件文本没有报错。")
        manifest = json.loads((cache / "manifest.json").read_text())
        invalid = dict(manifest["raw_conditions_by_index"][0], time="2024-02-30T12:00")
        try:
            build_six_condition_text(invalid, load_condition_descriptions(texts_path))
        except ValueError:
            pass
        else:
            raise AssertionError("非法日期被静默接受。")
        result["six_cache"] = {"rows": manifest["row_count"], "max_tokens": manifest["model"]["max_tokens"],
                               "original_time_preserved": True, "unknown_text_rejected": True,
                               "invalid_date_rejected": True}
        del prior
        for group, stem in (("C", "joint_clip512_constant"), ("D", "joint_clip512_random"),
                            ("E", "joint_clip512_no_film"), ("F", "six_condition_clip512")):
            torch.manual_seed(0)
            config = REPO_ROOT / "configs/rtdetrv2_obb" / f"rtdetrv2_obb_hgnetv2_s_atrumod_{stem}.yml"
            result[group] = check_module1(config, 2, device, True, cache if group == "F" else None)
            print(f"{group}组真实样本GPU前后向、profiler与EMA检查通过。")
            gc.collect()
            torch.cuda.empty_cache()
        for group, stem in (("old_module1", "module1"), ("old_joint", "module1_joint_clip512")):
            config = REPO_ROOT / "configs/rtdetrv2_obb" / f"rtdetrv2_obb_hgnetv2_s_atrumod_{stem}.yml"
            checkpoint = REPO_ROOT / "outputs" / f"rtdetrv2_obb_hgnetv2_s_atrumod_{stem}" / "best_stg1.pth"
            result[group] = check_module1(config, 2, device, False, checkpoint=checkpoint)
            print(f"{group}旧EMA权重严格加载与显式条件检查通过。")
            gc.collect()
            torch.cuda.empty_cache()
        result["old_add"] = check_legacy_add(device)
    return result


def main() -> None:
    """限时运行CUDA检查，可选保存新报告；已有报告拒绝覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="可选新JSON报告路径，已有文件不覆盖")
    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        raise FileExistsError(f"报告已存在：{args.report}；请指定新路径。")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；本检查不自动回退CPU。")
    signal.signal(signal.SIGALRM, timeout_error)
    signal.alarm(540)
    start = time.monotonic()
    torch.set_num_threads(2)
    result = run_checks(torch.device("cuda:0"))
    result["elapsed_seconds"] = round(time.monotonic() - start, 2)
    result["optimizer_steps"] = 0
    result["device"] = torch.cuda.get_device_name(0)
    signal.alarm(0)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
