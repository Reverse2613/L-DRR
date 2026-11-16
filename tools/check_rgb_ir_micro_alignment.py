"""B+RGB→IR微校准限时GPU验收；不执行优化步、整集评测或权重下载。

检查像素方向/边界、零初态、梯度、B配方/旧EMA兼容、真实batch8和profiler。
总时限540秒；只有所有检查完成后才写新的JSON报告。
"""

import argparse
import copy
import gc
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
CONFIG_DIR = REPO_ROOT / "configs/rtdetrv2_obb"
BASE_CONFIG = CONFIG_DIR / "rtdetrv2_obb_hgnetv2_s_atrumod_module1_joint_clip512.yml"
ALIGN_CONFIG = CONFIG_DIR / "rtdetrv2_obb_hgnetv2_s_atrumod_joint_clip512_rgb_ir_micro_align.yml"
BASE_CHECKPOINT = REPO_ROOT / "outputs/rtdetrv2_obb_hgnetv2_s_atrumod_module1_joint_clip512/best_stg1.pth"
ALIGN_PREFIX = "backbone.rgb_ir_alignments."


def timeout_error(_signum: int, _frame: Any) -> None:
    """超时立即失败，不把部分检查记为通过。

    Args:
        _signum: POSIX告警信号。
        _frame: 当前执行帧。
    """
    raise TimeoutError("RGB→IR微校准检查达到540秒，未完成，不自动续跑。")


def expect_value_error(action: Callable[[], Any]) -> None:
    """确保非法条件确实报错。

    Args:
        action: 不应成功返回的零参数检查调用。
    """
    try:
        action()
    except ValueError:
        return
    raise AssertionError("非法微校准输入被静默接受。")


def check_sampling(device: torch.device) -> Dict[str, Any]:
    """用可解析的斜坡图核验采样方向、尺度单位与两阶段梯度。

    Args:
        device: CUDA设备。

    Returns:
        零偏移、正向采样、统一像素上限、IR保持和梯度检查摘要。
    """
    from engine.backbone.rgb_ir_micro_alignment import RGBIRMicroAlignment, build_alignment_geometry

    height = torch.tensor([[0.0], [1.0]], device=device)
    angle = torch.tensor([[1.0], [0.0]], device=device)
    geometry, fraction = build_alignment_geometry(height, angle)
    assert geometry.shape == (2, 5) and fraction.shape == (2, 1)
    expect_value_error(lambda: build_alignment_geometry(height, angle + 1.0))
    expect_value_error(lambda: build_alignment_geometry(height.double(), angle))
    expect_value_error(lambda: build_alignment_geometry(height[:, 0], angle))
    expect_value_error(lambda: build_alignment_geometry(height * float("nan"), angle))
    source = torch.randn(2, 8, 6, 13, device=device)
    ir = torch.randn_like(source)
    ir_before = ir.clone()
    bounds = []
    for stride in (8, 16, 32):
        layer = RGBIRMicroAlignment(8, stride, 8.0).to(device)
        torch.testing.assert_close(layer(source, ir, geometry, fraction), source, rtol=0, atol=0)
        with torch.autocast("cuda", dtype=torch.float16):
            half = source.half()
            torch.testing.assert_close(layer(half, ir.half(), geometry, fraction), half, rtol=0, atol=0)
        # 饱和偏移必须在三尺度恢复为同一输入像素上限，而不是同一特征像素。
        with torch.no_grad():
            layer.offset_head.bias.copy_(layer.offset_head.bias.new_tensor((80.0, -80.0)))
        offsets = layer.compute_offsets(source, ir, geometry, fraction)
        target_bound = 8.0 * fraction[:, :, None, None].expand(-1, 1, 6, 13)
        torch.testing.assert_close(offsets[:, :1], target_bound, rtol=0, atol=0)
        torch.testing.assert_close(offsets[:, 1:], -target_bound, rtol=0, atol=0)
        bounds.append(float(offsets[0, 0, 0, 0]))
        # 正x输入像素偏移=4：内部斜坡应增加4/stride，验证逆向重采样方向。
        with torch.no_grad():
            layer.offset_head.bias.copy_(layer.offset_head.bias.new_tensor((torch.atanh(torch.tensor(0.5)).item(), 0.0)))
        ramp = torch.arange(13, device=device, dtype=torch.float32)[None, None, None, :].expand(2, 8, 6, -1).clone()
        sampled = layer(ramp, ir, geometry, fraction)
        expected = ramp + (4.0 * fraction / stride)[:, :, None, None]
        torch.testing.assert_close(sampled[..., 1:-2], expected[..., 1:-2], rtol=0, atol=3e-6)
    torch.testing.assert_close(ir, ir_before, rtol=0, atol=0)

    layer = RGBIRMicroAlignment(8, 8, 8.0).to(device)
    target = source + torch.randn_like(source)
    (layer(source, ir, geometry, fraction) - target).square().mean().backward()
    first = layer.offset_head.weight.grad
    hidden_first = layer.spatial_context[0].weight.grad
    assert first is not None and torch.isfinite(first).all() and first.abs().sum() > 0
    assert hidden_first is not None and hidden_first.abs().sum() == 0
    # 只在合成夹具手工扰动读出，验证启动后隐藏卷积也可学；不更新正式模型。
    layer.zero_grad(set_to_none=True)
    with torch.no_grad():
        layer.offset_head.weight.normal_(0, 0.01)
    (layer(source, ir, geometry, fraction) - target).square().mean().backward()
    hidden_later = layer.spatial_context[0].weight.grad
    assert hidden_later is not None and torch.isfinite(hidden_later).all() and hidden_later.abs().sum() > 0
    return {"zero_offset_exact_identity_fp32_fp16": True, "positive_x_sampling_verified": True,
            "input_pixel_bounds_by_stride": bounds, "ir_unmodified": True,
            "invalid_metadata_rejected": True, "first_head_gradient_nonzero": True,
            "hidden_gradient_zero_at_zero_head_expected": True, "hidden_gradient_after_fixture_perturbation_nonzero": True}


def check_recipe(base: Any, aligned: Any) -> Dict[str, Any]:
    """核验展开配置仅增加微校准选项和独立输出目录。

    Args:
        base: 原B的YAMLConfig。
        aligned: 新原型的YAMLConfig。

    Returns:
        配方、图像规格、训练预算与唯一结构差异摘要。
    """
    left, right = copy.deepcopy(base.yaml_cfg), copy.deepcopy(aligned.yaml_cfg)
    for config in (left, right):
        config.pop("__include__", None)
        config.pop("output_dir")
    assert right["DualHGNetv2_PG"].pop("enable_rgb_ir_alignment") is True
    assert right["DualHGNetv2_PG"].pop("alignment_max_shift_px") == 8.0
    assert left == right, "新原型改动了B的其他展开配置。"
    assert aligned.yaml_cfg["output_dir"] != base.yaml_cfg["output_dir"]
    assert aligned.yaml_cfg["epoches"] == 36
    return {"all_other_expanded_config_values_equal": True, "epochs": 36,
            "train_batch": aligned.yaml_cfg["train_dataloader"]["total_batch_size"],
            "warmup_iter": aligned.yaml_cfg["warmup_iter"], "flat_epoch": aligned.yaml_cfg["flat_epoch"],
            "optimizer": aligned.yaml_cfg["optimizer"], "eval_spatial_size": aligned.yaml_cfg["eval_spatial_size"]}


def parameter_roles(optimizer: torch.optim.Optimizer) -> Dict[str, Tuple[float, float]]:
    """按参数名读取已有分组的学习率与衰减角色。

    Args:
        optimizer: 已构造但未执行step的AdamW。

    Returns:
        参数名到(lr, weight_decay)的映射。
    """
    return {name: (group["lr"], group["weight_decay"])
            for group in optimizer.param_groups for name in group["param_names"]}


def forward_latency_ms(model: torch.nn.Module, images: torch.Tensor, conditions: Dict[str, Any]) -> float:
    """测小样本PyTorch前向开销，仅作启动检查。

    Args:
        model: eval状态检测器。
        images: 真实[1,6,512,640]CUDA图像。
        conditions: 该模型声明的显式元数据，不含GT框/类别。

    Returns:
        2次预热后5次FP32前向的平均毫秒；含Python校验，不含数据读取/AP。
        此短测量不能直接作为论文FPS结果。
    """
    with torch.inference_mode():
        for _ in range(2):
            model(images, conditions=conditions)
        torch.cuda.synchronize(images.device)
        start = time.perf_counter()
        for _ in range(5):
            model(images, conditions=conditions)
        torch.cuda.synchronize(images.device)
    return (time.perf_counter() - start) * 1000.0 / 5


def run_checks(config: Path, batch_size: int, device: torch.device) -> Dict[str, Any]:
    """验证完整原型的B兼容性与真实单批次前后向。

    Args:
        config: 新原型配置；默认仅新增微校准。
        batch_size: 真实训练接口检查batch，默认8。
        device: CUDA设备。

    Returns:
        配方、共享初始化、旧EMA、显式条件、损失/梯度/显存和计算量摘要。
    """
    from engine.core import YAMLConfig
    from engine.data.condition_utils import get_required_condition_fields, stack_batch_conditions
    from engine.misc.profiler_utils import stats
    from tools.check_module1 import move_to_device

    result: Dict[str, Any] = {"sampling": check_sampling(device)}
    base_cfg, cfg = YAMLConfig(str(BASE_CONFIG)), YAMLConfig(str(config))
    result["recipe"] = check_recipe(base_cfg, cfg)
    torch.manual_seed(0)
    base_model = base_cfg.model
    torch.manual_seed(0)
    model = cfg.model
    base_state, aligned_state = base_model.state_dict(), model.state_dict()
    assert set(base_state) == {name for name in aligned_state if not name.startswith(ALIGN_PREFIX)}
    assert all(torch.equal(value, aligned_state[name]) for name, value in base_state.items())
    initial_state = {name: value.detach().cpu().clone() for name, value in aligned_state.items()}
    initial_parameter_names = tuple(dict(model.named_parameters()))
    roles_base, roles_align = parameter_roles(base_cfg.optimizer), parameter_roles(cfg.optimizer)
    assert all(roles_align[name] == role for name, role in roles_base.items())
    for name, role in roles_align.items():
        if name.startswith(ALIGN_PREFIX):
            assert role == (1e-4, 0.0 if name.endswith("bias") else 1e-4)
    grouped = [id(p) for group in cfg.optimizer.param_groups for p in group["params"]]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(p) for p in model.parameters() if p.requires_grad}
    fields = get_required_condition_fields(model)
    base_fields = get_required_condition_fields(base_model)
    assert fields == ("weather", "illumination", "location", "height", "angle")
    assert base_fields == ("weather", "illumination", "location")
    for split in ("train_dataloader", "val_dataloader"):
        cfg.yaml_cfg[split]["num_workers"] = 0  # 检查只取一批；正式配置仍为4。
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = 2
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = batch_size
    images, targets = next(iter(cfg.val_dataloader))
    images, targets = move_to_device(images, device), move_to_device(targets, device)
    conditions = stack_batch_conditions(targets, device, fields=fields)
    conditions_base = stack_batch_conditions(targets, device, fields=base_fields)
    model, base_model = model.to(device).eval(), base_model.to(device).eval()
    diagnostic_maps: Dict[str, Any] = {}
    handles = []
    def capture_stage(name: str) -> Callable[..., None]:
        """暂存诊断特征，定位零初态差异，不修改模型输出。

        Args:
            name: 暂存特征的诊断标签。

        Returns:
            只读forward hook。
        """
        def capture(_module: torch.nn.Module, _args: Any, output: Any) -> None:
            """记录本次检查的输出副本。

            Args:
                _module: 被观察的组件。
                _args: 原调用参数。
                output: 三尺度特征列表。
            """
            diagnostic_maps[name] = [item.detach().clone() for item in output]
        return capture
    for label, candidate in (("B", base_model), ("align", model)):
        for component in ("backbone", "encoder"):
            handles.append(getattr(candidate, component).register_forward_hook(capture_stage(label + component)))
    identity_errors = []
    for layer in model.backbone.rgb_ir_alignments:
        handles.append(layer.register_forward_hook(lambda _module, args, output: identity_errors.append(float((output - args[0]).abs().max()))))
    with torch.inference_mode():
        prediction_base = base_model(images, conditions=conditions_base)
        prediction = model(images, conditions=conditions)
    for handle in handles:
        handle.remove()
    initial_diagnostics = {
        "alignment_identity_errors": identity_errors,
        "backbone_initial_max_diffs": [float((x - y).abs().max()) for x, y in zip(diagnostic_maps["Bbackbone"], diagnostic_maps["alignbackbone"])],
        "encoder_initial_max_diffs": [float((x - y).abs().max()) for x, y in zip(diagnostic_maps["Bencoder"], diagnostic_maps["alignencoder"])],
    }
    assert all(value == 0 for value in initial_diagnostics["alignment_identity_errors"])
    assert all(value == 0 for value in initial_diagnostics["backbone_initial_max_diffs"])
    assert all(value == 0 for value in initial_diagnostics["encoder_initial_max_diffs"])
    print(initial_diagnostics, flush=True)
    for key in ("pred_boxes", "pred_logits"):
        torch.testing.assert_close(prediction[key], prediction_base[key], rtol=0, atol=0)
    del prediction_base, prediction, base_state, aligned_state, diagnostic_maps, handles
    # 释放诊断副本后再重复普通前向，不能靠hook改变分配时序而掩盖差异。
    with torch.inference_mode():
        for _ in range(3):
            ordinary_base = base_model(images, conditions=conditions_base)
            ordinary_aligned = model(images, conditions=conditions)
            for key in ("pred_boxes", "pred_logits"):
                torch.testing.assert_close(ordinary_aligned[key], ordinary_base[key], rtol=0, atol=0)
    del ordinary_base, ordinary_aligned

    # B旧EMA仍严格加载；新模型仅允许明确列出的新增微校准参数缺失。
    saved = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=True)
    state = saved["ema"]["module"]
    base_model.load_state_dict(state, strict=True)
    incompatible = model.load_state_dict(state, strict=False)
    expected_missing = {name for name in model.state_dict() if name.startswith(ALIGN_PREFIX)}
    assert set(incompatible.missing_keys) == expected_missing and not incompatible.unexpected_keys
    with torch.inference_mode():
        old_prediction = base_model(images, conditions=conditions_base)
        aligned_prediction = model(images, conditions=conditions)
        for key in ("pred_boxes", "pred_logits"):
            torch.testing.assert_close(aligned_prediction[key], old_prediction[key], rtol=0, atol=0)
        processed = cfg.postprocessor.to(device)(aligned_prediction, torch.stack([t["orig_size"] for t in targets]))
        assert len(processed) == 2
        assert all(torch.isfinite(item[key]).all() for item in processed for key in ("boxes", "scores"))
        expect_value_error(lambda: model(images, conditions=conditions_base))
    one_conditions = {name: value[:1] for name, value in conditions.items()}
    one_base = {name: value[:1] for name, value in conditions_base.items()}
    latency = {"B_ms": round(forward_latency_ms(base_model, images[:1], one_base), 3),
               "B_align_ms": round(forward_latency_ms(model, images[:1], one_conditions), 3)}
    result["compatibility"] = {"shared_initial_state_exact": True, "zero_init_predictions_exact": True,
                               "initial_stage_diagnostics": initial_diagnostics,
                               "unhooked_prediction_equal_repeats": 3,
                               "old_B_EMA_strict_load": True, "old_EMA_aligned_predictions_exact": True,
                               "new_missing_keys_whitelisted": sorted(expected_missing),
                               "shared_optimizer_roles_equal": True, "parameters_covered_once": True,
                               "missing_geometry_rejected": True, "condition_fields": list(fields)}
    result["latency_smoke_fp32_batch1"] = latency
    del base_model, base_cfg, saved, state, old_prediction, aligned_prediction, processed, images, targets
    gc.collect()
    torch.cuda.empty_cache()
    model.load_state_dict(initial_state, strict=True)  # 完整接口检查从ImageNet初态开始。
    params, profile_info = stats(cfg)
    extra_params = sum(p.numel() for p in model.backbone.rgb_ir_alignments.parameters())
    result["parameters"] = {"training_total": sum(p.numel() for p in model.parameters()),
                            "deploy_total": params, "alignment_added": extra_params,
                            "calflops_info": sorted(profile_info), "grid_sample_FLOPs_fully_counted": False}
    print("共享初始化、旧B兼容、条件白名单和profiler检查通过。", flush=True)

    model.train()
    criterion = cfg.criterion.to(device)
    images, targets = next(iter(cfg.train_dataloader))
    images, targets = move_to_device(images, device), move_to_device(targets, device)
    conditions = stack_batch_conditions(targets, device, fields=fields)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(images, targets=targets, conditions=conditions)
    with torch.autocast("cuda", enabled=False):
        losses = criterion(outputs, targets, epoch=0, step=0, global_step=0)
        total_loss = sum(losses.values())
    assert torch.isfinite(total_loss), {name: value.item() for name, value in losses.items()}
    total_loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    alignment_gradients = []
    for layer in model.backbone.rgb_ir_alignments:
        gradient = layer.offset_head.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        alignment_gradients.append(float(gradient.abs().sum()))
        hidden = layer.spatial_context[0].weight.grad
        assert hidden is not None and hidden.abs().sum() == 0
    current_state = model.state_dict()
    assert all(torch.equal(initial_state[name], current_state[name].detach().cpu()) for name in initial_parameter_names)
    ema = cfg.ema
    fixed = model.backbone.condition_encoder.ema_copy_buffers
    originals = {name: getattr(model.backbone.condition_encoder, name).clone() for name in fixed}
    for _ in range(3):
        ema.update(model)
    assert all(torch.equal(value, getattr(ema.module.backbone.condition_encoder, name)) for name, value in originals.items())
    torch.cuda.synchronize(device)
    result["real_batch"] = {"input": list(images.shape), "logits": list(outputs["pred_logits"].shape),
                            "boxes": list(outputs["pred_boxes"].shape), "loss": float(total_loss),
                            "finite_gradient_tensors": len(gradients), "offset_head_gradient_L1_by_stage": alignment_gradients,
                            "hidden_zero_gradient_initial_expected": True, "all_parameters_unupdated": True,
                            "EMA_fixed_CLIP_exact": True,
                            "peak_allocated_GiB": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
                            "full_AdamW_training_memory_measured": False}
    return result


def main() -> None:
    """运行限时CUDA验收，成功后可保存新报告。

    Raises:
        RuntimeError: GPU不可访问，不回退CPU。
        FileExistsError: 指定报告已存在，不覆盖旧结果。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ALIGN_CONFIG)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size必须为正。")
    if args.report is not None and args.report.exists():
        raise FileExistsError(f"报告已存在：{args.report}，请指定新路径。")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可访问；需要在可访问RTX3080的rio-detr环境验收。")
    signal.signal(signal.SIGALRM, timeout_error)
    signal.alarm(540)
    start = time.monotonic()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    result = run_checks(args.config.resolve(), args.batch_size, torch.device("cuda:0"))
    result["elapsed_seconds"] = round(time.monotonic() - start, 2)
    result["optimizer_steps"] = 0
    result["device"] = torch.cuda.get_device_name(0)
    signal.alarm(0)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
