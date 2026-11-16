"""第二原型GPU验收；默认batch1/显存限额，保护正在训练的微校准组。

检查公式、交错通道、打分复用、零态/梯度、母体权重与真实前后向。
不执行optimizer.step、不整集评测、不下载权重；总时限540秒。
"""

import argparse
import copy
import gc
import hashlib
import json
import math
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
CONFIG_DIR = REPO_ROOT / "configs/rtdetrv2_obb"
BASE_CONFIG = CONFIG_DIR / "rtdetrv2_obb_hgnetv2_s_atrumod_joint_clip512_rgb_ir_micro_align.yml"
NEW_CONFIG = CONFIG_DIR / "rtdetrv2_obb_hgnetv2_s_atrumod_joint_clip512_align_gain_comp.yml"
BASE_CHECKPOINT = REPO_ROOT / "outputs/rtdetrv2_obb_hgnetv2_s_atrumod_joint_clip512_rgb_ir_micro_align/best_stg1.pth"
ACTIVE_BACKBONE = "DualHGNetv2_PG_GainComp"


def timeout_error(_signum: int, _frame: Any) -> None:
    """超时直接失败，未完成的检查不记录为通过。

    Args:
        _signum: POSIX告警信号。
        _frame: 当前执行帧。
    """
    raise TimeoutError("联合增益/混洗互补检查达到540秒，未完成，不自动续跑。")


def expect_value_error(action: Callable[[], Any]) -> None:
    """确认非法输入确实被拒绝。

    Args:
        action: 预期报ValueError的零参数调用。
    """
    try:
        action()
    except ValueError:
        return
    raise AssertionError("非法输入被静默接受。")


def is_extra_key(name: str) -> bool:
    """识别本轮唯二新增的模型分支键。

    Args:
        name: 完整state_dict/参数名称。

    Returns:
        是否属于三尺度融合的joint_gain/complement分支。
    """
    return name.startswith("backbone.fusion_layers.") and (
        ".joint_gain." in name or ".complement." in name
    )


def check_units(device: torch.device) -> Dict[str, Any]:
    """验证增益含义、交错顺序、零态、独立开关与两阶段梯度。

    Args:
        device: CUDA设备。

    Returns:
        小张量已知答案、分数复用与学习通路检查摘要。
    """
    from engine.backbone.cg_dmf import CGDMF
    from engine.backbone.gain_complement_fusion import GainComplementFusion, JointGainGate, PairedShuffleComplement

    gate = JointGainGate().to(device)
    scores_r = torch.tensor([[[[2.0, -1.0], [0.5, 3.0]]]], device=device)
    scores_i = torch.tensor([[[[0.0, -2.0], [1.5, 1.0]]]], device=device)
    torch.testing.assert_close(gate(scores_r, scores_i), torch.ones_like(scores_r), rtol=0, atol=0)
    gate(scores_r, scores_i).sum().backward()
    assert gate.coefficients.grad is not None and (gate.coefficients.grad.abs() > 0).all()
    with torch.no_grad():
        gate.coefficients.copy_(gate.coefficients.new_tensor((0.25, -0.1, 0.3)))
    expected = 2 * torch.sigmoid(0.25 * torch.maximum(scores_r, scores_i) - 0.1 * (scores_r + scores_i) + 0.3)
    torch.testing.assert_close(gate(scores_r, scores_i), expected, rtol=0, atol=0)
    expect_value_error(lambda: gate(scores_r * float("nan"), scores_i))
    expect_value_error(lambda: gate(scores_r[:, 0], scores_i))

    channels = 8
    rgb = torch.randn(2, channels, 6, 7, device=device)
    ir = torch.randn_like(rgb)
    condition = torch.randn(2, 16, device=device)
    complement = PairedShuffleComplement(channels, 2).to(device)
    marker_r = torch.arange(channels, device=device, dtype=torch.float32)[None, :, None, None].expand(1, -1, 3, 4)
    marker_i = marker_r + 100
    shuffled = complement.interleave(marker_r, marker_i)
    torch.testing.assert_close(shuffled[:, 0::2], marker_r, rtol=0, atol=0)
    torch.testing.assert_close(shuffled[:, 1::2], marker_i, rtol=0, atol=0)
    # 成对分组的已知答案：第0输出只读取R1右邻域与I1当前位置，不能读R2/I2。
    pair_fixture = PairedShuffleComplement(2, 2).to(device)
    ramp_r = torch.arange(5, device=device, dtype=torch.float32)[None, None, None, :].expand(1, 2, 3, -1).clone()
    ramp_i = torch.full_like(ramp_r, 10.0)
    with torch.no_grad():
        pair_fixture.pair_conv.weight.zero_()
        pair_fixture.pair_conv.weight[0, 0, 1, 2] = 1.0
        pair_fixture.pair_conv.weight[0, 1, 1, 1] = 2.0
    paired_output = pair_fixture.pair_conv(pair_fixture.interleave(ramp_r, ramp_i))
    torch.testing.assert_close(paired_output[:, 0, :, 1:-1], ramp_r[:, 0, :, 2:] + 20.0, rtol=0, atol=0)
    ramp_r[:, 1].add_(1000.0)
    ramp_i[:, 1].add_(2000.0)
    changed_output = pair_fixture.pair_conv(pair_fixture.interleave(ramp_r, ramp_i))
    torch.testing.assert_close(changed_output[:, 0], paired_output[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(complement(rgb, ir), torch.zeros_like(rgb), rtol=0, atol=0)
    complement(rgb, ir).mul(torch.randn_like(rgb)).mean().backward()
    assert complement.out_projection.weight.grad is not None and complement.out_projection.weight.grad.abs().sum() > 0
    assert complement.pair_conv.weight.grad is not None and complement.pair_conv.weight.grad.abs().sum() == 0
    complement.zero_grad(set_to_none=True)
    # 只扰动合成夹具的出口，不更新正式模型；确认零出口学习后前序卷积也可学。
    with torch.no_grad():
        complement.out_projection.weight.normal_(0, 0.01)
    complement(rgb, ir).mul(torch.randn_like(rgb)).mean().backward()
    assert complement.pair_conv.weight.grad is not None and complement.pair_conv.weight.grad.abs().sum() > 0

    mother = CGDMF(channels, condition_dim=16, query_dim=4, norm_groups=2)
    variants = {}
    for use_gain, use_complement in ((False, False), (True, False), (False, True), (True, True)):
        before = torch.random.get_rng_state().clone()
        augmented = GainComplementFusion.from_base(mother, use_gain, use_complement)
        assert torch.equal(before, torch.random.get_rng_state())
        augmented = augmented.to(device)
        variants[f"gain_{use_gain}_comp_{use_complement}"] = augmented
    mother = mother.to(device)
    expected_output = mother(rgb, ir, condition)
    for augmented in variants.values():
        torch.testing.assert_close(augmented(rgb, ir, condition), expected_output, rtol=0, atol=0)
    counts = {"rgb": 0, "ir": 0}
    augmented = variants["gain_True_comp_True"]
    def count(name: str) -> Callable[..., None]:
        """建立只读评分调用计数hook。

        Args:
            name: RGB/IR评分标签。

        Returns:
            不修改输出的forward hook。
        """
        def record(_module: torch.nn.Module, _args: Any, _output: Any) -> None:
            """记录一次评分调用。

            Args:
                _module: 评分模块。
                _args: 原输入。
                _output: 原输出。
            """
            counts[name] += 1
        return record
    handles = [augmented.rgb_score.register_forward_hook(count("rgb")),
               augmented.ir_score.register_forward_hook(count("ir"))]
    augmented(rgb, ir, condition)
    for handle in handles:
        handle.remove()
    assert counts == {"rgb": 1, "ir": 1}
    with torch.autocast("cuda", dtype=torch.float16):
        half_output = augmented(rgb.half(), ir.half(), condition)
    assert torch.isfinite(half_output).all()
    return {"gain_formula_known_answer": True, "gain_all_three_gradients_with_nonzero_scores": True,
            "shuffle_order_R1_I1_R2_I2": True, "complement_zero_initial": True,
            "pair_group_and_spatial_neighbor_known_answer": True,
            "complement_exit_first_gradient_nonzero": True, "hidden_gradient_after_fixture_perturbation_nonzero": True,
            "all_four_switch_combinations_initial_equal": True, "factory_preserves_global_CPU_rng": True,
            "score_calls_per_forward": counts, "AMP_finite": True, "invalid_score_rejected": True}


def check_recipe(base: Any, combined: Any) -> Dict[str, Any]:
    """比较生效骨干选项与其他完整展开配置，阻止母体12像素设置被改回8。

    Args:
        base: 当前正在训练的微校准YAMLConfig。
        combined: 新组合原型YAMLConfig。

    Returns:
        同配方、同条件/微校准和新增开关摘要。
    """
    import inspect
    from engine.backbone.dual_hgnetv2_pg import DualHGNetv2_PG

    left, right = copy.deepcopy(base.yaml_cfg), copy.deepcopy(combined.yaml_cfg)
    added = right.pop(ACTIVE_BACKBONE)
    assert added.pop("enable_joint_gain") is True and added.pop("enable_complement") is True
    defaults = {name: p.default for name, p in inspect.signature(DualHGNetv2_PG.__init__).parameters.items()
                if name != "self"}
    assert set(added).issubset(defaults)
    old_options = {**defaults, **left["DualHGNetv2_PG"]}
    new_options = {**defaults, **added}
    assert old_options == new_options, "新生效骨干的公共选项与当前母体不一致。"
    assert right["RTv4_PG"]["backbone"] == ACTIVE_BACKBONE
    right["RTv4_PG"]["backbone"] = left["RTv4_PG"]["backbone"]
    for cfg in (left, right):
        cfg.pop("__include__", None)
        cfg.pop("output_dir")
    assert left == right, "新组改变了母体的其他训练/数据/损失设置。"
    return {"all_other_expanded_config_equal": True, "effective_backbone_parent_options_equal": True,
            "alignment_max_shift_px": new_options["alignment_max_shift_px"], "epochs": combined.yaml_cfg["epoches"],
            "formal_train_batch": combined.yaml_cfg["train_dataloader"]["total_batch_size"],
            "optimizer": combined.yaml_cfg["optimizer"], "warmup_iter": combined.yaml_cfg["warmup_iter"],
            "flat_epoch": combined.yaml_cfg["flat_epoch"], "new_switches": ["joint_gain", "complement"]}


def optimizer_roles(optimizer: torch.optim.Optimizer) -> Dict[str, Tuple[float, float]]:
    """按参数名称读取学习率与衰减；不执行优化步。

    Args:
        optimizer: 已构造的原pg_module1 AdamW。

    Returns:
        参数名到(lr,weight_decay)的映射。
    """
    return {name: (group["lr"], group["weight_decay"])
            for group in optimizer.param_groups for name in group["param_names"]}


def snapshot_checkpoint(source: Path, target: Path) -> str:
    """只复制稳定的母体best权重，避免与正在保存的文件竞态。

    Args:
        source: 正在训练组的既有best checkpoint，只读。
        target: 本检查的/tmp副本。

    Returns:
        副本SHA256；加载副本而非正在写入的源文件。

    Raises:
        RuntimeError: 源文件在复制期间改变，不接受半份checkpoint。
    """
    before = source.stat()
    shutil.copyfile(source, target)
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("母体checkpoint复制期间改变；请待保存完成再运行检查。")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def query_available_memory(device: torch.device) -> Dict[str, int]:
    """取CUDA与驱动空余的较小值，避免共享GPU/WSL统计口径误判。

    Args:
        device: 当前单GPU环境的cuda:0。

    Returns:
        CUDA空余、驱动空余、保守空余与CUDA总显存，均为字节。

    Raises:
        RuntimeError: 驱动查询不可用、多物理GPU或设备名称不匹配。
            不静默退回可能低估其他进程占用的CUDA单一统计。
    """
    cuda_free, total = torch.cuda.mem_get_info(device)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(rows) != 1:
            raise ValueError("本限额检查面向当前单GPU环境，驱动返回的设备数不是1。")
        name, free_mib = (value.strip() for value in rows[0].rsplit(",", 1))
        if name != torch.cuda.get_device_name(device):
            raise ValueError("驱动与CUDA设备名称不匹配。")
        driver_free = int(free_mib) * 2**20
        if driver_free < 0:
            raise ValueError("驱动空余显存为负。")
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise RuntimeError("无法核验驱动显存占用，停止本检查以保护其他训练。") from error
    return {"cuda_free_bytes": cuda_free, "driver_free_bytes": driver_free,
            "effective_free_bytes": min(cuda_free, driver_free), "total_bytes": total}


def run_checks(config: Path, batch_size: int, device: torch.device) -> Dict[str, Any]:
    """核验新组合的公共初始化、旧权重、真实单批次检测与EMA。

    Args:
        config: 新组合配置路径。
        batch_size: 本次检查batch；默认1，不改变正式配置的batch8。
        device: CUDA设备，已设本进程显存限额。

    Returns:
        配方、兼容、梯度、显存和参数摘要；不含真实AP或训练结果。
    """
    from engine.core import YAMLConfig
    from engine.data.condition_utils import get_required_condition_fields, stack_batch_conditions
    from engine.misc.profiler_utils import stats
    from tools.check_module1 import move_to_device

    result: Dict[str, Any] = {"units": check_units(device)}
    gc.collect()
    torch.cuda.empty_cache()
    mother_cfg, cfg = YAMLConfig(str(BASE_CONFIG)), YAMLConfig(str(config))
    result["recipe"] = check_recipe(mother_cfg, cfg)
    torch.manual_seed(0)
    mother = mother_cfg.model
    torch.manual_seed(0)
    model = cfg.model
    mother_state, combined_state = mother.state_dict(), model.state_dict()
    assert set(mother_state) == {name for name in combined_state if not is_extra_key(name)}
    assert all(torch.equal(value, combined_state[name]) for name, value in mother_state.items())
    initial_state = {name: value.detach().cpu().clone() for name, value in combined_state.items()}
    parameter_names = tuple(dict(model.named_parameters()))
    base_roles, new_roles = optimizer_roles(mother_cfg.optimizer), optimizer_roles(cfg.optimizer)
    assert all(new_roles[name] == role for name, role in base_roles.items())
    for name, parameter in model.named_parameters():
        if is_extra_key(name):
            expected_decay = 0.0 if parameter.ndim <= 1 else 1e-4
            assert new_roles[name] == (1e-4, expected_decay)
    grouped_ids = [id(p) for group in cfg.optimizer.param_groups for p in group["params"]]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == {id(p) for p in model.parameters() if p.requires_grad}
    fields = get_required_condition_fields(model)
    assert fields == get_required_condition_fields(mother) == ("weather", "illumination", "location", "height", "angle")
    assert all(layer.max_shift_px == result["recipe"]["alignment_max_shift_px"] for layer in model.backbone.rgb_ir_alignments)
    for split in ("train_dataloader", "val_dataloader"):
        cfg.yaml_cfg[split]["num_workers"] = 0  # 本次仅取一批；正式仍为4。
        cfg.yaml_cfg[split]["total_batch_size"] = batch_size
    images, targets = next(iter(cfg.val_dataloader))
    images, targets = move_to_device(images, device), move_to_device(targets, device)
    conditions = stack_batch_conditions(targets, device, fields=fields)
    model, mother = model.to(device).eval(), mother.to(device).eval()
    with torch.inference_mode():
        for _ in range(2):
            base_output = mother(images, conditions=conditions)
            output = model(images, conditions=conditions)
            for name in ("pred_boxes", "pred_logits"):
                torch.testing.assert_close(output[name], base_output[name], rtol=0, atol=0)
    del mother_state, combined_state, base_output, output

    with tempfile.TemporaryDirectory(prefix="rio_gain_comp_check_") as temporary:
        snapshot = Path(temporary) / "parent_best.pth"
        checkpoint_sha256 = snapshot_checkpoint(BASE_CHECKPOINT, snapshot)
        saved = torch.load(snapshot, map_location="cpu", weights_only=True)
        state = saved["ema"]["module"]
        mother.load_state_dict(state, strict=True)
        loaded = model.load_state_dict(state, strict=False)
        expected_missing = {name for name in model.state_dict() if is_extra_key(name)}
        assert set(loaded.missing_keys) == expected_missing and not loaded.unexpected_keys
        with torch.inference_mode():
            base_output = mother(images, conditions=conditions)
            output = model(images, conditions=conditions)
            for name in ("pred_boxes", "pred_logits"):
                torch.testing.assert_close(output[name], base_output[name], rtol=0, atol=0)
            processed = cfg.postprocessor.to(device)(output, torch.stack([target["orig_size"] for target in targets]))
            assert len(processed) == batch_size
            assert all(torch.isfinite(item[key]).all() for item in processed for key in ("boxes", "scores"))
        checkpoint_epoch = saved.get("last_epoch")
        del saved, state
    result["compatibility"] = {"shared_initial_state_exact": True, "initial_predictions_equal_repeats": 2,
                               "mother_EMA_strict_load": True, "mother_EMA_zero_new_branches_prediction_equal": True,
                               "mother_checkpoint_sha256": checkpoint_sha256, "checkpoint_epoch_field": checkpoint_epoch,
                               "new_missing_keys_whitelisted": sorted(expected_missing),
                               "optimizer_common_roles_equal": True, "all_parameters_covered_once": True,
                               "condition_fields_unchanged": list(fields)}
    print("公式/通道顺序、公共初始化、母体EMA与配方兼容通过。", flush=True)
    del mother, mother_cfg, base_output, output, processed, images, targets
    gc.collect()
    torch.cuda.empty_cache()
    model.load_state_dict(initial_state, strict=True)
    deployed_params, profile_info = stats(cfg)
    training_params = sum(parameter.numel() for parameter in model.parameters())
    extra_params = sum(parameter.numel() for name, parameter in model.named_parameters() if is_extra_key(name))
    result["parameters"] = {"training_total": training_params, "deploy_total": deployed_params,
                            "gain_comp_added_vs_parent": extra_params, "calflops_info": sorted(profile_info),
                            "grid_sample_and_functional_gain_FLOPs_fully_counted": False}

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
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    complement_gradients, gain_gradients = [], []
    for layer in model.backbone.fusion_layers:
        exit_grad = layer.complement.out_projection.weight.grad
        assert exit_grad is not None and torch.isfinite(exit_grad).all() and exit_grad.abs().sum() > 0
        hidden_grad = layer.complement.pair_conv.weight.grad
        assert hidden_grad is not None and hidden_grad.abs().sum() == 0
        gain_grad = layer.joint_gain.coefficients.grad
        assert gain_grad is not None and torch.isfinite(gain_grad).all()
        # 原评分头初态全零，增益前两个系数的首步梯度也为零；不伪报已全部启动。
        torch.testing.assert_close(gain_grad[:2], torch.zeros_like(gain_grad[:2]), rtol=0, atol=0)
        complement_gradients.append(float(exit_grad.abs().sum()))
        gain_gradients.append(gain_grad.detach().cpu().tolist())
    current_state = model.state_dict()
    assert all(torch.equal(initial_state[name], current_state[name].detach().cpu()) for name in parameter_names)
    ema = cfg.ema
    fixed_names = model.backbone.condition_encoder.ema_copy_buffers
    originals = {name: getattr(model.backbone.condition_encoder, name).clone() for name in fixed_names}
    for _ in range(3):
        ema.update(model)
    assert all(torch.equal(value, getattr(ema.module.backbone.condition_encoder, name)) for name, value in originals.items())
    torch.cuda.synchronize(device)
    result["real_batch"] = {"checked_batch": batch_size, "input": list(images.shape),
                            "logits": list(outputs["pred_logits"].shape), "boxes": list(outputs["pred_boxes"].shape),
                            "loss": float(total_loss), "finite_gradient_tensors": len(gradients),
                            "complement_exit_gradient_L1_by_stage": complement_gradients,
                            "gain_coefficients_first_gradient_by_stage": gain_gradients,
                            "initial_zero_hidden_and_score_coeff_gradients_expected": True,
                            "all_parameters_unupdated": True, "EMA_fixed_CLIP_exact": True,
                            "peak_allocated_GiB": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
                            "full_batch8_AdamW_training_memory_measured": False}
    return result


def main() -> None:
    """先检查GPU空余与本进程限额，再运行限时验收。

    默认只检查batch1，正式batch8配置不变；闲置GPU可显式选择batch8并增大
    显存限额。空余不足时明确退出，不杀掉/暂停其他训练，也不回退CPU。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=NEW_CONFIG)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--memory-limit-gib", type=float, default=3.0)
    parser.add_argument("--reserve-memory-gib", type=float, default=3.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("检查batch必须为正。")
    if not all(math.isfinite(value) and value > 0 for value in (args.memory_limit_gib, args.reserve_memory_gib)):
        raise ValueError("显存限额与预留量必须为正有限GiB数。")
    if args.report is not None and args.report.exists():
        raise FileExistsError(f"报告已存在：{args.report}，不覆盖。")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可访问；请在可访问GPU的rio-detr环境检查，不回退CPU。")
    device = torch.device("cuda:0")
    memory = query_available_memory(device)
    free_bytes, total_bytes = memory["effective_free_bytes"], memory["total_bytes"]
    if free_bytes < (args.memory_limit_gib + args.reserve_memory_gib) * 2**30:
        raise RuntimeError("GPU空余不足以同时满足检查限额与训练预留；请待当前训练结束后复查。")
    if args.memory_limit_gib * 2**30 >= total_bytes:
        raise ValueError("检查显存限额必须小于设备总显存。")
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib * 2**30 / total_bytes, device)
    signal.signal(signal.SIGALRM, timeout_error)
    signal.alarm(540)
    start = time.monotonic()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    result = run_checks(args.config.resolve(), args.batch_size, device)
    result["elapsed_seconds"] = round(time.monotonic() - start, 2)
    result["optimizer_steps"] = 0
    result["device"] = torch.cuda.get_device_name(device)
    result["memory_budget"] = {"limit_GiB": args.memory_limit_gib, "reserve_GiB": args.reserve_memory_gib,
                               "free_before_GiB": round(free_bytes / 2**30, 3),
                               "driver_free_before_GiB": round(memory["driver_free_bytes"] / 2**30, 3),
                               "cuda_free_before_GiB": round(memory["cuda_free_bytes"] / 2**30, 3)}
    signal.alarm(0)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as target:
            json.dump(result, target, ensure_ascii=False, indent=2)
            target.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
