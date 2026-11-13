"""模块一GPU启动检查：真实图像、显式条件、推理、AMP损失与单次反向。

只检查计算图，不调用optimizer.step、不训练或下载模型；CUDA不可用即报错。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def move_to_device(value: Any, device: torch.device) -> Any:
    r"""递归移动真实样本张量，保留字典、dtype和框的数值。

    Args:
        value: 张量或嵌套样本结构。
        device: GPU设备。

    Returns:
        同结构、同数据类型的设备端对象。
    """
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def check_cg_dmf(device: torch.device) -> None:
    r"""用小GPU张量核验打分顺序、单位和及初态梯度。

    Args:
        device: 可用CUDA设备。
    """
    from engine.backbone.cg_dmf import CGDMF

    fusion = CGDMF(8, norm_groups=2, query_dim=4).to(device)
    rgb, ir = (torch.randn(2, 8, 4, 5, device=device) for _ in range(2))
    condition = torch.randn(2, 128, device=device)
    m_rgb, m_ir = fusion.compute_weights(rgb, ir, condition)
    torch.testing.assert_close(m_rgb, torch.full_like(m_rgb, 0.5))
    torch.testing.assert_close(m_rgb + m_ir, torch.ones_like(m_rgb))
    captured = []
    handle = fusion.ir_score.query.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
    with torch.no_grad():
        fusion.rgb_score.condition_bias.bias.fill_(2)
        fusion.ir_score.condition_bias.bias.fill_(-1)
    m_rgb, _ = fusion.compute_weights(rgb, ir, condition)
    handle.remove()
    torch.testing.assert_close(captured[0], torch.cat((ir, rgb), dim=1))
    torch.testing.assert_close(m_rgb, torch.full_like(m_rgb, torch.sigmoid(torch.tensor(3.)).item()))
    with torch.no_grad():
        fusion.rgb_score.condition_bias.bias.zero_()
        fusion.ir_score.condition_bias.bias.zero_()
    output = fusion(rgb, ir, condition)
    (output * torch.randn_like(output)).mean().backward()
    for layer in (fusion.rgb_score.key, fusion.rgb_score.visual_score[1]):
        assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum() > 0
    print("CG-DMF GPU检查：有序Query、初态0.5、权重和、条件评分与梯度通过。")


def check_module1(
    config: Path, batch_size: int, device: torch.device, profile: bool,
    anchor_dir: Optional[Path] = None, checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    r"""检查完整训练配置的一批真实输入，保持全部模型参数不更新。

    Args:
        config: 模块一YAML路径。
        batch_size: 本次检查的实际batch；默认与训练配置相同为8。
        device: CUDA设备。
        profile: 是否同时核验训练启动时的profiler。
        anchor_dir: 可选临时缓存路径，供F组限时检查；不修改YAML文件。
        checkpoint: 可选既有权重，仅用于兼容检查；不更新或覆盖该文件。

    Returns:
        设备、输入/输出形状、损失、梯度和峰值显存摘要。
    """
    from engine.core import YAMLConfig
    from engine.data.condition_utils import get_required_condition_fields, stack_batch_conditions
    from engine.misc.profiler_utils import stats
    from engine.optim.lr_scheduler import flat_cosine_schedule

    cfg = YAMLConfig(str(config))
    if anchor_dir is not None:
        cfg.yaml_cfg["DualHGNetv2_PG"]["anchor_dir"] = str(anchor_dir.resolve())
    for split in ("train_dataloader", "val_dataloader"):
        cfg.yaml_cfg[split]["num_workers"] = 0  # 只取一批，避免启动额外worker拖慢检查。
        cfg.yaml_cfg[split]["total_batch_size"] = batch_size
    assert cfg.yaml_cfg["DualHGNetv2_PG"]["pretrained"] is True
    model = cfg.model.to(device)
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = saved["ema"]["module"] if "ema" in saved else saved["model"]
        model.load_state_dict(state, strict=True)
        del saved, state
    fields = get_required_condition_fields(model)
    criterion = cfg.criterion.to(device)
    postprocessor = cfg.postprocessor.to(device)
    optimizer = cfg.optimizer  # 只核验分组；全文件不调用step。
    groups = optimizer.param_groups
    grouped_ids = [id(p) for group in groups for p in group["params"]]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == {id(p) for p in model.parameters() if p.requires_grad}
    for group in groups:
        base = group["initial_lr"]
        assert flat_cosine_schedule(10000, 500, 6000, 0, 0, base, base * 0.1) == 0
        assert flat_cosine_schedule(10000, 500, 6000, 0, 500, base, base * 0.1) == base
        assert flat_cosine_schedule(10000, 500, 6000, 0, 10000, base, base * 0.1) == base * 0.1

    if profile:
        params, model_info = stats(cfg)
        print(f"Profiler启动检查通过：{params:,} parameters，{model_info}")
    model.eval()
    val_images, val_targets = next(iter(cfg.val_dataloader))
    val_images, val_targets = move_to_device(val_images, device), move_to_device(val_targets, device)
    with torch.no_grad():
        # 仅条件进入推理模型；GT框/类别不会作为推理输入。
        prediction = model(val_images, conditions=stack_batch_conditions(val_targets, device, fields=fields))
        expected_logits = (batch_size, 300, 11)
        assert tuple(prediction["pred_logits"].shape) == expected_logits
        assert tuple(prediction["pred_boxes"].shape) == (batch_size, 300, 5)
        assert all(torch.isfinite(prediction[key]).all() for key in ("pred_logits", "pred_boxes"))
        sizes = torch.stack([target["orig_size"] for target in val_targets])
        processed = postprocessor(prediction, sizes)
        assert len(processed) == batch_size
        for item in processed:
            assert torch.isfinite(item["boxes"]).all() and torch.isfinite(item["scores"]).all()
        if model.backbone.condition_prior in ("frozen_clip_joint", "frozen_clip_six") or not model.backbone.enable_film:
            # F字符串输入和E未执行的FiLM不能让只读可视化依赖不存在的hook。
            from tools.visualize_module1 import run_forward
            one_condition = stack_batch_conditions(val_targets[:1], device, fields=fields)
            maps, _ = run_forward(model, val_images[:1], one_condition)
            assert maps["effective_condition"].shape == (model.backbone.condition_dim,)
            for stage in ("S3", "S4", "S5"):
                assert abs(float((maps[f"{stage}_rgb_weight"] + maps[f"{stage}_ir_weight"]).mean()) - 1) < 1e-6
            del maps
    del prediction, processed, val_images, val_targets

    model.train()
    images, targets = next(iter(cfg.train_dataloader))
    images, targets = move_to_device(images, device), move_to_device(targets, device)
    conditions = stack_batch_conditions(targets, device, fields=fields)
    # 抽取一份参数验证没有更新；检查仅backward，不执行优化或保存训练checkpoint。
    checked_parameter = next(model.backbone.backbone_rgb.parameters())
    original_weight = checked_parameter.detach().clone()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        outputs = model(images, targets=targets, conditions=conditions)
    with torch.autocast(device_type=device.type, enabled=False):
        losses = criterion(outputs, targets, epoch=0, step=0, global_step=0)
        loss = sum(losses.values())
    assert torch.isfinite(loss), {name: value.item() for name, value in losses.items()}
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    checked_layers = []
    if model.backbone.enable_exchange:
        checked_layers.append(model.backbone.exchange_layers[0].rgb_to_ir.difference_conv)
    else:
        assert model.backbone.exchange_layers[0].rgb_to_ir.difference_conv.weight.grad is None
    if model.backbone.dynamic_fusion:
        checked_layers.append(model.backbone.fusion_layers[0].rgb_score.visual_score[1])
    for layer in checked_layers:
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
    if not model.backbone.enable_film:
        for exchange in model.backbone.exchange_layers:
            assert exchange.rgb_film.weight.grad is None and exchange.ir_film.weight.grad is None
    # 冻结条件查表不能因EMA的浮点运算而漂移；旧DT-CPN未声明此契约。
    fixed_names = getattr(model.backbone.condition_encoder, "ema_copy_buffers", ())
    if fixed_names:
        ema = cfg.ema
        originals = {name: getattr(model.backbone.condition_encoder, name).clone() for name in fixed_names}
        for _ in range(12):
            ema.update(model)
        for name, original in originals.items():
            assert torch.equal(getattr(model.backbone.condition_encoder, name), original)
            assert torch.equal(getattr(ema.module.backbone.condition_encoder, name), original)
    assert torch.equal(original_weight, checked_parameter)
    torch.cuda.synchronize(device)
    return {
        "device": torch.cuda.get_device_name(device), "batch": batch_size,
        "input": list(images.shape), "logits": expected_logits,
        "loss": loss.item(), "finite_gradient_tensors": len(gradients),
        "peak_allocated_GiB": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
        "optimizer_steps": 0,
        "condition_fields": list(fields), "condition_prior": model.backbone.condition_prior,
        "enable_film": model.backbone.enable_film, "ema_fixed_buffers": list(fixed_names),
    }


def main() -> None:
    r"""运行CUDA检查，缺GPU时报错，不自动退回CPU。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_module1.yml")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--anchor-dir", type=Path, help="仅本次检查覆盖缓存路径，建议使用/tmp缓存")
    parser.add_argument("--checkpoint", type=Path, help="仅用于严格加载兼容性检查，不更新已有权重")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；请在可访问GPU的rio-detr环境运行本检查。")
    torch.manual_seed(0)
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    check_cg_dmf(device)
    result = check_module1(args.config, args.batch_size, device, args.profile, args.anchor_dir, args.checkpoint)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
