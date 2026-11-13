"""模块一已训练权重的GPU诊断：条件扰动、消息量、融合权重和检测梯度。

默认随机取64对验证图像；--samples 0 --evaluate评估全验证集。
这里的关分支/错条件是推理干预，不能代替重新训练的消融。
不建立optimizer、不更新参数、不下载权重；CUDA不可用即报错。
"""

import argparse
from contextlib import redirect_stdout
from functools import partial
import io
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Mapping, Tuple
import xml.etree.ElementTree as ET

import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from check_module1 import move_to_device
from engine.core import YAMLConfig
from engine.data.condition_utils import stack_batch_conditions


MODES = ("real", "shuffle", "zero", "no_exchange", "equal_fusion")


def capture(
    cache: Dict[str, Any], name: str, module: nn.Module,
    inputs: Tuple[Any, ...], output: Any,
) -> None:
    r"""通过只读hook保留当前前向的张量，不替换模块输出。

    Args:
        cache: 每次前向后清理的暂存字典，不跨batch保留计算图。
        name: 当前尺度/分支的名称。
        module: 被观察的模块。
        inputs: 模块真实输入；方向消息的受体S为inputs[1]，[B,C,H,W]。
        output: 融合特征[B,C,H,W]、分数[B,1,H,W]或(G,E)元组。
    """
    cache[name] = (inputs, output)


def relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    r"""返回逐图L2比值的batch均值，供变化量/注入量诊断使用。

    Args:
        value: 变化量或消息量，[B,C,H,W]。
        reference: 参照特征，[B,C,H,W]，与value同形状。

    Returns:
        mean_b(||value_b|| / ||reference_b||)；不是AP增益。
    """
    denominator = reference.float().flatten(1).norm(dim=1)
    if not torch.isfinite(denominator).all() or (denominator == 0).any():
        raise ValueError("参照特征L2为0或非有限，不能计算相对变化。")
    return (value.float().flatten(1).norm(dim=1) / denominator).mean().item()


def configure_mode(backbone: nn.Module, mode: str) -> None:
    r"""设置单因素干预；shuffle由调用方整体置换六条件元组。

    Args:
        backbone: DualHGNetv2_PG，带三个消融开关。
        mode: MODES中的一种；每次先恢复其他因素，避免干预混合。
    """
    if mode not in MODES:
        raise ValueError(f"未知诊断模式：{mode}")
    backbone.condition_mode = "zero" if mode == "zero" else "real"
    backbone.enable_exchange = mode != "no_exchange"
    backbone.dynamic_fusion = mode != "equal_fusion"


def collect_mechanism(cache: Mapping[str, Any]) -> List[Dict[str, float]]:
    r"""读取三个尺度的真实条件门控/消息/权重，均值只作数值诊断。

    Args:
        cache: 当前real前向的hook输出。

    Returns:
        三个尺度的统计；空间权重[B,1,H,W]，消息/门控[B,C,H,W]。
        空间统计含背景，不能解释为目标区域质量或物理可靠性。
    """
    records = []
    for index in range(3):
        rgb_score = cache[f"rgb_score_{index}"][1]
        ir_score = cache[f"ir_score_{index}"][1]
        weight = (rgb_score - ir_score).sigmoid().float()
        record = {
            "rgb_weight_mean": weight.mean().item(),
            "rgb_weight_std": weight.std(unbiased=False).item(),
            "weight_extreme_fraction": ((weight < 0.05) | (weight > 0.95)).float().mean().item(),
        }
        for direction in ("rgb_to_ir", "ir_to_rgb"):
            inputs, (gate, message) = cache[f"{direction}_{index}"]
            record[f"{direction}_gate_mean"] = gate.float().mean().item()
            record[f"{direction}_injection_to_S_l2"] = relative_l2(gate * message, inputs[1])
        records.append(record)
    return records


def gradient_probe(cfg: YAMLConfig, model: nn.Module, device: torch.device) -> Dict[str, Any]:
    r"""在一个训练batch计算原检测损失的梯度，不执行优化器更新。

    Args:
        cfg: 模块一配置；训练数据无随机增强。
        model: 已加载的检测模型；开始前恢复完整模块一。
        device: CUDA设备。

    Returns:
        检测损失和关键条件/消息参数的梯度L2；非零不等于对AP有益。
    """
    configure_mode(model.backbone, "real")
    model.train()
    # 小批量诊断保持已保存的BN统计，避免改动checkpoint的推理状态。
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._NormBase):
            module.eval()
    model.zero_grad(set_to_none=True)
    images, targets = next(iter(cfg.train_dataloader))
    images, targets = move_to_device(images, device), move_to_device(targets, device)
    conditions = stack_batch_conditions(targets, device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(images, targets=targets, conditions=conditions)
    with torch.autocast(device_type="cuda", enabled=False):
        losses = cfg.criterion.to(device)(outputs, targets, epoch=0, step=0, global_step=0)
        loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise ValueError("训练batch的检测损失含NaN/Inf。")
    loss.backward()
    prefixes = ["backbone.condition_encoder"]
    for index in range(3):
        prefixes.extend([
            f"backbone.exchange_layers.{index}.rgb_film",
            f"backbone.exchange_layers.{index}.ir_film",
            f"backbone.exchange_layers.{index}.rgb_to_ir.condition_bias",
            f"backbone.exchange_layers.{index}.ir_to_rgb.condition_bias",
            f"backbone.exchange_layers.{index}.rgb_to_ir.difference_conv",
            f"backbone.exchange_layers.{index}.ir_to_rgb.difference_conv",
            f"backbone.fusion_layers.{index}.rgb_score.key",
            f"backbone.fusion_layers.{index}.ir_score.key",
        ])
    norms = {prefix: torch.zeros((), device=device) for prefix in prefixes}
    finite_tensors = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise ValueError(f"{name}的检测梯度含NaN/Inf。")
        finite_tensors += 1
        for prefix in prefixes:
            if name.startswith(prefix + "."):
                norms[prefix] += parameter.grad.float().square().sum()
    result = {
        "train_batch": images.shape[0], "loss": loss.item(),
        "finite_gradient_tensors": finite_tensors,
        "gradient_l2": {key: value.sqrt().item() for key, value in norms.items()},
        "optimizer_steps": 0,
    }
    model.zero_grad(set_to_none=True)
    model.eval()
    return result


def diagnose(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    r"""对相同验证图像比较五种干预，并保存可复查的样本与权重来源。

    Args:
        args: CLI参数；samples=0表示全验证集，evaluate启用已有AP评估器。
        device: CUDA设备；模型前向、特征统计和反向均在该设备执行。

    Returns:
        诊断字典；--evaluate时另含VOC07 mAP@0.5，与现有DotaEvaluator一致。
    """
    started = time.monotonic()

    def check_time_budget() -> None:
        r"""超过本次短诊断预算时明确停止，不把部分评估保存为完整结果。"""
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("已超过GPU诊断时间预算；请由用户增加--max-seconds后自行运行。")

    cfg = YAMLConfig(str(args.config))
    if cfg.yaml_cfg.get("model") != "RTv4_PG":
        raise ValueError("此工具只用于RTv4_PG模块一配置。")
    for split in ("train_dataloader", "val_dataloader"):
        cfg.yaml_cfg[split]["num_workers"] = 0
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = args.batch_size
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = min(
        args.batch_size, cfg.yaml_cfg["train_dataloader"]["total_batch_size"])
    # checkpoint必须完整严格匹配；此处不用ImageNet重复初始化或联网加载。
    cfg.yaml_cfg["DualHGNetv2_PG"]["pretrained"] = False
    model = cfg.model.to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["ema"]["module"] if args.weights == "ema" else checkpoint["model"]
    model.load_state_dict(state, strict=True)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()), "weights": args.weights,
        "epoch": checkpoint["last_epoch"],
        "saved_optimizer_groups": [
            {key: group[key] for key in ("group_name", "initial_lr", "weight_decay") if key in group}
            | {"parameter_tensors": len(group["params"])}
            for group in checkpoint["optimizer"]["param_groups"]
        ],
    }
    del checkpoint, state
    dataset = cfg.val_dataloader.dataset
    total = len(dataset) if args.samples == 0 else args.samples
    if total < 2 or total > len(dataset):
        raise ValueError(f"samples需为0或2–{len(dataset)}，不会静默缩小样本数。")
    generator = torch.Generator().manual_seed(args.seed)
    # 全量评估保留训练验证的原始顺序；同分预测的排序可能影响VOC07 AP。
    indices = (list(range(len(dataset))) if total == len(dataset)
               else torch.randperm(len(dataset), generator=generator).tolist()[:total])
    # 整体置换六字段；不单独乱序属性，避免制造不存在的组合。
    parsed = [{"conditions": dataset._parse_conditions(
        ET.parse(Path(dataset.ann_folder) / f"{dataset.img_ids[index]}.xml").getroot(),
        str(Path(dataset.ann_folder) / f"{dataset.img_ids[index]}.xml"),
    )} for index in indices]
    bank = stack_batch_conditions(parsed, device)
    # 把随机排列连成一个环，保证无自配对；全量评估也不退化为邻帧交换。
    cycle = torch.randperm(total, generator=generator).to(device)
    donors = torch.empty_like(cycle)
    donors[cycle] = cycle.roll(1)
    shuffled_bank = {key: value[donors] for key, value in bank.items()}
    changed = torch.zeros(total, dtype=torch.bool, device=device)
    for key in bank:
        changed |= (bank[key] != shuffled_bank[key]).reshape(total, -1).any(dim=1)
    if not changed.any():
        raise ValueError("所选样本条件完全相同；整体置换没有干预作用，请增加samples或换seed。")
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=indices,
                        collate_fn=cfg.val_dataloader.collate_fn, num_workers=0)
    cache: Dict[str, Any] = {}
    handles = []
    for index in range(3):
        fusion, exchange = model.backbone.fusion_layers[index], model.backbone.exchange_layers[index]
        observed = [(f"feature_{index}", fusion.out_projection),
                    (f"rgb_score_{index}", fusion.rgb_score), (f"ir_score_{index}", fusion.ir_score),
                    (f"rgb_to_ir_{index}", exchange.rgb_to_ir), (f"ir_to_rgb_{index}", exchange.ir_to_rgb)]
        handles.extend(module.register_forward_hook(partial(capture, cache, name)) for name, module in observed)
    evaluators = {}
    if args.evaluate:
        from engine.data.dataset.dota_evaluator import DotaEvaluator
        evaluators = {mode: DotaEvaluator(dataset) for mode in MODES}
    comparison = {mode: {"Y_relative_l2": [0.0, 0.0, 0.0]} for mode in MODES}
    mechanism = [dict() for _ in range(3)]
    processed = 0
    print(f"GPU诊断：{args.weights} epoch={metadata['epoch']}，{total}对验证图像。", flush=True)
    try:
        with torch.no_grad():
            for images, targets in loader:
                check_time_budget()
                images, targets = move_to_device(images, device), move_to_device(targets, device)
                batch = images.shape[0]
                real = stack_batch_conditions(targets, device)
                shuffled = {key: value[processed:processed + batch] for key, value in shuffled_bank.items()}
                for key in real:
                    torch.testing.assert_close(real[key], bank[key][processed:processed + batch])
                reference = []
                for mode in MODES:
                    cache.clear()
                    configure_mode(model.backbone, mode)
                    # 使用与正常验证相同的FP32推理，避免把AMP差异混成消融影响。
                    outputs = model(images, conditions=shuffled if mode == "shuffle" else real)
                    if not all(torch.isfinite(outputs[key]).all() for key in ("pred_logits", "pred_boxes")):
                        raise ValueError(f"{mode}预测含NaN/Inf。")
                    features = [cache[f"feature_{index}"][1] for index in range(3)]
                    if mode == "real":
                        reference = features
                        for index, record in enumerate(collect_mechanism(cache)):
                            for key, value in record.items():
                                mechanism[index][key] = mechanism[index].get(key, 0.0) + value * batch / total
                    else:
                        for index in range(3):
                            comparison[mode]["Y_relative_l2"][index] += relative_l2(
                                features[index] - reference[index], reference[index]) * batch / total
                    if args.evaluate:
                        sizes = torch.stack([target["orig_size"] for target in targets])
                        predictions = cfg.postprocessor.to(device)(outputs, sizes)
                        evaluators[mode].update({target["image_id"].item(): prediction
                                                 for target, prediction in zip(targets, predictions)})
                cache.clear()
                processed += batch
                if processed == total or processed % 128 == 0:
                    print(f"已完成 {processed}/{total} 对图像的五种GPU前向。", flush=True)
        for mode, evaluator in evaluators.items():
            check_time_budget()
            with redirect_stdout(io.StringIO()):
                evaluator.summarize()
            comparison[mode]["mAP50_voc07"] = evaluator.stats["mAP"]
            print(f"{mode}: mAP@0.5={evaluator.stats['mAP']:.6f}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        cache.clear()
        configure_mode(model.backbone, "real")
    check_time_budget()
    gradients = gradient_probe(cfg, model, device)
    torch.cuda.synchronize(device)
    return metadata | {
        "device": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "sample_scope": "full_validation" if total == len(dataset) else "random_subset_diagnostic",
        "seed": args.seed, "samples": processed, "dataset_size": len(dataset),
        "eval_batch_size": args.batch_size,
        "sampling_order": "dataset_order" if total == len(dataset) else "seeded_random_subset",
        "shuffle_scheme": "seeded_single_cycle_of_whole_condition_tuples",
        "sample_ids": [dataset.img_ids[index] for index in indices],
        "shuffle_changed_tuple_fraction": changed.float().mean().item(),
        "comparison": comparison, "mechanism_S3_S4_S5": mechanism,
        "gradient_probe": gradients, "elapsed_seconds": round(time.monotonic() - started, 2),
        "interpretation": "推理干预/非零梯度只诊断模型依赖；正增益须通过同设置重新训练的消融确认。",
    }


def main() -> None:
    r"""解析参数，执行CUDA诊断，写入独立JSON；已有结果拒绝覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_module1.yml")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "outputs/rtdetrv2_obb_hgnetv2_s_atrumod_module1/best_stg1.pth")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=540.0)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/module1_diagnostics/summary.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"已有诊断结果：{args.output}；请用--output指定另一文件。")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；本工具不退回CPU。")
    if args.samples < 0 or args.batch_size < 1 or args.max_seconds <= 0:
        raise ValueError("samples不能为负，batch-size和max-seconds必须为正。")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    result = diagnose(args, torch.device("cuda:0"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("samples", "comparison", "elapsed_seconds")}, ensure_ascii=False))
    print(f"详细数值已保存：{args.output}")


if __name__ == "__main__":
    main()
