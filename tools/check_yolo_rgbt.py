"""Bounded paired-YOLO GPU checks; no optimizer.step or weight downloads."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import importlib.util
import json
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _timeout(_signal: int, _frame: Any) -> None:
    """Abort after 540 seconds, reporting an incomplete check."""
    raise TimeoutError("The paired-YOLO check reached its 540-second limit.")


def _reject(action: Callable[[], Any]) -> None:
    """Require an invalid input to raise a clear error.

    Args:
        action: Zero-argument callable expected to fail.
    """
    try:
        action()
    except (ValueError, TypeError, FileNotFoundError):
        return
    raise AssertionError("Invalid input was silently accepted.")


def _original(filename: str) -> Any:
    """Load a self-contained old module for numerical comparison only.

    Args:
        filename: Trusted repository-relative source path.

    Returns:
        Module without importing the engine registration tree.
    """
    name = "_check_original_" + Path(filename).stem
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _memory_guard(device: torch.device, limit_gib: float, reserve_gib: float) -> Dict[str, Any]:
    """Use the smaller driver/CUDA free-memory report before allocating.

    Args:
        device: Explicit CUDA device.
        limit_gib: Per-process PyTorch allocator cap.
        reserve_gib: Minimum unallocated reserve for other work.

    Returns:
        Memory-budget provenance. Failure does not launch the large test.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GPU checks are not silently replaced by CPU checks.")
    if limit_gib <= 0 or reserve_gib < 0:
        raise ValueError("Invalid GPU memory budget.")
    torch.cuda.set_device(device)
    cuda_free, total = torch.cuda.mem_get_info(device)
    command = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    rows = result.stdout.strip().splitlines()
    index = device.index or 0
    driver_free = float(rows[index].strip()) * 1024 ** 2
    available = min(float(cuda_free), driver_free)
    if available < (limit_gib + reserve_gib) * 1024 ** 3:
        raise RuntimeError(f"Only {available / 1024 ** 3:.2f}GiB free; wait for other GPU jobs to finish.")
    torch.cuda.set_per_process_memory_fraction(min(limit_gib * 1024 ** 3 / total, 0.95), device)
    torch.cuda.reset_peak_memory_stats(device)
    return {"CUDA_free_GiB": cuda_free / 1024 ** 3, "driver_free_GiB": driver_free / 1024 ** 3,
            "allocator_limit_GiB": limit_gib, "reserve_GiB": reserve_gib}


def check_math(device: torch.device, config: Any) -> Dict[str, Any]:
    """Check source equivalence, pixel direction, zero identity and gradients.

    Args:
        device: CUDA device.
        config: Requested experiment.

    Returns:
        Known-answer module checks, independent of measured detection AP.
    """
    from yolo_rgbt.conditions import FrozenEnvironmentPrior, geometry_descriptor
    from yolo_rgbt.fusion import ConditionFusion, RGBIRMicroAlignment

    old_geometry = _original("engine/backbone/rgb_ir_micro_alignment.py")
    old_score = _original("engine/backbone/cg_dmf.py")
    conditions = {
        "weather": torch.tensor([0, 1], device=device),
        "illumination": torch.tensor([1, 6], device=device),
        "location": torch.tensor([2, 3], device=device),
        "height": torch.tensor([[0.0], [1.0]], device=device),
        "angle": torch.tensor([[1.0], [0.0]], device=device),
    }
    geometry, radius = geometry_descriptor(conditions, 2, device, "geometry")
    expected_geometry, expected_radius = old_geometry.build_alignment_geometry(conditions["height"], conditions["angle"])
    torch.testing.assert_close(geometry, expected_geometry, rtol=0, atol=0)
    torch.testing.assert_close(radius, expected_radius, rtol=0, atol=0)
    _reject(lambda: geometry_descriptor({}, 2, device, "geometry"))
    _reject(lambda: geometry_descriptor({**conditions, "height": conditions["height"] + 2}, 2, device, "geometry"))
    absent, fraction = geometry_descriptor({}, 2, device, "visual")
    assert torch.count_nonzero(absent) == 0 and torch.all(fraction == 1)
    prior = FrozenEnvironmentPrior(config.anchor_dir, config.texts_path).to(device)
    cache = prior.joint_anchor.clone()
    prior.half().float()
    torch.testing.assert_close(prior.joint_anchor, cache, rtol=0, atol=0)
    vector = prior(conditions)
    torch.testing.assert_close(vector, prior({**conditions, "height": 1 - conditions["height"]}), rtol=0, atol=0)
    _reject(lambda: prior({**conditions, "location": torch.tensor([11, 11], device=device)}))

    rgb = torch.randn(2, 32, 6, 13, device=device)
    ir = torch.randn_like(rgb)
    ir_copy = ir.clone()
    for stride in (8, 16, 32):
        layer = RGBIRMicroAlignment(32, stride, 12.0).to(device)
        torch.testing.assert_close(layer(rgb, ir, geometry, radius), rgb, rtol=0, atol=0)
        with torch.autocast("cuda", dtype=torch.float16):
            half = rgb.half()
            torch.testing.assert_close(layer(half, ir.half(), geometry, radius), half, rtol=0, atol=0)
        with torch.no_grad():
            layer.offset_head.bias.copy_(layer.offset_head.bias.new_tensor((80.0, -80.0)))
        offsets = layer.compute_offsets(rgb, ir, geometry, radius)
        bound = 12.0 * radius[:, :, None, None].expand(-1, -1, 6, 13)
        torch.testing.assert_close(offsets[:, :1], bound, rtol=0, atol=0)
        torch.testing.assert_close(offsets[:, 1:], -bound, rtol=0, atol=0)
        with torch.no_grad():
            layer.offset_head.bias.copy_(layer.offset_head.bias.new_tensor((np.arctanh(0.5), 0.0)))
        ramp = torch.arange(13, device=device, dtype=torch.float32)[None, None, None].expand(2, 32, 6, -1)
        expected = ramp + (6.0 * radius / stride)[:, :, None, None]
        torch.testing.assert_close(layer(ramp, ir, geometry, radius)[..., 1:-2],
                                   expected[..., 1:-2], rtol=0, atol=5e-6)
    torch.testing.assert_close(ir, ir_copy, rtol=0, atol=0)
    layer = RGBIRMicroAlignment(32, 8, 12.0).to(device)
    (layer(rgb, ir, geometry, radius) - torch.randn_like(rgb)).square().mean().backward()
    assert torch.isfinite(layer.offset_head.weight.grad).all() and layer.offset_head.weight.grad.abs().sum() > 0
    assert layer.spatial_context[0].weight.grad.abs().sum() == 0

    block = ConditionFusion(32, 8, 12.0, alignment="off").to(device)
    original = old_score.CGDMF(32, condition_dim=512).to(device)
    original.rgb_score.load_state_dict(block.rgb_score.state_dict(), strict=True)
    original.ir_score.load_state_dict(block.ir_score.state_dict(), strict=True)
    original.out_projection.load_state_dict(block.out_projection.state_dict(), strict=True)
    sr, si = block.rgb_work(rgb), block.ir_work(ir)
    torch.testing.assert_close(block(rgb, ir, vector, geometry, radius), original(sr, si, vector), rtol=0, atol=0)
    wr, wi = block.compute_weights(sr, si, vector)
    assert torch.all(wr == 0.5) and torch.all(wi == 0.5)
    return {"old_geometry_descriptor_exact": True, "old_CGDMF_path_exact_FP32": True,
            "FP32_FP16_zero_shift_exact_identity": True, "input_pixel_bounds_and_direction": True,
            "IR_unmodified": True, "first_offset_head_gradient": True,
            "zero_hidden_first_gradient_expected": True, "raw_CLIP_preserved_under_half": True,
            "environment_ignores_geometry": True, "missing_or_invalid_metadata_rejected": True}


def check_metric_fixtures(config: Any, device: torch.device, directory: Path) -> Dict[str, Any]:
    """Check border-preserving native validation and a known geometric IoU.

    Args:
        config: Experiment settings.
        device: CUDA device.
        directory: Temporary diagnostic output directory.

    Returns:
        Exact geometry AP50=1/AP75=0/AP50_95=0.3 for IoU=0.6.
    """
    from yolo_rgbt.data import CLASSES
    from yolo_rgbt.evaluation import evaluate_payload
    from yolo_rgbt.validator import PairedOBBValidator
    from ultralytics.cfg import get_cfg

    class FixtureDataset:
        image_ids = ["fixture"]
        audit = {"annotations_sha256": "fixture", "sample_ids_sha256": "fixture",
                 "split": "test", "input_size": [512, 640]}

        def get_ann_info(self, index: int) -> Dict[str, np.ndarray]:
            """Return the fixture's forty-by-twenty-pixel GT rectangle."""
            return {"bboxes": np.array([[50, 50, 40, 20, 0]], dtype=np.float32),
                    "labels": np.array([0], dtype=np.int64)}

    dataset = FixtureDataset()
    payload = {
        "schema_version": 1,
        "metadata": {"reference": "IR", "box_format": "cx,cy,w,h,angle_radians",
                     "class_names": list(CLASSES), "dataset_audit": dataset.audit},
        "predictions": [{"image_id": "fixture", "boxes": [[60, 50, 40, 20, 0]],
                         "scores": [0.9], "labels": [0]}],
    }
    metrics = evaluate_payload(dataset, payload)["metrics"]
    assert abs(metrics["AP50"] - 1) < 1e-12 and metrics["AP75"] == 0
    assert abs(metrics["AP50_95"] - 0.3) < 1e-12
    validator = PairedOBBValidator(config, None, directory, get_cfg(overrides=config.native_args("0")))
    validator.device = device
    validator.data = {"val": "fixture"}
    stub = torch.nn.Identity()
    stub.names = dict(enumerate(CLASSES))
    validator.init_metrics(stub)
    raw = {"img": torch.zeros(1, 6, 512, 640, device=device),
           "batch_idx": torch.tensor([0], device=device),
           "cls": torch.tensor([[0.0]], device=device),
           "bboxes": torch.tensor([[641/640, 100/512, 40/640, 20/512, 0]], device=device),
           "ori_shape": [(512, 640)], "ratio_pad": [((1, 1), (0, 0))], "im_file": ["fixture.jpg"]}
    pred = torch.tensor([[641, 100, 40, 20, 0.9, 0, 0]], device=device)
    prepared = validator._prepare_batch(0, raw)
    assert prepared["bbox"][0, 0] == 641
    assert validator._prepare_pred(pred, prepared)[0, 0] == 641
    validator.update_metrics([pred], raw)
    assert validator.get_stats()["metrics/mAP50(B)"] > 0.9
    validator.init_metrics(stub)
    validator.update_metrics([torch.empty(0, 7, device=device)], raw)
    assert validator.get_stats()["metrics/mAP50(B)"] == 0
    _reject(lambda: evaluate_payload(dataset, {**payload, "predictions": []}))
    return {**metrics, "border_GT_predictions_unclipped": True,
            "zero_TP_validation_does_not_reuse_previous_AP": True,
            "incomplete_prediction_cache_rejected": True}


def check_models(
    config: Any, dataset: Any, device: torch.device, batch_size: int,
    weights: Optional[Path], directory: Path,
) -> Dict[str, Any]:
    """Check native transfer, full-model gradient flow, EMA and trainer setup.

    Args:
        config: Full requested model.
        dataset: Real paired training split.
        device: CUDA device.
        batch_size: Explicit batch to verify.
        weights: Optional real local official pretrained checkpoint.
        directory: Temporary synthetic checkpoint/trainer output area.

    Returns:
        Structural and real-batch checks. Synthetic initialization is explicitly
        distinguished from verifying the user's actual pretrained checkpoint.
    """
    from yolo_rgbt.data import move_batch
    from yolo_rgbt.model import PairedYOLO11sOBB, load_local_checkpoint
    from yolo_rgbt.trainer import PairedOBBTrainer
    from ultralytics.nn.tasks import OBBModel
    from ultralytics.utils.torch_utils import ModelEMA

    if weights is not None:
        if not weights.is_file():
            raise FileNotFoundError(weights)
        artifact = torch.load(weights, map_location="cpu", weights_only=False)
        native = artifact.get("ema") or artifact.get("model")
        actual_pretrained = True
        source_path = weights
    else:
        native = OBBModel("yolo11s-obb.yaml", nc=15, verbose=False)
        source_path = directory / "UNTRAINED_native_structure_fixture.pt"
        torch.save({"model": native, "ema": None, "train_args": {"task": "obb"}}, source_path)
        actual_pretrained = False
    full_config = dataclasses.replace(config, pretrained=source_path, output_dir=directory / "trainer",
                                      batch=batch_size, val_batch=1, workers=0)
    torch.manual_seed(0)
    full = PairedYOLO11sOBB(full_config).load(native).to(device).eval()
    _reject(lambda: full.load(torch.nn.Identity()))
    assert full.model[0].conv.weight is not full.ir_backbone[0].conv.weight
    torch.testing.assert_close(full.model[0].conv.weight, full.ir_backbone[0].conv.weight, rtol=0, atol=0)
    off_config = dataclasses.replace(full_config, alignment="off")
    off = PairedYOLO11sOBB(off_config).to(device).eval()
    off.load_state_dict(full.state_dict(), strict=True)
    loader = DataLoader(Subset(dataset, list(range(batch_size))), batch_size=batch_size,
                        collate_fn=dataset.collate_fn, num_workers=0)
    batch = move_batch(next(iter(loader)), device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        expected = off.predict(batch["img"], conditions=batch["conditions"])[0]
        actual = full.predict(batch["img"], conditions=batch["conditions"])[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (batch_size, 16, 6720)
    del off, expected, actual
    gc.collect()
    torch.cuda.empty_cache()

    # A simple baseline must preserve the original single-stream graph when
    # both modalities are identical and the projection starts as their mean.
    concat_config = dataclasses.replace(full_config, variant="concat", alignment="off")
    concat = PairedYOLO11sOBB(concat_config).load(native, verbose=False).to(device).eval()
    single = OBBModel("yolo11s-obb.yaml", nc=11, verbose=False).to(device).eval()
    single.model.load_state_dict(concat.model.state_dict(), strict=True)
    image = batch["img"][:1, 3:]
    with torch.inference_mode():
        paired_output = concat.predict(torch.cat((image, image), 1))[0]
        single_output = single.predict(image)[0]
    torch.testing.assert_close(paired_output, single_output, rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        baseline_loss = concat.train().loss({key: value for key, value in batch.items()})[0]
    assert torch.isfinite(baseline_loss)
    del concat, single, paired_output, single_output
    gc.collect()
    torch.cuda.empty_cache()

    ema = ModelEMA(full)
    full.train()
    with torch.autocast("cuda", dtype=torch.float16):
        loss, components = full.loss(batch)
    loss.backward()
    gradients = {name: parameter.grad for name, parameter in full.named_parameters() if parameter.grad is not None}
    assert gradients and all(torch.isfinite(value).all() for value in gradients.values())
    finite_gradient_count = len(gradients)
    startup = {}
    for label, tokens in {"micro": ("offset_head",), "FiLM": ("rgb_film", "ir_film"),
                          "score_readout": ("visual_score.1", "condition_bias", ".key.")}.items():
        total = sum(float(value.detach().float().abs().sum()) for name, value in gradients.items()
                    if any(token in name for token in tokens))
        assert total > 0, f"No detection gradient for {label}."
        startup[label] = total

    # Known-answer rectangle-unit check: a 2.2px square must survive the 2px
    # loss threshold; an outside-center box stays; a 1.5px side is filtered.
    capture = {}
    def capture_targets(module: Any, arguments: Any) -> None:
        """Capture actual native assigner GT without modifying the assignment."""
        capture["boxes"] = arguments[4].detach().clone()
        capture["mask"] = arguments[5].detach().clone()
    handle = full.criterion.assigner.register_forward_pre_hook(capture_targets)
    fixture = {
        "img": batch["img"][:1], "conditions": {key: value[:1] for key, value in batch["conditions"].items()},
        "batch_idx": torch.zeros(3, device=device),
        "cls": torch.zeros(3, 1, device=device),
        "bboxes": torch.tensor([[50/640, 50/512, 2.2/640, 2.2/512, 0],
                                 [641/640, 100/512, 40/640, 20/512, 0],
                                 [50/640, 50/512, 1.5/640, 3/512, 0]], device=device),
    }
    with torch.no_grad():
        full.loss(fixture)
    handle.remove()
    assert capture["boxes"].shape[1] == 2 and capture["mask"].all()
    torch.testing.assert_close(capture["boxes"][0, 0, 2:4], torch.tensor([2.2, 2.2], device=device))
    assert capture["boxes"][0, 1, 0] == 641

    clip = full.environment_prior.joint_anchor.clone()
    for _ in range(3):
        ema.update(full)
    torch.testing.assert_close(ema.ema.environment_prior.joint_anchor, clip, rtol=0, atol=0)
    full.eval()
    path = directory / "paired_roundtrip.pt"
    torch.save({"ema": copy.deepcopy(ema.ema).half(), "model": None, "train_args": vars(full.args)}, path)
    restored = load_local_checkpoint(path, full_config)
    torch.testing.assert_close(restored.environment_prior.joint_anchor, clip.cpu(), rtol=0, atol=0)
    assert restored.environment_prior.joint_anchor.dtype == torch.float32
    transfer_report = full.pretrained_report
    del full, ema, restored, gradients, batch
    gc.collect()
    torch.cuda.empty_cache()

    # Exercise the actual native trainer setup without entering its train loop.
    trainer = PairedOBBTrainer(full_config, str(device.index or 0))
    trainer._setup_train(world_size=1)
    assert trainer.accumulate == 1 and trainer.amp
    assert round(trainer.args.warmup_epochs * len(trainer.train_loader)) == config.warmup_iterations
    names = [name for group in trainer.optimizer.param_groups for name in group["param_names"]]
    expected_names = [name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad]
    assert len(names) == len(set(names)) and set(names) == set(expected_names)
    optimizer_type = type(trainer.optimizer).__name__
    assert optimizer_type == config.optimizer
    for group in trainer.optimizer.param_groups:
        assert group["weight_decay"] == (0 if group["kind"] == "no_decay" else config.weight_decay)
        assert group["lr"] == (config.lr_pretrained if group["role"] == "pretrained" else config.lr_new)
        if optimizer_type == "SGD":
            assert group["momentum"] == config.momentum and group["nesterov"] is config.nesterov
    assert not trainer.optimizer.state  # no optimizer.step has occurred
    mini = DataLoader(Subset(trainer.test_loader.dataset, [0, 1]), batch_size=1,
                      collate_fn=dataset.collate_fn, num_workers=0)
    trainer.validator.dataloader = mini
    trainer.loss_items = torch.zeros(3, device=device)
    mini_stats = trainer.validator(trainer)
    assert trainer.validator.seen == 2 and not trainer.optimizer.state

    # Native save/resume/final_eval must handle the custom model class and the
    # frozen cache. These are temporary UNTRAINED artifacts, not experiments.
    trainer.epoch = 0
    trainer.fitness = mini_stats["fitness"]
    trainer.best_fitness = trainer.fitness
    trainer.metrics = {key: value for key, value in mini_stats.items() if key != "fitness"}
    trainer.save_metrics(trainer.metrics)
    trainer.save_model()
    saved = load_local_checkpoint(trainer.best, full_config)
    torch.testing.assert_close(saved.environment_prior.joint_anchor,
                               trainer.model.environment_prior.joint_anchor.cpu(), rtol=0, atol=0)
    resumed = PairedOBBTrainer(full_config, str(device.index or 0), resume=trainer.last)
    resumed._setup_train(world_size=1)
    assert resumed.start_epoch == 1 and not resumed.optimizer.state
    torch.testing.assert_close(resumed.model.environment_prior.joint_anchor,
                               trainer.model.environment_prior.joint_anchor, rtol=0, atol=0)
    trainer.final_eval()
    stripped = load_local_checkpoint(trainer.best, full_config)
    torch.testing.assert_close(stripped.environment_prior.joint_anchor,
                               trainer.model.environment_prior.joint_anchor.cpu(), rtol=0, atol=0)
    assert not trainer.optimizer.state and not resumed.optimizer.state
    return {
        "initialization": "real_local_pretrained" if actual_pretrained else "UNTRAINED_structure_fixture_only",
        "real_pretrained_verified": actual_pretrained,
        "transfer": transfer_report, "IR_RGB_separate_parameters": True,
        "concat_equal_inputs_preserve_native_graph": True, "baseline_native_loss_finite": True,
        "zero_alignment_full_predictions_exact": True,
        "input_shape": [batch_size, 6, 512, 640], "native_prediction_shape": [batch_size, 16, 6720],
        "real_batch_loss": float(loss.detach()), "finite_gradient_tensors": finite_gradient_count,
        "rectangular_loss_pixel_filter_and_border_GT_verified": True,
        "startup_gradient_L1": startup, "CLIP_EMA_checkpoint_FP32_exact": True,
        "native_trainer_optimizer_warmup_metadata_validator": True,
        "native_checkpoint_save_resume_final_eval_verified": True,
        "optimizer": optimizer_type,
        "optimizer_recipe": {
            "lr_pretrained": config.lr_pretrained,
            "lr_new": config.lr_new,
            "lr_final_ratio": config.lr_final_ratio,
            "momentum": config.momentum if optimizer_type == "SGD" else None,
            "warmup_momentum": config.warmup_momentum if optimizer_type == "SGD" else None,
            "nesterov": config.nesterov if optimizer_type == "SGD" else None,
            "warmup_iterations": config.warmup_iterations,
            "weight_decay": config.weight_decay,
            "groups": len(trainer.optimizer.param_groups),
        },
        "mini_validation_images": 2, "optimizer_steps": 0,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run a bounded audit/check and write a new success or failure report.

    Args:
        argv: Optional CLI argument sequence.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/yolo_rgbt/atrumod_yolo11s_condition_align.yml")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--memory-limit-gib", type=float, default=4.0)
    parser.add_argument("--reserve-memory-gib", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite an existing report: {args.report}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(540)
    started = time.monotonic()
    report: Dict[str, Any] = {"scope": "data audit only" if args.audit_only else "bounded GPU structure/integration checks"}
    try:
        from yolo_rgbt.config import load_config
        from yolo_rgbt.data import ATRUMODYOLODataset

        torch.set_num_threads(1)
        torch.manual_seed(0)
        config = load_config(args.config)
        if not args.audit_only and (
            config.variant != "condition" or config.alignment != "geometry" or not config.enable_film
            or not config.dynamic_fusion or not config.score_condition or config.condition_mode != "real"
        ):
            raise ValueError("Run the integrated check with atrumod_yolo11s_condition_align.yml.")
        train = ATRUMODYOLODataset(config.data_root, "train", config.input_size, config.min_gt_side_px)
        test = ATRUMODYOLODataset(config.data_root, "test", config.input_size, config.min_gt_side_px)
        report["datasets"] = {"train": train.audit, "test": test.audit}
        if not args.audit_only:
            device = torch.device(args.device)
            report["GPU_budget"] = _memory_guard(device, args.memory_limit_gib, args.reserve_memory_gib)
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            with tempfile.TemporaryDirectory(prefix="rio_yolo_rgbt_checks_") as name:
                directory = Path(name)
                report["module_checks"] = check_math(device, config)
                report["metric_fixtures"] = check_metric_fixtures(config, device, directory / "validator")
                weights = args.weights or (config.pretrained if config.pretrained.is_file() else None)
                report["integration"] = check_models(config, train, device, args.batch_size, weights, directory)
            report["peak_allocated_GiB"] = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        signal.alarm(0)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"status": report.get("status"), "seconds": report["seconds"], "report": str(args.report)}))


if __name__ == "__main__":
    main()
