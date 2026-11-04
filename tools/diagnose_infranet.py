"""
tools/diagnose_infranet.py

InfraNet 深度科研机理透视分析仪 (支持 ATR-UMOD 嵌套字典与全尺度 CKA 矩阵).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any

from engine.core import YAMLConfig


# ==============================================================================
# 递归设备安全迁移函数 (完美支持多层嵌套字典与张量)
# ==============================================================================
def to_device(item: Any, device: torch.device) -> Any:
    """递归将嵌套字典、列表中的所有 Tensor 转移到目标设备."""
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, dict):
        return {k: to_device(v, device) for k, v in item.items()}
    elif isinstance(item, (list, tuple)):
        return [to_device(v, device) for v in item]
    return item


# ==============================================================================
# CKA (Centered Kernel Alignment) 线性无偏核心算法
# ==============================================================================
def centering(K: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    unit = torch.ones((n, n), device=K.device) / n
    return K - unit @ K - K @ unit + unit @ K @ unit


def linear_hsic(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    K = X @ X.T
    L = Y @ Y.T
    K_c = centering(K)
    L_c = centering(L)
    return (K_c * L_c).sum()


def compute_linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    r"""计算两组高维特征的 Linear CKA 相似度 [0, 1]."""
    X_flat = X.flatten(1).float()
    Y_flat = Y.flatten(1).float()

    hsic_xy = linear_hsic(X_flat, Y_flat)
    hsic_xx = linear_hsic(X_flat, X_flat)
    hsic_yy = linear_hsic(Y_flat, Y_flat)

    denom = torch.sqrt(hsic_xx * hsic_yy).clamp(min=1e-8)
    cka = (hsic_xy / denom).item()
    return float(np.clip(cka, 0.0, 1.0))


# ==============================================================================
# 核心诊断主函数
# ==============================================================================
def run_infranet_diagnosis(cfg_path: str, checkpoint_path: str):
    print("=" * 80)
    print("🔬 [InfraNet 深度科研机理透视分析仪] 启动...")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = YAMLConfig(cfg_path)
    
    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device)

    print(f"📦 正在载入待分析权重: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in state and "module" in state["ema"]:
        model_weights = state["ema"]["module"]
    elif "model" in state:
        model_weights = state["model"]
    else:
        model_weights = state
    
    clean_weights = {k.replace("module.", ""): v for k, v in model_weights.items()}
    model.load_state_dict(clean_weights, strict=False)
    print("✅ 权重载入完毕！\n")

    # 1. 抓取一个真实的数据 Batch 并安全迁移
    loader = cfg.val_dataloader
    for samples_raw, targets_raw in loader:
        # 防御性截取前 4 个样本，兼顾代表性与显存安全 (防 OOM)
        limit = min(4, samples_raw.shape[0])
        samples = samples_raw[:limit].to(device)
        targets = [to_device(t, device) for t in targets_raw[:limit]]
        break

    batch_size = samples.shape[0]
    print(f"🔍 捕获真实测试样本: Batch Size = {batch_size}, 形状 = {list(samples.shape)}")

    # ==========================================================================
    # 诊断 1: 审计 QualGate 内部参数与激活特征 (均值/方差/稀疏度/q值)
    # ==========================================================================
    print("\n" + "-" * 80)
    print("📊 [诊断 1]: QualGate 各融合节点激活统计与可靠性标量 q 审计")
    print("-" * 80)
    
    model.train()  # 开启双流模式
    with torch.no_grad():
        backbone_out = model.backbone(samples)
        feats_main = backbone_out["feats_main"]
        feats_aux = backbone_out["feats_aux"]
        q_dict = backbone_out["quality_scores"]

    for stage_idx in [3, 4, 5]:
        q_val = q_dict[f"q{stage_idx}"].cpu().numpy().flatten()
        fa = feats_aux[stage_idx - 3]
        zero_ratio = (fa.abs() < 1e-5).float().mean().item() * 100.0

        print(f"▶ Stage {stage_idx} 融合节点:")
        print(f"   ├─ 可靠性标量 q{stage_idx}: 均值 = {q_val.mean():.4f}, 最小值 = {q_val.min():.4f}, 最大值 = {q_val.max():.4f}, 方差 = {q_val.var():.6f}")
        print(f"   ├─ 输出特征 A{stage_idx}: 均值 = {fa.mean().item():.4f}, 方差 = {fa.var().item():.4f}")
        print(f"   └─ 0 元素比例 (Sparsity): {zero_ratio:.2f}%")

    # ==========================================================================
    # 诊断 2: CKA (Centered Kernel Alignment) 表征相似度对齐分析 (全三尺度)
    # ==========================================================================
    print("\n" + "-" * 80)
    print("🧬 [诊断 2]: CKA 多模态表征相似度矩阵 (检验是否发生模态坍缩)")
    print("-" * 80)

    x_ir = samples[:, :3, :, :]
    x_rgb = samples[:, 3:, :, :]
    with torch.no_grad():
        f_ir_list = model.backbone.backbone_ir(x_ir)
        # 单独跑一次原始 RGB 前向提取纯 RGB 特征
        x_rgb_stem = model.backbone.rgb_stem(x_rgb)
        x_rgb_s0 = model.backbone.rgb_stage0(x_rgb_stem)
        r3 = model.backbone.rgb_stage1(x_rgb_s0)
        r4 = model.backbone.rgb_stage2_cascade(r3)
        r5 = model.backbone.rgb_stage3_cascade(r4)
        f_rgb_list = [r3, r4, r5]

    print("┌" + "─" * 12 + "┬" + "─" * 22 + "┬" + "─" * 22 + "┬" + "─" * 22 + "┐")
    print(f"│ {'Stage':<10} │ {'CKA(IR, RGB)':<20} │ {'CKA(IR, Fused)':<20} │ {'CKA(RGB, Fused)':<20} │")
    print("├" + "─" * 12 + "┼" + "─" * 22 + "┼" + "─" * 22 + "┼" + "─" * 22 + "┤")

    for idx, s_name in enumerate(["Stage 3", "Stage 4", "Stage 5"]):
        cka_ir_rgb = compute_linear_cka(f_ir_list[idx], f_rgb_list[idx])
        cka_ir_fused = compute_linear_cka(f_ir_list[idx], feats_aux[idx])
        cka_rgb_fused = compute_linear_cka(f_rgb_list[idx], feats_aux[idx])
        print(f"│ {s_name:<10} │ {cka_ir_rgb:>18.4f} │ {cka_ir_fused:>18.4f} │ {cka_rgb_fused:>18.4f} │")

    print("└" + "─" * 12 + "┴" + "─" * 22 + "┴" + "─" * 22 + "┴" + "─" * 22 + "┘")
    print("   💡 学术解读准则:")
    print("      - 若 CKA(IR, Fused) 接近 1.0 且 CKA(RGB, Fused) 极低: 说明融合分支彻底抛弃了可见光，退化为纯红外！")
    print("      - 若 CKA(IR, RGB) > 0.85: 说明两路特征高度同质化，未能学到互补信息！")

    # ==========================================================================
    # 诊断 3: 主损失 vs 辅助损失 梯度余弦夹角冲突检验 (Gradient Conflict)
    # ==========================================================================
    print("\n" + "-" * 80)
    print("⚡ [诊断 3]: 主红外检测梯度 vs 辅助特权梯度 冲突检验 (Cosine Similarity)")
    print("-" * 80)

    model.zero_grad()
    criterion.zero_grad()

    outputs = model(samples, targets=targets)
    outputs_main = {k: v for k, v in outputs.items() if k not in ['aux_infranet', 'quality_scores']}
    outputs_aux = outputs['aux_infranet']

    # (a) 计算主损失梯度
    loss_main_dict = criterion._compute_single_branch_losses(outputs_main, targets)
    loss_main = sum(loss_main_dict.values())
    loss_main.backward(retain_graph=True)

    grads_main = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads_main[name] = param.grad.detach().clone()

    model.zero_grad()

    # (b) 计算辅助损失梯度
    loss_aux_dict = criterion._compute_single_branch_losses(outputs_aux, targets)
    loss_aux = sum(loss_aux_dict.values())
    loss_aux.backward()

    grads_aux = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads_aux[name] = param.grad.detach().clone()

    # (c) 检查核心共享模块的梯度方向
    conflict_modules = ["encoder", "decoder", "backbone.backbone_ir"]
    for mod_name in conflict_modules:
        dot_product = 0.0
        norm_main_sq = 0.0
        norm_aux_sq = 0.0
        match_count = 0

        for p_name in grads_main:
            if mod_name in p_name and p_name in grads_aux:
                gm = grads_main[p_name].flatten()
                ga = grads_aux[p_name].flatten()
                dot_product += (gm * ga).sum().item()
                norm_main_sq += (gm * gm).sum().item()
                norm_aux_sq += (ga * ga).sum().item()
                match_count += 1

        norm_main = np.sqrt(norm_main_sq)
        norm_aux = np.sqrt(norm_aux_sq)
        cos_sim = dot_product / (norm_main * norm_aux + 1e-9)

        conflict_flag = "❌ 存在严重梯度对立冲突！" if cos_sim < 0 else "✅ 梯度正向协同"
        print(f"▶ 模块 [{mod_name}]:")
        print(f"   ├─ 统计参数张量数: {match_count}")
        print(f"   ├─ 梯度余弦夹角 Cosine: {cos_sim:+.4f}")
        print(f"   └─ 判定结论: {conflict_flag}")

    print("\n" + "=" * 80)
    print("🏁 [机理透视诊断完成] 请根据以上硬核数据剖析物理成因！")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    cfg_file = "configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_infranet.yml"
    ckpt_file = "outputs/rtdetrv2_obb_hgnetv2_s_atrumod_infranet/best_stg1_origin.pth"
    run_infranet_diagnosis(cfg_file, ckpt_file)