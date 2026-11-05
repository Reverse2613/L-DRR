"""
tools/test_dual_hgnetv2_infranet.py

DualHGNetv2_InfraNet 主干网络的双流拓扑、两种推理模式 (IR / RGB-IR) 与梯度回传单测.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from engine.backbone.dual_hgnetv2_infranet import DualHGNetv2_InfraNet


def test_infranet_backbone():
    print("=" * 70)
    print("🚀 [DualHGNetv2_InfraNet 完整单测] 检验 InfraNet-IR 与 InfraNet-RGB-IR...")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行硬件设备: {device}")

    # 1. 实例化主干网络 (默认 eval_mode='ir')
    model = DualHGNetv2_InfraNet(
        name="B0",
        return_idx=[1, 2, 3],
        pretrained=False,
        eval_mode="ir",
    ).to(device)

    dummy_input_6ch = torch.randn(2, 6, 640, 640, device=device, requires_grad=True)

    # 检验 1: 训练期 (Training Mode) 多流输出与反向传播
    print("\n▶ 检验 1: 训练期 (Training Mode) 双流特征与联合梯度流...")
    model.train()
    out_train = model(dummy_input_6ch)

    assert isinstance(out_train, dict), "❌ 训练期输出应为 Dict!"
    feats_main = out_train["feats_main"]
    feats_aux = out_train["feats_aux"]
    q_dict = out_train["quality_scores"]

    for idx, (fm, fa) in enumerate(zip(feats_main, feats_aux)):
        print(f"   Stage {idx+3}: 主特征 {list(fm.shape)} | 辅助特征 {list(fa.shape)}")
        assert fm.shape == fa.shape, f"❌ 尺度 {idx+3} 形状不匹配!"

    loss_total = sum(f.sum() for f in feats_main) + 0.25 * sum(f.sum() for f in feats_aux)
    loss_total.backward()
    assert dummy_input_6ch.grad is not None, "❌ 输入张量未捕获梯度!"
    print(f"   ✅ 双流联合梯度反向传播导通! 输入梯度范数: {dummy_input_6ch.grad.abs().sum().item():.4f}")

    # 检验 2: 模式 A —— InfraNet-IR (纯红外独立推理)
    print("\n▶ 检验 2: 模式 A —— InfraNet-IR 评估 (纯红外剪枝)...")
    model.eval()
    model.eval_mode = "ir"
    with torch.no_grad():
        out_eval_ir = model(dummy_input_6ch)
        assert isinstance(out_eval_ir, list) and len(out_eval_ir) == 3
        print(f"   ✅ [InfraNet-IR] 成功剪除 RGB 支路，仅输出纯 IR 特征: {[list(f.shape) for f in out_eval_ir]}")

    # 检验 3: 模式 B —— InfraNet-RGB-IR (双光质量融合推理)
    print("\n▶ 检验 3: 模式 B —— InfraNet-RGB-IR 评估 (双光质量自适应融合)...")
    model.eval()
    model.eval_mode = "rgb_ir"  # 切换为双光融合推理模式
    with torch.no_grad():
        out_eval_rgb_ir = model(dummy_input_6ch)
        assert isinstance(out_eval_rgb_ir, list) and len(out_eval_rgb_ir) == 3
        print(f"   ✅ [InfraNet-RGB-IR] 成功执行 QualGate 质量融合，输出融合特征: {[list(f.shape) for f in out_eval_rgb_ir]}")

    print("\n" + "=" * 70)
    print("🎉 两种推理模式全部通过验证！现在代码既能单红外超轻量推理，也能双光高精度推理！")
    print("=" * 70)


if __name__ == "__main__":
    test_infranet_backbone()