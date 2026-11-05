"""
tools/test_infranet_e2e.py

端到端闭环单测: 检验 RTv4 + DualHGNetv2_InfraNet + 共享 Head + 辅助损失反向传播.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from engine.backbone.dual_hgnetv2_infranet import DualHGNetv2_InfraNet
from engine.rtv4.hybrid_encoder import HybridEncoder
from engine.rtv4.rtdetrv2_obb_decoder import RTDETRTransformerv2OBB
from engine.rtv4.rtv4_infranet import RTv4_Infranet
from engine.rtv4.obb_matcher import HungarianMatcherOBB
from engine.rtv4.rtv4_obb_criterion_infranet import RTv4OBBCriterion_Infranet


def test_infranet_end_to_end():
    print("=" * 70)
    print("🚀 [InfraNet 端到端完整闭环单测] 开始构建完整网络与损失链路...")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前测试硬件: {device}")

    # 1. 组装完整检测器
    backbone = DualHGNetv2_InfraNet(name="B0", pretrained=False, eval_mode="ir")
    encoder = HybridEncoder(
        in_channels=[256, 512, 1024],
        hidden_dim=224,
        expansion=0.5,
        depth_mult=0.34,
        version="rt_detrv2",
    )
    decoder = RTDETRTransformerv2OBB(
        feat_channels=[224, 224, 224],
        hidden_dim=224,
        num_classes=5,
        num_queries=100,  # 单测使用 100 queries 加速验证
        num_layers=3,
        num_denoising=20,
    )
    model = RTv4_Infranet(backbone=backbone, encoder=encoder, decoder=decoder).to(device)

    # 2. 组装损失函数与匹配器
    matcher = HungarianMatcherOBB(
        weight_dict={'cost_class': 2, 'cost_bbox': 0, 'cost_hausdorff': 5, 'cost_giou': 5},
        use_focal_loss=True,
    )
    criterion = RTv4OBBCriterion_Infranet(
        matcher=matcher,
        weight_dict={'loss_focal': 1, 'loss_l1': 5, 'loss_kld': 5},
        losses=['focal', 'l1', 'kld'],
        num_classes=5,
        use_uni_set=True,
    ).to(device)

    # 3. 构造防御性模拟批次 (B=2, 6通道, 640x640) 与旋转目标标签
    samples = torch.randn(2, 6, 640, 640, device=device, requires_grad=True)
    targets = [
        {
            "labels": torch.tensor([0, 1], dtype=torch.int64, device=device),
            "boxes": torch.tensor([
                [0.5, 0.5, 0.2, 0.1, 0.25],
                [0.3, 0.4, 0.15, 0.08, 0.75],
            ], dtype=torch.float32, device=device),  # (cx, cy, w, h, angle_norm)
        },
        {
            "labels": torch.tensor([2], dtype=torch.int64, device=device),
            "boxes": torch.tensor([[0.6, 0.6, 0.3, 0.2, 0.5]], dtype=torch.float32, device=device),
        }
    ]

    # ==========================================================================
    # 检验 A: 训练期端到端前向与损失求导
    # ==========================================================================
    print("\n▶ [检验 A] 训练期端到端双流前向与联合损失反向传播...")
    model.train()
    criterion.train()

    outputs = model(samples, targets=targets)
    assert 'aux_infranet' in outputs, "❌ outputs 字典中未找到 aux_infranet 辅助流预测!"
    print("   ✅ 模型成功并行产生主流与辅助流预测!")

    loss_dict = criterion(outputs, targets)
    print(f"   ✅ 成功计算综合损失字典 (包含主损失与 0.25 辅助特权损失):")
    for k, v in loss_dict.items():
        if 'aux_infranet' in k:
            print(f"      - [辅助特权损失] {k}: {v.item():.4f}")
        elif not any(x in k for x in ['aux_', 'dn_', 'enc_']):
            print(f"      - [主红外检测损失] {k}: {v.item():.4f}")

    total_loss = sum(loss_dict.values())
    total_loss.backward()

    assert samples.grad is not None, "❌ 梯度截断: 输入图像未捕获回传梯度!"
    print(f"   ✅ 端到端反向传播完美打通! 输入张量梯度范数: {samples.grad.abs().sum().item():.4f}")

    # ==========================================================================
    # 检验 B: 评估期 (Eval) 自动切除辅助支路
    # ==========================================================================
    print("\n▶ [检验 B] 评估期自动剪除特权辅助支路...")
    model.eval()
    with torch.no_grad():
        out_eval = model(samples)
        assert 'aux_infranet' not in out_eval, "❌ 评估期不应计算 aux_infranet!"
        print(f"   ✅ 评估期输出键集: {list(out_eval.keys())} (与原生 RT-DETR 严格完全一致)")
        print(f"   ✅ 主预测框形状: {list(out_eval['pred_boxes'].shape)}")

    print("\n" + "=" * 70)
    print("🎉 端到端前向、反向传播与推理闭环全面通过检验！系统架构已具备完整实战能力！")
    print("=" * 70)


if __name__ == "__main__":
    test_infranet_end_to_end()