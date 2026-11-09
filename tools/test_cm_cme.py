"""CM-CME的CPU机制检查：仅合成特征、单次反向，不训练或下载。"""

import importlib.util
import io
import math
from pathlib import Path
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F


SOURCE = Path(__file__).resolve().parents[1] / "engine/backbone/cm_cme.py"
SPEC = importlib.util.spec_from_file_location("cm_cme_unit", SOURCE)
CM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CM)


def identity_work(model: nn.Module) -> None:
    """设置已知工作投影，便于用解析数值辨别S与Z。"""
    with torch.no_grad():
        for work in (model.rgb_work, model.ir_work):
            work[0].weight.copy_(torch.eye(4)[:, :, None, None])
            work[1] = nn.Identity()


class CMCMEChecks(unittest.TestCase):
    """验证供需方向、同时更新、初态和梯度可学习性。"""

    def setUp(self) -> None:
        """每项检查使用独立小模型和可复现合成特征。"""
        self.model = CM.CMCME(4, condition_dim=3, state_dim=2, norm_groups=2)
        generator = torch.Generator().manual_seed(4)
        self.rgb = torch.randn(2, 4, 3, 4, generator=generator)
        self.ir = torch.randn(2, 4, 3, 4, generator=generator)
        self.condition = torch.randn(2, 3, generator=generator)

    def test_initial_identity_gate_and_zero_message(self) -> None:
        """初始FiLM恒等、门控0.125、消息零，输出等于工作特征。"""
        s_rgb, s_ir = self.model.rgb_work(self.rgb), self.model.ir_work(self.ir)
        l_rgb, r_rgb, _ = self.model.rgb_state(s_rgb)
        l_ir, _, n_ir = self.model.ir_state(s_ir)
        gate, message = self.model.rgb_to_ir(s_rgb, s_ir, l_rgb, l_ir, r_rgb, n_ir, self.condition)
        torch.testing.assert_close(gate, torch.full_like(gate, 0.125))
        self.assertTrue(torch.equal(message, torch.zeros_like(message)))
        actual = self.model(self.rgb, self.ir, self.condition)
        torch.testing.assert_close(actual[0], s_rgb)
        torch.testing.assert_close(actual[1], s_ir)

    def test_equal_work_ignores_film_difference_in_message(self) -> None:
        """相同S产生零消息，即使FiLM让Z不同且卷积已非零。"""
        identity_work(self.model)
        with torch.no_grad():
            self.model.rgb_film.bias[:4].fill_(0.5)
            self.model.rgb_film.bias[4:].fill_(1)
            self.model.ir_film.bias[4:].fill_(-2)
            for direction in (self.model.rgb_to_ir, self.model.ir_to_rgb):
                direction.difference_conv.weight.fill_(0.2)
        same = torch.ones_like(self.rgb)
        rgb, ir = self.model(same, same, self.condition)
        torch.testing.assert_close(rgb, torch.full_like(rgb, 2.5))
        torch.testing.assert_close(ir, torch.full_like(ir, -1))

    def test_directional_gate_adds_logits_and_keeps_order(self) -> None:
        """门控使用R×N与相加logit，局部输入按供体/受体/差分排序。"""
        direction = self.model.rgb_to_ir
        with torch.no_grad():
            direction.condition_bias.bias.fill_(1)
            direction.local_output.bias.fill_(-0.25)
        donor = torch.full((2, 2, 3, 4), 3.0)
        receiver = torch.full_like(donor, 1.0)
        captured = []
        hook = direction.local_hidden.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
        support, need = torch.full_like(self.rgb, 0.2), torch.full_like(self.rgb, 0.8)
        gate, _ = direction(self.rgb, self.ir, donor, receiver, support, need, self.condition)
        hook.remove()
        torch.testing.assert_close(captured[0], torch.cat((donor, receiver, donor - receiver), dim=1))
        torch.testing.assert_close(gate, support * need * torch.sigmoid(torch.tensor(0.75)))
        self.assertTrue((gate <= support * need).all())

    def test_simultaneous_updates_use_donor_support_receiver_need_and_s(self) -> None:
        """非对称解析例子区分方向索引、S/Z和顺序更新。"""
        identity_work(self.model)
        with torch.no_grad():
            for state, r, n in ((self.model.rgb_state, 0.2, 0.3), (self.model.ir_state, 0.7, 0.8)):
                state.support_head.bias.fill_(math.log(r / (1 - r)))
                state.need_head.bias.fill_(math.log(n / (1 - n)))
            self.model.rgb_film.bias[:4].fill_(0.5)
            self.model.rgb_film.bias[4:].fill_(2)
            self.model.ir_film.bias[:4].fill_(-0.25)
            self.model.ir_film.bias[4:].fill_(-0.3)
            self.model.rgb_to_ir.difference_conv.weight[:, :, 1, 1].fill_(0.7)
            self.model.ir_to_rgb.difference_conv.weight[:, :, 1, 1].fill_(-0.3)
        rgb, ir = self.model(torch.full_like(self.rgb, 2), torch.ones_like(self.ir), self.condition)
        expected_rgb = 5 + 0.7 * 0.3 * 0.5 * F.gelu(torch.tensor(0.3))
        expected_ir = 0.45 + 0.2 * 0.8 * 0.5 * F.gelu(torch.tensor(0.7))
        torch.testing.assert_close(rgb, torch.full_like(rgb, expected_rgb))
        torch.testing.assert_close(ir, torch.full_like(ir, expected_ir))
        first = {id(p) for p in self.model.rgb_to_ir.parameters()}
        self.assertFalse(first & {id(p) for p in self.model.ir_to_rgb.parameters()})
        self.assertIsNone(self.model.rgb_to_ir.difference_conv.bias)

    def test_zero_message_start_has_live_gradients(self) -> None:
        """初态卷积/FiLM可学，门控梯度延迟到消息非零之后。"""
        rgb, ir = self.model(self.rgb, self.ir, self.condition)
        (rgb.square().mean() + ir.square().mean()).backward()
        for layer in (self.model.rgb_to_ir.difference_conv, self.model.ir_to_rgb.difference_conv,
                      self.model.rgb_film, self.model.ir_film):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        self.assertEqual(self.model.rgb_to_ir.condition_bias.weight.grad.abs().sum().item(), 0)
        self.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            self.model.rgb_to_ir.difference_conv.weight[:, :, 1, 1].fill_(0.2)
        rgb, ir = self.model(self.rgb, self.ir, self.condition)
        (rgb.square().mean() + ir.square().mean()).backward()
        gradient = self.model.rgb_to_ir.condition_bias.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_cpu_autocast_accepts_bfloat16_condition(self) -> None:
        """CPU混合精度接受低精度条件，输出形状正确且数值有限。"""
        with torch.autocast("cpu", dtype=torch.bfloat16):
            outputs = self.model(self.rgb.bfloat16(), self.ir.bfloat16(), self.condition.bfloat16())
        for output in outputs:
            self.assertEqual(output.shape, self.rgb.shape)
            self.assertTrue(torch.isfinite(output).all())

    def test_invalid_inputs_and_group_count_fail(self) -> None:
        """错配形状、类型、条件和非法数值报错，不自动替换。"""
        cases = [(self.rgb[:, :3], self.ir, self.condition), (self.rgb, self.ir[:, :, :2], self.condition),
                 (self.rgb, self.ir.double(), self.condition), (self.rgb, self.ir, self.condition[:, :2]),
                 (self.rgb, self.ir, self.condition.double()), (self.rgb, self.ir, self.condition * float("nan"))]
        for args in cases:
            with self.subTest(shape=args[0].shape), self.assertRaises(ValueError):
                self.model(*args)
        with self.assertRaises(ValueError):
            CM.CMCME(4, norm_groups=3)

    def test_state_dict_preserves_both_branches(self) -> None:
        """保存恢复保留两个方向的参数及两路输出。"""
        stream = io.BytesIO()
        torch.save(self.model.state_dict(), stream)
        stream.seek(0)
        restored = CM.CMCME(4, condition_dim=3, state_dim=2, norm_groups=2)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        for actual, expected in zip(restored(self.rgb, self.ir, self.condition),
                                    self.model(self.rgb, self.ir, self.condition)):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
