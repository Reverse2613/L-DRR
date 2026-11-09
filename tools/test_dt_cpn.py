"""DT-CPN的CPU单元检查：合成条件和缓存，不下载、不训练。"""

import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dt_cpn_unit", REPO_ROOT / "engine/backbone/dt_cpn.py")
DT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DT)


def make_cache(directory: Path) -> Path:
    """在临时目录生成带有效校验值的合成单位锚点，返回文本路径。"""
    source = directory / "texts.json"
    source.write_bytes(DT.DEFAULT_TEXTS_PATH.read_bytes())
    texts = json.loads(source.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "texts_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "texts_by_id": {field: [row["text"] for row in texts[field]] for field in DT.ANCHOR_SPECS},
        "model": {"model_id": "openai/clip-vit-base-patch16"},
        "projection": {"shape": [64, 512]}, "artifacts": {},
    }
    generator = torch.Generator().manual_seed(7)
    for count, filename in DT.ANCHOR_SPECS.values():
        table = torch.randn(count, 64, generator=generator)
        table = table / table.norm(dim=1, keepdim=True)
        path = directory / filename
        torch.save(table, path)
        manifest["artifacts"][filename] = {"shape": [count, 64], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source


def conditions(hour: float = 0.0) -> dict[str, torch.Tensor]:
    """返回两对图像的合法合成条件，时间在数据端编码为sin/cos。"""
    phase = 2 * math.pi * hour / 24
    return {"height": torch.tensor([[0.25], [0.5]]), "angle": torch.tensor([[0.0], [1.0]]),
            "time": torch.tensor([[math.sin(phase), math.cos(phase)]] * 2),
            "weather": torch.tensor([0, 1]), "illumination": torch.tensor([2, 3]),
            "location": torch.tensor([2, 9])}


class DTCPNTests(unittest.TestCase):
    """检查数学定义、冻结状态、输入边界和保存恢复。"""

    def setUp(self) -> None:
        """每项测试使用独立临时缓存，不触碰用户的真实锚点。"""
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = make_cache(self.directory)
        self.model = DT.DTCPN(self.directory, self.source)

    def test_fourier_interleaving_and_no_second_normalization(self) -> None:
        """频率和sin/cos交错正确，forward直接读取规范化高度。"""
        value = torch.tensor([[0.25]])
        expected = torch.tensor([[entry for k in range(8) for entry in
                                  (math.sin(math.pi * 2**k * 0.25), math.cos(math.pi * 2**k * 0.25))]])
        torch.testing.assert_close(self.model._fourier_encode(value), expected, atol=2e-5, rtol=0)
        captured = []
        handle = self.model.geo_projection.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0]))
        self.model(conditions())
        handle.remove()
        torch.testing.assert_close(captured[0][:, :16], self.model._fourier_encode(conditions()["height"]))

    def test_zero_residual_and_frozen_buffers(self) -> None:
        """初态等于文本锚点，锚点在buffer中而不在可训练参数中。"""
        embedding = self.model.weather_embedding
        torch.testing.assert_close(embedding(torch.tensor([0, 1])), embedding.anchor[:2])
        self.assertTrue(torch.equal(embedding.residual.weight, torch.zeros(10, 64)))
        self.assertIn("weather_embedding.anchor", dict(self.model.named_buffers()))
        self.assertNotIn("weather_embedding.anchor", dict(self.model.named_parameters()))
        self.assertFalse(embedding.anchor.requires_grad)

    def test_large_residual_is_bounded_without_unit_renormalization(self) -> None:
        """大残差被限制到0.2，锚点加残差后不额外强制单位化。"""
        embedding = self.model.weather_embedding
        with torch.no_grad():
            embedding.residual.weight.copy_(embedding.anchor * 100)
        result = embedding(torch.arange(10))
        distance = (result - embedding.anchor).norm(dim=1)
        self.assertTrue((distance <= 0.200001).all())
        torch.testing.assert_close(result.norm(dim=1), torch.full((10,), 1.2))

    def test_condition_only_forward_and_time_period(self) -> None:
        """仅六条件就能输出128维，午夜与24点的周期表示等价。"""
        zero, midnight = self.model(conditions(0)), self.model(conditions(24))
        self.assertEqual(zero.shape, (2, 128))
        self.assertTrue(torch.isfinite(zero).all())
        torch.testing.assert_close(zero, midnight)
        self.assertFalse(torch.allclose(zero, self.model(conditions(6))))

    def test_backward_reaches_selected_residuals_only(self) -> None:
        """单次反向能学习被查到的残差行，冻结锚点没有梯度。"""
        self.model(conditions()).square().mean().backward()
        gradient = self.model.location_embedding.residual.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[[2, 9]].abs().sum().item(), 0)
        self.assertEqual(gradient[0].abs().sum().item(), 0)
        self.assertIsNone(self.model.location_embedding.anchor.grad)
        self.assertGreater(self.model.geo_projection.weight.grad.abs().sum().item(), 0)

    def test_invalid_conditions_are_rejected(self) -> None:
        """缺字段、错误形状/类型和非法数值都报错而不补值。"""
        cases = [("height", torch.tensor([0.5, 0.5])), ("height", torch.full((2, 1), 120.0)),
                 ("height", torch.zeros(2, 1, dtype=torch.float64)), ("angle", torch.full((2, 1), float("nan"))),
                 ("time", torch.zeros(2, 2)), ("time", torch.ones(2, 1)),
                 ("weather", torch.tensor([10, 0])), ("weather", torch.tensor([0.0, 1.0])),
                 ("location", torch.tensor([-1, 0]))]
        missing = conditions()
        del missing["weather"]
        with self.assertRaises(ValueError):
            self.model(missing)
        for field, value in cases:
            invalid = conditions()
            invalid[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.model(invalid)

    def test_cache_hash_order_and_missing_file_fail(self) -> None:
        """缓存校验不符、原始编号错序和缺文件不能被静默接受。"""
        path = self.directory / "weather_anchor.pt"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaises(ValueError):
            DT.load_anchor_tables(self.directory, self.source)
        make_cache(self.directory)
        source = json.loads(self.source.read_text())
        source["weather"][0]["id"] = 5
        self.source.write_text(json.dumps(source))
        manifest_path = self.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["texts_file_sha256"] = hashlib.sha256(self.source.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            DT.load_anchor_tables(self.directory, self.source)
        make_cache(self.directory)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            DT.load_anchor_tables(self.directory, self.source)

    def test_state_dict_round_trip_preserves_buffers_and_output(self) -> None:
        """保存恢复同时保留锚点buffer、频率和网络输出。"""
        stream = io.BytesIO()
        torch.save(self.model.state_dict(), stream)
        stream.seek(0)
        restored = DT.DTCPN(self.directory, self.source)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(restored(conditions()), self.model(conditions()))
        self.assertTrue(torch.equal(restored.frequencies, self.model.frequencies))
        self.assertTrue(torch.equal(restored.location_embedding.anchor, self.model.location_embedding.anchor))


if __name__ == "__main__":
    unittest.main()
