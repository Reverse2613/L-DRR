"""三条件CLIP的重训对照：固定常量向量和固定随机编码。

两组都保留原来的512维条件接口；只改变逐图条件向量的来源。
CLIP原表、对照表均为冻结buffer，FiLM与融合评分层仍由检测任务训练。
"""

from pathlib import Path
from typing import Mapping, Union

import torch

from .frozen_clip_condition import (
    DEFAULT_TEXTS_PATH, JOINT_ROW_COUNT, FrozenCLIPConditionPrior,
)


class ConstantCLIPConditionPrior(FrozenCLIPConditionPrior):
    """C组：所有样本共用同一条真实CLIP向量，不提供逐图条件信息。

    Attributes:
        constant_anchor: [1,512] float32冻结buffer，保持原向量的幅度。
        ema_copy_buffers: 对原表和常量向量均执行精确复制。
    """

    ema_copy_buffers = ("joint_anchor", "constant_anchor")

    def __init__(
        self, anchor_dir: Union[str, Path],
        texts_path: Union[str, Path] = DEFAULT_TEXTS_PATH,
        reference_row: int = 0,
    ) -> None:
        """从原770行缓存选定一条常量，未增加可学习参数。

        Args:
            anchor_dir: 与B组相同的三条件联合CLIP缓存目录。
            texts_path: 与B组相同的原始属性码表。
            reference_row: 实验前固定的缓存行号；默认0，不按检测成绩选择。

        Raises:
            ValueError: 行号不是合法整数或越界。
        """
        if type(reference_row) is not int or not 0 <= reference_row < JOINT_ROW_COUNT:
            raise ValueError(f"reference_row需要0到{JOINT_ROW_COUNT - 1}的整数。")
        super().__init__(anchor_dir, texts_path)
        self.reference_row = reference_row
        self.register_buffer(
            "constant_anchor", self.joint_anchor[reference_row:reference_row + 1].clone(),
        )

    @torch.no_grad()
    def forward(self, conditions: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """保留输入检查，返回与每张图的真实属性无关的常量。

        Args:
            conditions: 三类别张量各[B] int64，与缓存同设备。

        Returns:
            c [B,512] float32，无梯度；batch内外的条件向量完全相同。
        """
        self._validate_conditions(conditions)
        return self.constant_anchor.expand(conditions["weather"].shape[0], -1)


class RandomCLIPConditionPrior(FrozenCLIPConditionPrior):
    """D组：真实组合编号查固定随机表，保留条件身份、去掉CLIP语义。

    Attributes:
        random_anchor: [770,512] float32冻结buffer，逐行匹配原表L2范数。
        random_seed: 对照表的固定种子，与模型训练seed分开记录。
    """

    ema_copy_buffers = ("joint_anchor", "random_anchor")

    def __init__(
        self, anchor_dir: Union[str, Path],
        texts_path: Union[str, Path] = DEFAULT_TEXTS_PATH,
        random_seed: int = 20261112,
    ) -> None:
        """用独立CPU生成器构造一次随机表，不改变全局初始化序列。

        Args:
            anchor_dir: 与B组相同的三条件联合CLIP缓存目录。
            texts_path: 原始属性码表，用于严格核验编号。
            random_seed: 固定非负整数；不能根据结果挑选随机表。

        Raises:
            ValueError: 随机种子非法或随机行的范数异常。
        """
        if type(random_seed) is not int or not 0 <= random_seed < 2 ** 63:
            raise ValueError("random_seed需要[0,2**63)的整数。")
        super().__init__(anchor_dir, texts_path)
        self.random_seed = random_seed
        generator = torch.Generator(device="cpu").manual_seed(random_seed)
        random = torch.randn(self.joint_anchor.shape, generator=generator, dtype=torch.float32)
        lengths = torch.linalg.vector_norm(random, dim=1, keepdim=True)
        if not torch.isfinite(lengths).all() or (lengths <= 1e-7).any():
            raise ValueError("随机条件表含非法范数，不静默补值。")
        # 原CLIP表没有额外L2归一化，因此匹配每行幅度，而非统一改成单位球。
        random = random / lengths * torch.linalg.vector_norm(
            self.joint_anchor, dim=1, keepdim=True,
        )
        self.register_buffer("random_anchor", random)

    @torch.no_grad()
    def forward(self, conditions: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """按与B相同的真实三条件组合编号查固定随机表。

        Args:
            conditions: 三类别张量各[B] int64，与缓存同设备。

        Returns:
            c [B,512] float32，无梯度；同一组合在训练/推理中始终相同。
        """
        self._validate_conditions(conditions)
        index = (conditions["weather"] * 7 + conditions["illumination"]) * 11 + conditions["location"]
        return self.random_anchor.index_select(0, index)
