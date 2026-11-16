"""CG-DMF之后的联合增益与成对通道混洗互补；不新增监督损失。

阅读顺序：JointGainGate（复用分数）→ PairedShuffleComplement（互补残差）
→ GainComplementFusion.forward（一次打分，门控原加权项，再加残差）。
增益是任务响应，不解释为已校准前景概率；两项的开关各自独立。
"""

from typing import Optional

import torch
import torch.nn as nn

from .cg_dmf import CGDMF


class JointGainGate(nn.Module):
    r"""用两张已有模态分数生成[0,2]范围的空间增益，初始恒为1。"""

    def __init__(self) -> None:
        r"""建立三个零初始化标量a_max、a_sum、a_bias。

        三个系数保存在同一个[3]Parameter中；已有pg_module1将一维参数
        放入无衰减组。原评分初始为零时，前两个系数首步梯度为零是预期。
        """
        super().__init__()
        self.coefficients = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def forward(self, rgb_score: torch.Tensor, ir_score: torch.Tensor) -> torch.Tensor:
        r"""计算2*sigmoid(a_max*max(s_r,s_i)+a_sum*(s_r+s_i)+a_bias)。

        Args:
            rgb_score: 浮点RGB分数[B,1,H,W]，不是前景概率。
            ir_score: 同形状、dtype、设备的IR分数。

        Returns:
            float32 [B,1,H,W]空间增益；精确算术范围(0,2)，浮点可饱和到端点。
            所有系数为零时输出精确为1；不再次调用原评分网络。

        Raises:
            ValueError: 分数的形状/类型/设备非法，或输入/计算出现NaN/Inf。
        """
        if not all(isinstance(value, torch.Tensor) for value in (rgb_score, ir_score)):
            raise ValueError("联合增益需要两张分数张量，不补默认分数。")
        if (rgb_score.ndim != 4 or rgb_score.shape[1] != 1
                or any(size <= 0 for size in rgb_score.shape) or ir_score.shape != rgb_score.shape):
            raise ValueError("联合增益两张分数必须同为非空[B,1,H,W]。")
        if rgb_score.dtype not in (torch.float32, torch.float16, torch.bfloat16) or ir_score.dtype != rgb_score.dtype:
            raise ValueError("联合增益分数必须是相同的受支持浮点类型。")
        if rgb_score.device != self.coefficients.device or ir_score.device != rgb_score.device:
            raise ValueError("联合增益分数与系数必须同设备。")
        if not all(torch.isfinite(value).all() for value in (rgb_score, ir_score)):
            raise ValueError("联合增益分数含NaN/Inf，不自动替换。")
        with torch.autocast(device_type=rgb_score.device.type, enabled=False):
            rgb, ir = rgb_score.float(), ir_score.float()
            a_max, a_sum, a_bias = self.coefficients.float().unbind()
            logits = a_max * torch.maximum(rgb, ir) + a_sum * (rgb + ir) + a_bias
            if not torch.isfinite(logits).all():
                raise ValueError("联合增益logit溢出或含NaN/Inf，不自动裁剪。")
            gain = 2.0 * torch.sigmoid(logits)
        return gain


class PairedShuffleComplement(nn.Module):
    r"""[R1,...,RC,I1,...,IC]→[R1,I1,...,RC,IC]的局部互补残差。

    混洗无参数；后续分组卷积、GroupNorm与逐点投影均有可学习参数。
    groups=C的3×3卷积每组读取一个(R_c,I_c)通道对并输出两个局部通道。
    """

    def __init__(self, channels: int, norm_groups: int) -> None:
        r"""建立成对3×3卷积→GroupNorm→GELU→零初始化2C→C投影。

        Args:
            channels: RGB/IR各自通道C。
            norm_groups: GroupNorm组数，本轮32，必须整除2C。

        Raises:
            ValueError: 通道或分组非法；不自动调小分组数。
        """
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (channels, norm_groups)):
            raise ValueError("混洗互补channels/norm_groups必须是正整数。")
        if (2 * channels) % norm_groups:
            raise ValueError("混洗互补norm_groups必须整除2C，不自动改变。")
        self.channels = channels
        self.pair_conv = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1, groups=channels, bias=False)
        self.norm = nn.GroupNorm(norm_groups, 2 * channels)
        self.activation = nn.GELU()
        self.out_projection = nn.Conv2d(2 * channels, channels, 1, bias=True)
        # 只置零出口；首步out_projection可学，前序卷积/Norm暂时零梯度是预期。
        nn.init.zeros_(self.out_projection.weight)
        nn.init.zeros_(self.out_projection.bias)

    def interleave(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        r"""严格按模态/通道维度交错，不沿空间轴混洗。

        Args:
            rgb: 浮点RGB特征[B,C,H,W]。
            ir: 同形状、dtype、设备的IR特征。

        Returns:
            [B,2C,H,W]，通道顺序R1,I1,R2,I2,...；保留两路梯度。
        """
        if not all(isinstance(value, torch.Tensor) for value in (rgb, ir)):
            raise ValueError("混洗互补需要两路特征张量。")
        if (rgb.ndim != 4 or rgb.shape[1] != self.channels
                or any(size <= 0 for size in rgb.shape) or ir.shape != rgb.shape):
            raise ValueError(f"混洗互补两路特征必须同为非空[B,{self.channels},H,W]。")
        if rgb.dtype not in (torch.float32, torch.float16, torch.bfloat16) or ir.dtype != rgb.dtype:
            raise ValueError("混洗互补两路特征必须使用相同浮点类型。")
        if rgb.device != ir.device or rgb.device != self.out_projection.weight.device:
            raise ValueError("混洗互补特征与模块必须同设备。")
        if not all(torch.isfinite(value).all() for value in (rgb, ir)):
            raise ValueError("混洗互补输入含NaN/Inf，不自动替换。")
        batch, _, height, width = rgb.shape
        grouped = torch.cat((rgb, ir), dim=1).reshape(batch, 2, self.channels, height, width)
        return grouped.transpose(1, 2).reshape(batch, 2 * self.channels, height, width)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        r"""从已FiLM调制的双流特征生成互补残差。

        Args:
            rgb: Z_rgb [B,C,H,W]，已完成微校准与FiLM。
            ir: Z_ir [B,C,H,W]，已完成原IR FiLM。

        Returns:
            [B,C,H,W]残差；初始严格为零。实际卷积dtype遵循外层AMP。
            本残差位于联合增益之外，不声称该门能关断全部融合路径。
        """
        shuffled = self.interleave(rgb, ir)
        hidden = self.activation(self.norm(self.pair_conv(shuffled)))
        output = self.out_projection(hidden)
        if not torch.isfinite(output).all():
            raise ValueError("混洗互补输出含NaN/Inf，不自动置零。")
        return output


class GainComplementFusion(CGDMF):
    r"""保留CG-DMF参数名与打分，增加独立开关的联合增益/互补残差。"""

    def __init__(
        self, channels: int, condition_dim: int, query_dim: int, norm_groups: int,
        enable_joint_gain: bool, enable_complement: bool,
    ) -> None:
        r"""建立与原CG-DMF同名的公共组件及新增出口。

        Args:
            channels: 本尺度C。
            condition_dim: 原CLIP条件维度，本轮512。
            query_dim: 原Query/Key维度，本轮32。
            norm_groups: 原输出GroupNorm分组，本轮32。
            enable_joint_gain: True实例化三个增益系数；False不增加该参数。
            enable_complement: True实例化混洗互补；False不增加该分支参数。
        """
        if any(type(flag) is not bool for flag in (enable_joint_gain, enable_complement)):
            raise ValueError("联合增益与互补开关必须是bool。")
        super().__init__(channels, condition_dim, query_dim, norm_groups)
        self.enable_joint_gain = enable_joint_gain
        self.enable_complement = enable_complement
        self.joint_gain: Optional[JointGainGate] = JointGainGate() if enable_joint_gain else None
        self.complement: Optional[PairedShuffleComplement] = (
            PairedShuffleComplement(channels, norm_groups) if enable_complement else None
        )

    @classmethod
    def from_base(
        cls, base: CGDMF, enable_joint_gain: bool, enable_complement: bool,
    ) -> "GainComplementFusion":
        r"""保留母体公共state，不消耗后续Encoder/Decoder的全局初始化随机数。

        Args:
            base: 已建立的原CG-DMF；保留其全部参数/缓冲区数值。
            enable_joint_gain: 联合增益开关。
            enable_complement: 混洗互补开关。

        Returns:
            公共参数名与数值相同的扩展CG-DMF；只新增已声明的两个分支。

        Raises:
            RuntimeError: 母体权重不兼容，或出现非新增分支的缺失/多余键。
        """
        if not isinstance(base, CGDMF):
            raise ValueError("扩展融合只能从原CGDMF母体建立。")
        with torch.random.fork_rng(devices=[]):
            result = cls(base.channels, base.condition_dim, base.rgb_score.query.out_channels,
                         base.out_projection[1].num_groups, enable_joint_gain, enable_complement)
        old_state = base.state_dict()
        expected_missing = {name for name in result.state_dict() if name not in old_state}
        if any(not name.startswith(("joint_gain.", "complement.")) for name in expected_missing):
            raise RuntimeError("扩展融合改变了原CG-DMF公共参数结构。")
        loaded = result.load_state_dict(old_state, strict=False)
        if set(loaded.missing_keys) != expected_missing or loaded.unexpected_keys:
            raise RuntimeError("母体融合权重存在非新增分支的缺失/多余键。")
        return result

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        r"""一次计算每路分数；增益乘原加权项，随后加互补，再原输出投影。

        Args:
            rgb: 已FiLM调制的RGB特征[B,C,H,W]。
            ir: 已FiLM调制的IR特征[B,C,H,W]。
            condition: 原条件向量[B,condition_dim]。

        Returns:
            融合特征[B,C,H,W]。初始G=1、互补=0，退化到母体CG-DMF。
            原三项打分、有序RGB/IR Query与末端投影/GroupNorm保持原样。
        """
        self._validate_inputs(rgb, ir, condition)
        rgb_score = self.rgb_score(rgb, ir, condition)
        ir_score = self.ir_score(ir, rgb, condition)
        rgb_weight = (rgb_score - ir_score).sigmoid()
        fused = rgb_weight * rgb + (1.0 - rgb_weight) * ir
        if self.joint_gain is not None:
            fused = self.joint_gain(rgb_score, ir_score) * fused
        if self.complement is not None:
            fused = fused + self.complement(rgb, ir)
        return self.out_projection(fused)
