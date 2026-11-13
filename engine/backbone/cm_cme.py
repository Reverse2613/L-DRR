"""CM-CME：条件调制的双向局部差分互补，保留RGB与IR两条支路。

阅读顺序：LocalModalityState（供体支持/受体需求）→ DirectionalExchange
（一个方向的门控和消息）→ CMCME.forward（工作投影、FiLM、同时更新）。
本文件实现一个尺度；S3/S4/S5以后各实例化一份，参数互不共享。
"""

from typing import Tuple

import torch
import torch.nn as nn


class LocalModalityState(nn.Module):
    r"""从未调制的工作特征读取局部状态、支持响应与需求响应。"""

    def __init__(self, channels: int, state_dim: int = 32) -> None:
        r"""建立局部状态投影及两套独立响应头。

        Args:
            channels: 工作特征通道C。
            state_dim: 低维局部状态维度，方案默认32。
        """
        super().__init__()
        self.state_projection = nn.Conv2d(channels, state_dim, 1)
        self.activation = nn.GELU()
        self.support_head = nn.Conv2d(state_dim, channels, 1)
        self.need_head = nn.Conv2d(state_dim, channels, 1)
        # 初始R=N=0.5；两头独立，不能把需求写成1-R。
        for head in (self.support_head, self.need_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, work: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""计算本模态的局部状态和两个控制响应。

        Args:
            work: 未经FiLM调制的S [B,C,H,W]。

        Returns:
            L [B,state_dim,H,W]；R、N各[B,C,H,W]，经Sigmoid约束到[0,1]。
            R/N是任务控制响应，不是真实SNR或已校准传感器质量。
        """
        local = self.activation(self.state_projection(work))
        support = self.support_head(local).sigmoid()
        need = self.need_head(local).sigmoid()
        return local, support, need


class DirectionalExchange(nn.Module):
    r"""一个供体→受体方向的条件/视觉门控与局部差分消息。"""

    def __init__(self, channels: int, condition_dim: int = 128, state_dim: int = 32) -> None:
        r"""建立方向独立的投影与无偏置差分卷积。

        Args:
            channels: 工作空间通道C。
            condition_dim: DT-CPN条件维度，默认128。
            state_dim: 局部状态维度，默认32。
        """
        super().__init__()
        self.local_hidden = nn.Conv2d(3 * state_dim, state_dim, 1)
        self.local_activation = nn.GELU()
        self.local_output = nn.Conv2d(state_dim, channels, 1)
        self.condition_bias = nn.Linear(condition_dim, channels)
        self.difference_conv = nn.Conv2d(channels, channels, 3, padding=1,
                                         groups=channels, bias=False)
        self.message_activation = nn.GELU()
        # p=h=0；与R=N=0.5组合，初始G=0.5×0.5×0.5=0.125。
        for head in (self.local_output, self.condition_bias):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        # 消息从零开始，但此卷积本身可以获得首步梯度。
        # 不再串联一个同为零的可学习系数，以免整个消息分支成为死路。
        nn.init.zeros_(self.difference_conv.weight)

    def forward(
        self, donor_work: torch.Tensor, receiver_work: torch.Tensor,
        donor_local: torch.Tensor, receiver_local: torch.Tensor,
        donor_support: torch.Tensor, receiver_need: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""计算一个方向的传输门控G和未乘门控的消息E。

        Args:
            donor_work: 供体S [B,C,H,W]，未经过FiLM。
            receiver_work: 受体S [B,C,H,W]，与供体同一工作坐标。
            donor_local: 供体L [B,state_dim,H,W]。
            receiver_local: 受体L [B,state_dim,H,W]。
            donor_support: 供体R [B,C,H,W]，不是受体R。
            receiver_need: 受体N [B,C,H,W]，不是供体N。
            condition: c [B,condition_dim]。

        Returns:
            G、E各[B,C,H,W]；G≤R_donor×N_receiver，E读取供体减受体的S。
        """
        # 输入顺序有意义：[供体L,受体L,供体L-受体L]，不能改成无序平均。
        ordered = torch.cat((donor_local, receiver_local, donor_local - receiver_local), dim=1)
        visual = self.local_output(self.local_activation(self.local_hidden(ordered)))
        prior = self.condition_bias(condition)[:, :, None, None]
        # 条件与局部响应在logit空间相加，局部证据可以修正条件倾向。
        gate = donor_support * receiver_need * (prior + visual).sigmoid()
        message = self.message_activation(self.difference_conv(donor_work - receiver_work))
        return gate, message


class CMCME(nn.Module):
    r"""单尺度双向互补：输入两路特征与条件，输出两路更新特征。

    本步默认工作空间采用GroupNorm(32组)。投影/归一化是可训练工作表示，
    不代表已经完成跨传感器空间配准。输出顺序固定为RGB、IR。
    """

    def __init__(
        self, channels: int, condition_dim: int = 128,
        state_dim: int = 32, norm_groups: int = 32, enable_film: bool = True,
    ) -> None:
        r"""建立一个尺度的两条主路径及两个独立方向。

        Args:
            channels: 本尺度通道；S3/S4/S5分别为256/512/1024。
            condition_dim: 条件维度，默认128。
            state_dim: 局部状态维度，默认32。
            norm_groups: GroupNorm组数，必须整除channels，默认32。
            enable_film: False只绕过FiLM，保留其参数、工作投影与消息分支。
                默认True，旧配置与checkpoint的张量结构及计算保持原样。

        Raises:
            ValueError: 维度非正整数，或归一化组数不能整除通道数。
        """
        super().__init__()
        if any(type(value) is not int or value <= 0
               for value in (channels, condition_dim, state_dim, norm_groups)):
            raise ValueError("CM-CME各维度及归一化组数必须是正整数。")
        if channels % norm_groups:
            raise ValueError("norm_groups必须整除channels，不自动调整组数。")
        if type(enable_film) is not bool:
            raise ValueError("enable_film必须是bool。")
        self.channels = channels
        self.condition_dim = condition_dim
        self.enable_film = enable_film
        self.rgb_work = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False),
                                      nn.GroupNorm(norm_groups, channels))
        self.ir_work = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False),
                                     nn.GroupNorm(norm_groups, channels))
        self.rgb_film = nn.Linear(condition_dim, 2 * channels)
        self.ir_film = nn.Linear(condition_dim, 2 * channels)
        self.rgb_state = LocalModalityState(channels, state_dim)
        self.ir_state = LocalModalityState(channels, state_dim)
        self.rgb_to_ir = DirectionalExchange(channels, condition_dim, state_dim)
        self.ir_to_rgb = DirectionalExchange(channels, condition_dim, state_dim)
        # 初始γ=β=0，所以Z=S；工作投影与隐藏层仍保持非零常规初始化。
        for film in (self.rgb_film, self.ir_film):
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

    def _validate_inputs(
        self, rgb: torch.Tensor, ir: torch.Tensor, condition: torch.Tensor,
    ) -> None:
        r"""检查形状、类型、设备和数值，防止模态错配或无声广播。

        Args:
            rgb: RGB特征 [B,C,H,W]。
            ir: IR特征 [B,C,H,W]，与RGB形状和类型相同。
            condition: c [B,condition_dim]；三者与本模块同设备。
        """
        if not all(isinstance(value, torch.Tensor) for value in (rgb, ir, condition)):
            raise ValueError("RGB、IR和condition都必须是张量。")
        if (rgb.ndim != 4 or rgb.shape[1] != self.channels
                or any(size <= 0 for size in rgb.shape) or ir.shape != rgb.shape):
            raise ValueError(f"两路特征必须同为非空[B,{self.channels},H,W]。")
        if condition.shape != (rgb.shape[0], self.condition_dim):
            raise ValueError(f"condition必须为[{rgb.shape[0]},{self.condition_dim}]。")
        allowed = (torch.float32, torch.float16, torch.bfloat16)
        if rgb.dtype not in allowed or ir.dtype != rgb.dtype or condition.dtype not in allowed:
            raise ValueError("特征需要相同浮点类型，条件也需要float32/float16/bfloat16。")
        device = self.rgb_work[0].weight.device
        if any(value.device != device for value in (rgb, ir, condition)):
            raise ValueError(f"两路特征与条件必须在模型设备{device}。")
        if not all(torch.isfinite(value).all() for value in (rgb, ir, condition)):
            raise ValueError("CM-CME输入含NaN/Inf，不自动替换。")

    def _apply_film(self, work: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        r"""在工作空间执行条件仿射调制。

        Args:
            work: S [B,C,H,W]。
            modulation: 条件线性层输出 [B,2C]，前C项γ、后C项β。

        Returns:
            Z=(1+γ)×S+β [B,C,H,W]，调制参数沿空间广播。
        """
        gamma, beta = modulation.chunk(2, dim=1)
        return (1 + gamma[:, :, None, None]) * work + beta[:, :, None, None]

    def forward(
        self, rgb: torch.Tensor, ir: torch.Tensor, condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""从同一份旧特征计算两个方向，再同时更新RGB和IR。

        Args:
            rgb: 本尺度RGB特征 [B,C,H,W]。
            ir: 本尺度IR特征 [B,C,H,W]。
            condition: DT-CPN输出 [B,condition_dim]。

        Returns:
            (rgb_updated, ir_updated)，各[B,C,H,W]，供CG-DMF评估贡献。
            这里保留两路，尚未将它们融合成一张特征图。
        """
        self._validate_inputs(rgb, ir, condition)
        # 1. 分别投影并归一化，得到共同工作空间S。
        s_rgb, s_ir = self.rgb_work(rgb), self.ir_work(ir)
        # 2. FiLM只调制残差主路径；后面的供需/差分仍从S读取。
        if self.enable_film:
            z_rgb = self._apply_film(s_rgb, self.rgb_film(condition))
            z_ir = self._apply_film(s_ir, self.ir_film(condition))
        else:
            z_rgb, z_ir = s_rgb, s_ir
        l_rgb, r_rgb, n_rgb = self.rgb_state(s_rgb)
        l_ir, r_ir, n_ir = self.ir_state(s_ir)
        # 3. 两个方向参数独立，均读取未更新的S和L。
        g_rgb_to_ir, e_rgb_to_ir = self.rgb_to_ir(
            s_rgb, s_ir, l_rgb, l_ir, r_rgb, n_ir, condition)
        g_ir_to_rgb, e_ir_to_rgb = self.ir_to_rgb(
            s_ir, s_rgb, l_ir, l_rgb, r_ir, n_rgb, condition)
        # 4. 供体消息加到受体Z；不先更新一路再用它计算另一路。
        rgb_updated = z_rgb + g_ir_to_rgb * e_ir_to_rgb
        ir_updated = z_ir + g_rgb_to_ir * e_rgb_to_ir
        return rgb_updated, ir_updated
