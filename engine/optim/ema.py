"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch
import torch.nn as nn

import math
from copy import deepcopy

from ..core import register
from ..misc import dist_utils

__all__ = ['ModelEMA']


@register()
class ModelEMA(object):
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model: nn.Module, decay: float=0.9999, warmups: int=1000, start: int=0):
        super().__init__()

        self.module = deepcopy(dist_utils.de_parallel(model)).eval()
        # if next(model.parameters()).device.type != 'cpu':
        #     self.module.half()  # FP16 EMA

        self.decay = decay
        self.warmups = warmups
        self.before_start = 0
        self.start = start
        self.updates = 0  # number of EMA updates
        if warmups == 0:
            self.decay_fn = lambda x: decay
        else:
            self.decay_fn = lambda x: decay * (1 - math.exp(-x / warmups))  # decay exponential ramp (to help early epochs)

        for p in self.module.parameters():
            p.requires_grad_(False)

        # 新冻结CLIP先验声明其固定缓存；精确复制，避免浮点EMA算术改变向量。
        # 旧模型未声明ema_copy_buffers，所有既有参数/缓冲区仍按原规则更新。
        self._ema_copy_buffer_names: set[str] = set()
        for module_name, submodule in self.module.named_modules():
            names = getattr(submodule, 'ema_copy_buffers', ())
            if not isinstance(names, (tuple, list)):
                raise ValueError('ema_copy_buffers必须是本模块固定buffer名称序列。')
            for name in names:
                if not isinstance(name, str) or name not in submodule._buffers:
                    raise ValueError(f'{module_name}声明了不存在的EMA固定buffer：{name}。')
                key = f'{module_name}.{name}' if module_name else name
                self._ema_copy_buffer_names.add(key)


    def update(self, model: nn.Module):
        if self.before_start < self.start:
            self.before_start += 1
            return
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates)
            msd = dist_utils.de_parallel(model).state_dict()
            for k, v in self.module.state_dict().items():
                if k in self._ema_copy_buffer_names:
                    v.copy_(msd[k].detach())
                elif v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        return self

    def state_dict(self, ):
        return dict(module=self.module.state_dict(), updates=self.updates)

    def load_state_dict(self, state, strict=True):
        self.module.load_state_dict(state['module'], strict=strict)
        if 'updates' in state:
            self.updates = state['updates']

    def forwad(self, ):
        raise RuntimeError('ema...')

    def extra_repr(self) -> str:
        return f'decay={self.decay}, warmups={self.warmups}'



class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """
    def __init__(self, model, decay, device="cpu", use_buffers=True):

        self.decay_fn = lambda x: decay * (1 - math.exp(-x / 2000))

        def ema_avg(avg_model_param, model_param, num_averaged):
            decay = self.decay_fn(num_averaged)
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=use_buffers)
