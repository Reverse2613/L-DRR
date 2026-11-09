"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from .common import (
    get_activation,
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)
from .lsknet import LSKNet
from .presnet import PResNet
from .test_resnet import MResNet

from .timm_model import TimmModel
from .torchvision_model import TorchVisionModel

from .csp_resnet import CSPResNet
from .csp_darknet import CSPDarkNet, CSPPAN

from .hgnetv2 import HGNetv2

from .dual_hgnetv2_concat import DualHGNetv2_Concat
from .dual_hgnetv2_lg_dense import DualHGNetv2_LGDense
from .dual_hgnetv2_add import DualHGNetv2_Add
from .single_hgnetv2_ir import SingleHGNetv2_IR
from .single_hgnetv2_rgb import SingleHGNetv2_RGB

#InfraNet
from .qual_gate import QualGate
from .dual_hgnetv2_infranet import DualHGNetv2_InfraNet
from .dual_hgnetv2_infranet_improve import DualHGNetv2_InfraNet_Improve

from .dual_hgnetv2_pg import DualHGNetv2_PG
