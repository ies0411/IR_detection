import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.registry import MODELS
import torchvision.models as models
from torchvision.models import RegNet_Y_128GF_Weights

import torch.utils.checkpoint as checkpoint


@MODELS.register_module()
class MyBackBone(BaseModule):
    def __init__(self, pretrained=True, init_cfg=None):
        super().__init__(init_cfg)

        # Load the pretrained RegNet_Y_128GF model from torchvision
        self.regnet = models.regnet_y_128gf(
            weights=(
                RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
            )
        )
        self.stem = self.regnet.stem
        self.stage1 = self.regnet.trunk_output.block1
        self.stage2 = self.regnet.trunk_output.block2
        self.stage3 = self.regnet.trunk_output.block3
        self.stage4 = self.regnet.trunk_output.block4

        # You can directly use the model as your backbone
        # Removing the classification head to get feature maps
        self.regnet.fc = nn.Identity()

    def forward(self, x):

        def custom_forward(x):
            x = self.stem(x)
            x = self.stage1(x)
            stage1_out = x

            x = self.stage2(x)
            stage2_out = x

            x = self.stage3(x)
            stage3_out = x

            x = self.stage4(x)
            stage4_out = x

            return (stage1_out, stage2_out, stage3_out, stage4_out)

        x = checkpoint.checkpoint(custom_forward, x)
        return x
