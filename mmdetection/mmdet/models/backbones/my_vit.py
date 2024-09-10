import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.registry import MODELS
import torchvision.models as models
from torchvision.models import ViT_L_16_Weights

import torch.utils.checkpoint as checkpoint


# ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1


@MODELS.register_module()
class MyViT(BaseModule):
    def __init__(self, pretrained=True, init_cfg=None):
        super().__init__(init_cfg)

        # Load the pretrained RegNet_Y_128GF model from torchvision
        vit = models.vit_l_16(
            weights=(ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None)
        )
        # ViT_L_16_Weights.IMAGENET1K_V1

        print(vit)
        # vit.encoder
        exit()
        self.stem = regnet.stem
        self.stage1 = regnet.trunk_output.block1
        self.stage2 = regnet.trunk_output.block2
        self.stage3 = regnet.trunk_output.block3
        self.stage4 = regnet.trunk_output.block4

        # You can directly use the model as your backbone
        # Removing the classification head to get feature maps
        # self.regnet.fc = nn.Identity()

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

            return tuple([stage1_out, stage2_out, stage3_out, stage4_out])

        x = checkpoint.checkpoint(custom_forward, x)
        return x
