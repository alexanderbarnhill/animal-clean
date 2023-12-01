"""
Module: unet_model.py
Authors: Christian Bergler
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 06.02.2021
"""

"""
Code from https://github.com/milesial/Pytorch-UNet
Code modified compared to https://github.com/milesial/Pytorch-UNet
Access Data: 06.02.2021, Last Access Date: 06.02.2021
Full assembly of the parts to form the complete network
"""

import torch.nn.functional as F

from models.unet_parts import *

""" Unet-model  """
class UNet(nn.Module):
    def __init__(self, model_opts):
        super(UNet, self).__init__()
        self.model_opts = model_opts
        self.n_channels = self.model_opts.n_channels
        self.n_classes = self.model_opts.n_classes
        self.bilinear = self.model_opts.bilinear

        self.inc = DoubleConv(self.n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, self.bilinear)
        self.up2 = Up(512, 256, self.bilinear)
        self.up3 = Up(256, 128, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return torch.sigmoid(x)

    def get_layer_output(self):
        return self._layer_output
