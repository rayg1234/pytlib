# full assembly of the sub-parts to form the complete net
# code from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

import torch.nn.functional as F
import torch.nn as nn
from networks.unet_parts import up, down, inconv, outconv

# shallow unet 
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 128)
        self.up1 = up(256, 64)
        self.up2 = up(128, 32)
        self.up3 = up(64, 32)
        self.outc = outconv(32, n_classes)

    def feature_channels(self):
        return 128

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)        
        return x, x4