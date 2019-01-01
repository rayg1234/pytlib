import torch
from networks.mask_block import MaskConvBlock
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import ModuleList

class MaskResnetCNN(nn.Module):
    def __init__(self, block=MaskConvBlock, layers=[3, 4, 23, 3], initchans=3):
        self.initchans = initchans
        self.inplanes = 64
        super(MaskResnetCNN, self).__init__()
        self.all_layers = ModuleList()
        self.all_layers.append(self._make_layer(block, 64, 1, stride=2, inplanes=initchans))
        self.all_layers.append(self._make_layer(block, 64, layers[0], stride=2))
        self.all_layers.append(self._make_layer(block, 128, layers[1], stride=2))
        self.all_layers.append(self._make_layer(block, 256, layers[2], stride=2))
        self.all_layers.append(self._make_layer(block, 512, layers[3], stride=2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, inplanes=None):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes or self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = ModuleList()
        layers.append(block(inplanes or self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

    def forward(self, x):
        mask = torch.ones_like(x)
        all_masks = []
        for blocks in self.all_layers:
            for block in blocks:
                x,mask = block(x, mask)
                all_masks.append(mask)
        return x, all_masks


