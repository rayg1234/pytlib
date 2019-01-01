import torch
import torch.nn as nn
from torchvision.models.resnet import conv3x3

# extends a basic resnet block with an extra layer mask
class MaskConvBlock(nn.Module):
    def __init__(self, inchans, outchans, stride=1, downsample=None):
        super(MaskConvBlock, self).__init__()
        self.conv1 = conv3x3(inchans, outchans, stride)
        self.bn1 = nn.BatchNorm2d(outchans)
        self.relu = nn.ReLU(inplace=True)
        # output mask layer here
        self.conv2 = conv3x3(outchans, outchans+1)
        self.bn2 = nn.BatchNorm2d(outchans)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, mask):
        identity = mask*x
        out = self.conv1(mask*x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # assume channel dim is 1
        mask_channel = out.shape[1]-1
        new_mask = out[:,mask_channel,:,:].unsqueeze(1)
        # sigmoid the mask?
        # new_mask = torch.sigmoid(new_mask)

        new_out = out[:,0:mask_channel,:,:]

        new_out = self.bn2(new_out)

        if self.downsample is not None:
            identity = self.downsample(x)

        new_out += identity
        new_out = self.relu(new_out)
        return new_out, new_mask
