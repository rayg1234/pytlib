import torch
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.autograd import Variable
from networks import ResNetCNN

class MultiObjectDetector(nn.Module):
    def __init__(self):
        super(MultiObjectDetector, self).__init__()
        self.feature_map_generator = ResNetCNN()

    def forward(self, x):
        # CNN Compute
        feature_maps = self.feature_map_generator.forward(x)

        # Detector head compute
        # Simply use 1x1 convolution to regress down to HxWx(Nx5) result
        # where N is the number of boxes per super pixel
