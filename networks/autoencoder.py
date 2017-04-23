from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from networks.conv_stack import ConvolutionStack

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # conv, deconv
        self.convs = ConvolutionStack(3)
        self.convs.append(6,3,2)
        self.convs.append(16,3,1)
        self.convs.append(32,3,2)

        # get the output width height here

        self.deconvs = ConvolutionStack(32,transposed=True)
        self.deconvs.append(16,3,2)
        self.deconvs.append(6,3,1)
        self.deconvs.append(3,3,2)

    def forward(self, x):
        x = self.convs.forward(x)
        x = self.deconvs.forward(x)
        return x