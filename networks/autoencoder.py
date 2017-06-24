from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from networks.conv_stack import ConvolutionStack,TransposedConvolutionStack
import torch

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # conv, deconv
        torch.manual_seed(123)
        self.convs = ConvolutionStack(3)
        self.convs.append(3,3,2)
        self.convs.append(6,3,1)
        self.convs.append(16,3,1)
        self.convs.append(32,3,2)

        # get the output width height here

        self.tconvs = TransposedConvolutionStack(32)
        self.tconvs.append(16,3,2)
        self.tconvs.append(6,3,1)
        self.tconvs.append(3,3,1)
        self.tconvs.append(3,3,2)

    def forward(self, x):
        input_dims = x.size()
        x = self.convs.forward(x)
        output_dims = self.convs.get_output_dims()[:-1][::-1]
        output_dims.append(input_dims)
        # print output_dims
        # get outputs from conv and pass them back to deconv
        x = self.tconvs.forward(x,output_dims)
        return x

    def get_encoding(self):
        return self.encoding