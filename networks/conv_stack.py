from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

class ConvolutionStack(nn.Module):
    def __init__(self,in_chans,transposed=False):
        super(ConvolutionStack, self).__init__()
        self.convs = ModuleList()
        self.in_chans = in_chans
        if transposed:
            self.conv_func = nn.Conv2d
        else:
            self.conv_func = nn.ConvTranspose2d

    def append(self,out_chans,filter_size,stride):
        if len(self.convs)==0:
            self.convs.append(self.conv_func(self.in_chans, out_chans, filter_size, stride=stride))
        else:
            self.convs.append(self.conv_func(self.convs[-1].out_channels, out_chans, filter_size, stride=stride))

    # def output_dims(self,input):
    #     self.forward()

    def forward(self, x):
        for c in self.convs:
            x = F.relu(c(x))
        return x