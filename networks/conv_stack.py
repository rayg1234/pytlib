from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

class ConvolutionStack(nn.Module):
    def __init__(self,in_chans):
        super(ConvolutionStack, self).__init__()
        self.convs = ModuleList()
        self.in_chans = in_chans

    def append(self,out_chans,filter_size,stride):
        if len(self.convs)==0:
            self.convs.append(nn.Conv2d(self.in_chans, out_chans, filter_size, stride=stride, padding=1))
        else:
            self.convs.append(nn.Conv2d(self.convs[-1].out_channels, out_chans, filter_size, stride=stride, padding=1))

    def get_output_dims():
        return self.output_dims

    def forward(self, x):
        self.output_dims = []

        for i,c in enumerate(self.convs):
            x = F.relu(c(x))
            self.output_dims.append(x.size())
        return x

class TransposedConvolutionStack(nn.Module):
    def __init__(self,in_chans):
        super(TransposedConvolutionStack, self).__init__()
        self.convs = ModuleList()
        self.in_chans = in_chans
        self.output_dims = []

    def append(self,out_chans,filter_size,stride):
        if len(self.convs)==0:
            self.convs.append(nn.ConvTranspose2d(self.in_chans, out_chans, filter_size, stride=stride, padding=1))
        else:
            self.convs.append(nn.ConvTranspose2d(self.convs[-1].out_channels, out_chans, filter_size, stride=stride, padding=1))

    def forward(self, x, output_dims=[]):
        for i,c in enumerate(self.convs):
            if output_dims:
                x = F.relu(c(x,output_size=output_dims[i]))
            else:
                x = F.relu(c(x))
        return x
