import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

class ConvolutionStack(nn.Module):
    def __init__(self,in_chans,final_relu=True,padding=1):
        super(ConvolutionStack, self).__init__()
        self.convs = ModuleList()
        self.batchnorms = ModuleList()
        self.in_chans = in_chans
        self.final_relu = final_relu
        self.padding = padding

    def append(self,out_chans,filter_size,stride):
        if len(self.convs)==0:
            self.convs.append(nn.Conv2d(self.in_chans, out_chans, filter_size, stride=stride, padding=self.padding))
        else:
            self.convs.append(nn.Conv2d(self.convs[-1].out_channels, out_chans, filter_size, stride=stride, padding=self.padding))
        self.batchnorms.append(nn.BatchNorm2d(out_chans))

    def get_output_dims(self):
        return self.output_dims

    def forward(self, x):
        self.output_dims = []

        for i,c in enumerate(self.convs):
            # lrelu = nn.LeakyReLU(0.2,inplace=True)
            # x = lrelu(c(x))
            x = c(x)
            x = self.batchnorms[i](x)
            if i<len(self.convs)-1 or self.final_relu:
                x = F.relu(x)
            self.output_dims.append(x.size())
        return x

class TransposedConvolutionStack(nn.Module):
    def __init__(self,in_chans,final_relu=True,padding=1):
        super(TransposedConvolutionStack, self).__init__()
        self.convs = ModuleList()
        self.batchnorms = ModuleList()
        self.in_chans = in_chans
        self.output_dims = []
        self.final_relu = final_relu
        self.padding = padding

    def append(self,out_chans,filter_size,stride):
        if len(self.convs)==0:
            self.convs.append(nn.ConvTranspose2d(self.in_chans, out_chans, filter_size, stride=stride, padding=self.padding))
        else:
            self.convs.append(nn.ConvTranspose2d(self.convs[-1].out_channels, out_chans, filter_size, stride=stride, padding=self.padding))
        self.batchnorms.append(nn.BatchNorm2d(out_chans))

    def forward(self, x, output_dims=[]):
        # print self.convs
        for i,c in enumerate(self.convs):
            x = c(x,output_size=output_dims[i]) if output_dims else c(x)
            x = self.batchnorms[i](x)
            if i<len(self.convs)-1 or self.final_relu:
                x = F.relu(x)
        return x
