from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from utils.debug import pp

class ConvolutionStack(nn.Module):
    def __init__(self,in_chans):
        super(ConvolutionStack, self).__init__()
        self.convs = ModuleList()
        self.batchnorms = ModuleList()
        self.in_chans = in_chans

    def append(self,out_chans,filter_size,stride):
        if len(self.convs)==0:
            self.convs.append(nn.Conv2d(self.in_chans, out_chans, filter_size, stride=stride, padding=1))
        else:
            self.convs.append(nn.Conv2d(self.convs[-1].out_channels, out_chans, filter_size, stride=stride, padding=1))
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
            x = F.relu(x)
            self.output_dims.append(x.size())
        return x

class TransposedConvolutionStack(nn.Module):
    def __init__(self,in_chans,final_nonlinearity='sigmoid'):
        super(TransposedConvolutionStack, self).__init__()
        self.convs = ModuleList()
        self.batchnorms = ModuleList()
        self.in_chans = in_chans
        self.output_dims = []
        self.final_nonlinearity = final_nonlinearity

    def append(self,out_chans,filter_size,stride):
        if len(self.convs)==0:
            self.convs.append(nn.ConvTranspose2d(self.in_chans, out_chans, filter_size, stride=stride, padding=1))
        else:
            self.convs.append(nn.ConvTranspose2d(self.convs[-1].out_channels, out_chans, filter_size, stride=stride, padding=1))
        self.batchnorms.append(nn.BatchNorm2d(out_chans))

    def forward(self, x, output_dims=[]):
        # print self.convs
        for i,c in enumerate(self.convs):
            # pp(x.mean(),'input x: mean with dims {0}'.format(x.size()))
            x = c(x,output_size=output_dims[i])
            x = self.batchnorms[i](x)
            if(i==len(self.convs)-1):
                if self.final_nonlinearity=='sigmoid':
                    x = F.sigmoid(x)
                else:
                    x = F.relu(x)
            else:
                x = F.relu(x)
            # pp(x.mean(),'output x: mean with dims {0}'.format(x.size()))
        return x
