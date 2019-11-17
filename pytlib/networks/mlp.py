from builtins import range
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

class MLP(nn.Module):
    def __init__(self,depth=1,sizes=[128]):
        super(MLP, self).__init__()
        assert len(sizes)==depth, 'num_layers must match depth!'
        self.depth = depth
        self.sizes = sizes
        self.linear_weights = nn.ParameterList()

    def init_linear_weights(self,input):
        next_size = input.size(1)
        for i in range(0,self.depth):
            linear_weight = nn.Parameter(torch.Tensor(self.sizes[i],next_size))
            next_size = self.sizes[i]
            stdv = 1. / math.sqrt(linear_weight.size(1))
            linear_weight.data.uniform_(-stdv, stdv)
            self.linear_weights.append(linear_weight)
        if input.data.is_cuda:
            self.cuda()

    def forward(self, x):
        if self.linear_weights is None:
            self.init_linear_weights(x)

        for i in range(0,self.depth):
            x = F.linear(x,self.linear_weights[i])
        return x