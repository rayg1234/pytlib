import torch
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from utils.debug import pp

class MLP(nn.Module):
    def __init__(self,depth=1,sizes=[128]):
        super(MLP, self).__init__()
        assert len(sizes)==depth, 'num_layers must match depth!'
        self.depth = depth
        self.sizes = sizes
        self.linear_weights = None

    def init_linear_weights(self,input):
        self.linear_weights = []
        next_size = input.size(1)
        for i in range(0,self.depth):
            linear_weight = nn.Parameter(torch.Tensor(self.sizes[i],next_size))
            next_size = self.sizes[i]
            stdv = 1. / math.sqrt(linear_weight.size(1))
            linear_weight.data.uniform_(-stdv, stdv)
            linear_weight = linear_weight.cuda() if input.is_cuda else linear_weight
            self.linear_weights.append(linear_weight)

    def forward(self, x):
        if self.linear_weights is None:
            self.init_linear_weights(x)

        for i in range(0,self.depth):
            x = F.linear(x,self.linear_weights[i])
        return x