import torch
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.autograd import Variable

# basic RNN module, contains a hidden state
# each time forward is called, the hidden state is concatenated with the input
# and a fc layer computes both an output and the new hidden_state
# the basic rnn update equation
# we can do plus or concat here
# h_{t+1} = f([U*x_t,V*h_t])
# o_{t+1} = W*h_{t+1}
# f is some non-linearity (ie: tanh)
# this module's output is the same as the input
class BasicRNN(nn.Module):
    def __init__(self,hstate_size=128,output_size=128):
        super(BasicRNN, self).__init__()
        self.hstate_size = hstate_size
        self.output_size = output_size
        self.register_parameter('U', None)
        self.register_parameter('V', None)
        self.register_parameter('W', None)
        # hidden state, initialized to 0?
        self.hidden_state = nn.Variable(torch.Zeros(self.hstate_size))

    def get_hidden_state(self):
        return self.hidden_state

    # zero the hidden states
    def reset(self):
        self.hidden_state = nn.Variable(torch.Zeros(self.hstate_size))

    def init_weights(self,input):
        self.U = nn.Parameter(torch.Tensor(self.hstate_size,input.size(1)))
        stdv = 1. / math.sqrt(self.U.size(1))
        self.U.data.uniform_(-stdv, stdv)

        self.V = nn.Parameter(torch.Tensor(output_size,self.hstate_size))
        stdv = 1. / math.sqrt(self.V.size(1))
        self.V.data.uniform_(-stdv, stdv)  

        self.W = nn.Parameter(torch.Tensor(self.hstate_size,self.hstate_size))
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)

        self.cuda()

    def forward(self, x):
        if self.U is None:
            self.init_weights(x)
        # assert the dimensions are correct here
        cat = torch.cat((F.linear(x,self.U),F.linear(self.hidden_state,self.V)),0)
        self.hidden_state = F.tanh(cat)
        return F.linear(self.hidden_state,self.W)       
