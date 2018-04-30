from torch.autograd import Variable
from networks.basic_rnn import BasicRNN
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class DRAW(nn.Module):
    def __init__(self,q_size=10,encoding_size=128,timesteps=10,training=True,use_attention=False):
        super(DRAW, self).__init__()
        self.training = training
        self.encoding_size = encoding_size
        self.q_size = q_size
        self.use_attention = use_attention
        self.timesteps = timesteps
        # use equal encoding and decoding size
        self.encoder_rnn = BasicRNN(output_size=self.encoding_size)
        self.decoder_rnn = BasicRNN(output_size=self.encoding_size)
        self.register_parameter('decoder_linear_weights', None)
        self.register_parameter('encoding_mu_weights', None)
        self.register_parameter('encoding_logvar_weights', None)

    def initialize(self,x):
        batch_size = x.size(0)
        self.decoder_linear_weights = nn.Parameter(torch.Tensor(x.nelement()/batch_size,self.encoding_size))
        stdv = 1. / math.sqrt(self.decoder_linear_weights.size(1))
        self.decoder_linear_weights.data.uniform_(-stdv, stdv)        
        
        self.encoding_mu_weights = nn.Parameter(torch.Tensor(self.q_size,self.encoding_size))
        stdv = 1. / math.sqrt(self.encoding_mu_weights.size(1))
        self.encoding_mu_weights.data.uniform_(-stdv, stdv)   

        self.encoding_logvar_weights = nn.Parameter(torch.Tensor(self.q_size,self.encoding_size))
        stdv = 1. / math.sqrt(self.encoding_logvar_weights.size(1))
        self.encoding_logvar_weights.data.uniform_(-stdv, stdv)   
        if x.data.is_cuda:
            self.cuda()

    # selects where to sample from the input image, no attention version
    # dims is 2*W*H
    def read(self,x,x_hat,dec_state):
        return torch.cat((x,x_hat),1)

    # write takes use from "encoding space" to image space
    def write(self,decoding):
        return F.linear(decoding,self.decoder_linear_weights)

    # this converts the encoding into both a mu and logvar vector
    def sampleZ(self,encoding):
        mu = F.linear(encoding,self.encoding_mu_weights)
        logvar = F.linear(encoding,self.encoding_logvar_weights)
        return self.reparameterize(mu, logvar),mu,logvar

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    # takes an input, returns the sequence of outputs, mus, and logvars
    def forward(self,x):
        # flatten x to 1-d, except for batch dimension
        xview = x.view(x.size()[0],x.nelement()/x.size()[0])
        batch_size = x.size()[0]

        if self.decoder_linear_weights is None:
            self.initialize(xview)

        # zero out initial states
        self.encoder_rnn.reset_hidden_state(batch_size)
        self.decoder_rnn.reset_hidden_state(batch_size)
        outputs,mus,logvars = [],[],[]

        outputs.append(Variable(torch.zeros(x.size())))

        for t in range(0,self.timesteps):
            # import ipdb;ipdb.set_trace()
            # Step 1: diff the input against the prev output
            x_hat = xview - F.sigmoid(outputs[t].view(xview.size()))
            # Step 2: read
            rvec = self.read(xview,x_hat,self.decoder_rnn.get_hidden_state())
            # Step 3: encoder rnn
            # note the dimensions of r doesn't have to match with the decoding size because
            # we are just concating 2 dim-1 tensors, which is kind of wierd, but ok...
            cat = torch.cat((rvec,self.decoder_rnn.get_hidden_state().view(batch_size,self.encoding_size)),1)
            encoding = self.encoder_rnn.forward(cat)
            # Step 4: sample z
            z,mu,logvar = self.sampleZ(encoding)
            # store the mu and logvar for the loss function
            mus.append(mu)
            logvars.append(logvar)

            # Step 5: decoder rnn
            decoding = self.decoder_rnn.forward(z)
            # Step 6: write to canvas, (in the original dimensions of the input)
            outputs.append(torch.add(outputs[-1],F.sigmoid(self.write(decoding).view(x.size()))))

        return outputs, mus, logvars

