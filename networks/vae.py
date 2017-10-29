from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from networks.conv_stack import ConvolutionStack,TransposedConvolutionStack
import torch

class VAE(nn.Module):
    def __init__(self,encoding_size=100,training=True):
        super(VAE, self).__init__()
        self.training = training
        self.encoding_size = encoding_size
        self.outchannel_size = 64
        # encoding conv
        self.encoder = ConvolutionStack(3)
        self.encoder.append(3,3,2)
        self.encoder.append(16,3,1)
        self.encoder.append(16,3,2)
        self.encoder.append(32,3,1)
        self.encoder.append(32,3,2)
        self.encoder.append(64,3,1)
        self.encoder.append(self.outchannel_size,3,2)

        # take output of conv to pool or fc to mean and std vectors?
        # easiest to start with linear
        # self.fc21 = nn.Linear(400, self.encoding_size)
        # self.fc22 = nn.Linear(400, self.encoding_size)
        # just do average pool in the middle to reduce to cx1x1

        # decode
        self.decoder = TransposedConvolutionStack(self.outchannel_size)
        self.decoder.append(64,3,2)
        self.decoder.append(32,3,1)
        self.decoder.append(32,3,2)
        self.decoder.append(16,3,1)
        self.decoder.append(16,3,2)
        self.decoder.append(3,3,1)
        self.decoder.append(3,3,2)

        # linear layer parameters, lazily instantiated because they depend on the input size
        self.linear_mu = nn.Linear(self.outchannel_size,self.encoding_size)
        self.linear_logvar = nn.Linear(self.outchannel_size,self.encoding_size)
        self.linear_decode = nn.Linear(self.encoding_size,self.outchannel_size)

        # lazily instantiated
        self.pool_size = None

    def encode(self, x):
        input_dims = x.size()
        conv_out = self.encoder.forward(x) 
        self.conv_output_dims = self.encoder.get_output_dims()[:-1][::-1]
        self.conv_output_dims.append(input_dims)
        
        # assume bchw format [1,64,7,7] for inputs of size 100x100
        self.pool_size = conv_out.size(2)

        h1 = F.avg_pool2d(conv_out,kernel_size=self.pool_size,stride=self.pool_size)
        # assert that h1 has dimensions b x c x 1 x 1 (squeeze to b x c)

        # linear op y = x*A_T + b 
        # so here the dims are [b x c] * [c x s], then the weights need to have dims mxc
        # s is the encoding size
        # if self.linear_weights_mu.size()==torch.Size([]):
        #     self.linear_weights_mu = nn.Parameter(torch.randn(self.encoding_size,self.outchannel_size))
        # if self.linear_weights_logvar.size()==torch.Size([]):
        #     self.linear_weights_logvar = nn.Parameter(torch.randn(self.encoding_size,self.outchannel_size))

        # todo, add bias?
        mu = self.linear_mu(h1.view(-1,self.outchannel_size))
        logvar = self.linear_logvar(h1.view(-1,self.outchannel_size))
        # mu = F.linear(h1.view(-1,self.outchannel_size),self.linear_weights_mu)
        # logvar = F.linear(h1.view(-1,self.outchannel_size),self.linear_weights_logvar)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        # first scale up the encodings using a linear layer
        # if self.linear_weights_decoder.size()==torch.Size([]):
        #     self.linear_weights_decoder = nn.Parameter(torch.randn(self.outchannel_size,self.encoding_size))
        #     if self.use_cuda:
        #         self.linear_weights_decoder.cuda()
        
        h2 = self.linear_decode(z)
        # the output dims here should be [b x c] 

        assert self.pool_size is not None
        # next upsample here to dimensions of conv_out from the encoder 
        # TODO, whats the correct thing to do here? unpool, unsample, deconv?
        h3 = F.upsample(h2.view(-1,self.outchannel_size,1,1),scale_factor=self.pool_size) 
        h4 = self.decoder.forward(h3,self.conv_output_dims)
        return F.sigmoid(h4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar