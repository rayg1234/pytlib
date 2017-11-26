from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from networks.conv_stack import ConvolutionStack,TransposedConvolutionStack
import torch
import math

class VAE(nn.Module):
    def __init__(self,encoding_size=128,training=True):
        super(VAE, self).__init__()
        self.training = training
        self.encoding_size = encoding_size
        self.outchannel_size = 256
        # encoding conv
        self.encoder = ConvolutionStack(3)
        self.encoder.append(3,3,2)
        self.encoder.append(16,3,2)
        self.encoder.append(64,3,2)
        self.encoder.append(128,3,2)
        self.encoder.append(self.outchannel_size,3,2)

        # decode
        self.decoder = TransposedConvolutionStack(self.outchannel_size,final_relu=False)
        self.decoder.append(128,3,2)
        self.decoder.append(64,3,2)
        self.decoder.append(16,3,2)
        self.decoder.append(3,3,2)
        self.decoder.append(3,3,2)

        self.linear_mu_weights = nn.Parameter()
        self.linear_logvar_weights = nn.Parameter()
        self.linear_decode_weights = nn.Parameter()

        # # lazily instantiated
        # self.pool_size = None

    def initialize_linear_params(self,cuda):
        # linear op y = x*A_T + b 
        # so here the dims are [b x c] * [c x s], then the weights need to have dims (s x c)
        # where s is the encoding size and b is the batch size

        self.linear_mu_weights = nn.Parameter(torch.Tensor(self.encoding_size,self.linear_size))
        stdv = 1. / math.sqrt(self.linear_mu_weights.size(1))
        self.linear_mu_weights.data.uniform_(-stdv, stdv)

        self.linear_logvar_weights = nn.Parameter(torch.Tensor(self.encoding_size,self.linear_size))
        self.linear_logvar_weights.data.uniform_(-stdv, stdv)
        self.linear_decode_weights = nn.Parameter(torch.Tensor(self.linear_size,self.encoding_size)) 
        self.linear_decode_weights.data.uniform_(-stdv, stdv)
        if cuda:
            self.linear_mu_weights.cuda()
            self.linear_logvar_weights.cuda()
            self.linear_decode_weights.cuda() 

    def encode(self, x):
        input_dims = x.size()
        conv_out = self.encoder.forward(x)
        self.encoding_feature_map = conv_out
        self.conv_output_dims = self.encoder.get_output_dims()[:-1][::-1]
        self.conv_output_dims.append(input_dims)
        
        # print conv_out.size()
        # OPTION A -- AVERAGE POOL -> FC
        # assume bchw format [1,C,7,7] for inputs of size 100x100
        # self.pool_size = conv_out.size(2)
        # h1 = F.avg_pool2d(conv_out,kernel_size=self.pool_size,stride=self.pool_size)
        # assert that h1 has dimensions b x c x 1 x 1 (squeeze to b x c)

        # OPTION B -- DIRECT FC
        self.conv_out_spatial = [conv_out.size(2),conv_out.size(3)]
        self.linear_size = self.outchannel_size*conv_out.size(2)*conv_out.size(3)

        if self.linear_mu_weights.size()==torch.Size([]):
            self.initialize_linear_params(x.data.is_cuda)

        mu = F.linear(conv_out.view(-1,self.linear_size),self.linear_mu_weights)
        logvar = F.linear(conv_out.view(-1,self.linear_size),self.linear_logvar_weights)
        # mu = self.linear_mu(conv_out.view(-1,linear_size))
        # logvar = self.linear_logvar(conv_out.view(-1,linear_size))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        # h2 = F.relu(self.linear_decode(z))
        # the output dims here should be [b x c] 

        # OPTION A -- upsample
        # assert self.pool_size is not None
        # next upsample here to dimensions of conv_out from the encoder 
        # TODO, whats the correct thing to do here? unpool, unsample, deconv?

        # h3 = F.upsample(h2.view(-1,self.outchannel_size,1,1),scale_factor=self.pool_size) 
        # OPTION B -- Direct FC
        if self.linear_decode_weights.size()==torch.Size([]):
            self.initialize_linear_params(z.data.is_cuda)

        h2 = F.relu(F.linear(z,self.linear_decode_weights))

        h3 = h2.view(-1,self.outchannel_size,self.conv_out_spatial[0],self.conv_out_spatial[1])
        h4 = self.decoder.forward(h3,self.conv_output_dims)
        return F.sigmoid(h4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_encoder(self):
        return self.encoder

    def get_encoding_feature_map(self):
        return self.encoding_feature_map