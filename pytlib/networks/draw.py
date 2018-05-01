from torch.autograd import Variable
from networks.basic_rnn import BasicRNN
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class DRAW(nn.Module):
    def __init__(self,q_size=10,
                      encoding_size=128,
                      timesteps=10,
                      training=True,
                      use_attention=False,
                      grid_size=5):
        super(DRAW, self).__init__()
        self.training = training
        self.encoding_size = encoding_size
        self.q_size = q_size
        self.use_attention = use_attention
        self.timesteps = timesteps
        # use equal encoding and decoding size
        self.encoder_rnn = BasicRNN(hstate_size=self.encoding_size,output_size=self.encoding_size)
        self.decoder_rnn = BasicRNN(hstate_size=self.encoding_size,output_size=self.encoding_size)
        self.register_parameter('decoder_linear_weights', None)
        self.register_parameter('encoding_mu_weights', None)
        self.register_parameter('encoding_logvar_weights', None)
        self.filter_linear_layer = nn.Linear(self.encoding_size,5)
        self.grid_size = grid_size

    def initialize(self,x):
        batch_size = x.size(0)
        # we use attention, the decoder producers a patch of grid_size x grid_size
        # else it produces an output of the original image size
        if self.use_attention:
            self.decoder_linear_weights = nn.Parameter(torch.Tensor(self.grid_size*self.grid_size,self.encoding_size))
        else:
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

    # generate two sets of filterbanks 
    # 1) batch x N x W (Fx)
    # 2) batch x N x H (Fy)
    def generate_filter_matrices(self,gx,gy,sigma2,delta):
        N = self.grid_size
        grid_points = torch.arange(0,N).view((1,N,1))
        # gx is Bx1, grid is (1xNx1), so this is a broadcast op -> BxNx1
        mux = gx.view((-1,1,1)) + (grid_points - N/2 - 0.5) * delta 
        muy = gy.view((-1,1,1)) + (grid_points - N/2 - 0.5) * delta
        a = torch.arange(0,self.image_w).view((1,1,-1))
        b = torch.arange(0,self.image_h).view((1,1,-1))
        s2 = sigma2.view((-1,1,1))
        fx = torch.exp(-(a-mux).pow(2)/(2*s2))
        fy = torch.exp(-(b-muy).pow(2)/(2*s2))
        # normalize
        fx = fx/torch.sum(fx,2,keepdim=True)
        fy = fy/torch.sum(fy,2,keepdim=True)
        return fx,fy

    def generate_filter_params(self,state):
        filter_vector = self.filter_linear_layer(state)
        _gx,_gy,log_sigma2,log_delta,loggamma = filter_vector.split(5)
        gx=(self.image_w+1)/2*(_gx+1)
        gy=(self.image_h+1)/2*(_gy+1)
        sigma2=torch.exp(log_sigma2)
        delta=(max(self.image_w,self.image_h)-1)/(self.grid_size-1)*torch.exp(log_delta)       
        gamma=torch.exp(loggamma)
        return gx,gy,sigma2,gamma        

    def read_w_att(self,x,x_hat,dec_state):
        # 1) linear to convert dec_state into batchx5 params gx,gy,logsigma2,logdelta,loggamma
        # 2) convert to gaussian parameters
        gx,gy,sigma2,gamma = generate_filter_params(dec_state)

        # 3) generate filter matrices
        fx,fy = generate_filter_matrices(gx,gy,sigma2,delta)

        # 4) apply filter matrices to get glimpses
        output = gamma.view(-1,1,1)*torch.bmm(torch.bmm(fy,x),torch.transpose(fx,1,2))
        output_hat = gamma.view(-1,1,1)*torch.bmm(torch.bmm(fy,x_hat),torch.transpose(fx,1,2))
        output_total = torch.cat(output,output_hat)
        return output_total

    # write takes use from "encoding space" to image space
    def write(self,decoding):
        return F.linear(decoding,self.decoder_linear_weights)

    def write_w_att(self,decoding):
        write_patch = F.linear(decoding,self.decoder_linear_weights)
        gx,gy,sigma2,gamma = generate_filter_params(decoding)
        fx,fy = generate_filter_matrices(gx,gy,sigma2,delta)
        output = (1/gamma).view(-1,1,1)*torch.bmm(torch.bmm(fy.transpose(1,2),x),fx)
        return output

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
        # assume bchw dims
        self.image_w = x.size(3)
        self.image_h = x.size(2)
        batch_size = x.size()[0]

        if self.decoder_linear_weights is None:
            self.initialize(xview)

        # zero out initial states
        self.encoder_rnn.reset_hidden_state(batch_size,x.data.is_cuda)
        self.decoder_rnn.reset_hidden_state(batch_size,x.data.is_cuda)
        outputs,mus,logvars = [],[],[]

        init_tensor = Variable(torch.zeros(x.size()))
        if x.data.is_cuda:
            init_tensor = init_tensor.cuda()
        outputs.append(init_tensor)

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
            outputs.append(torch.add(outputs[-1],self.write(decoding).view(x.size())))

        # return the sigmoided versions
        for i in range(len(outputs)):
            outputs[i] = F.sigmoid(outputs[i])
        return outputs, mus, logvars

