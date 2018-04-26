from torch.autograd import Variable
from networks.basic_rnn import BasicRNN
import torch.nn as nn
import torch.nn.functional as F
import torch

class DRAW(nn.Module):
    def __init__(self,encoding_size=128,timesteps=10,training=True,use_attention=False):
        super(VAE, self).__init__()
        self.training = training
        self.encoding_size = encoding_size
        self.use_attention = use_attention
        self.timesteps = timesteps
        self.outputs,self.mus,self.logvars = [],[],[]
        # use equal encoding and decoding size
        self.encoder_rnn = BasicRNN(output_size=self.encoding_size)
        self.decoder_rnn = BasicRNN(output_size=self.encoding_size)
        self.register_parameter('decoder_linear_weights', None)
        self.initialized = False

    def initialize(self,x):
        pass

    def zero_states(self):
        # initialize the first output as zero tensor with same size as input (blank canvas)
        # the initial hidden states should be zero as well
        self.encoder_rnn.reset()
        self.decoder_rnn.reset()
        self.outputs.append(torch.Zeros(self.input_size))

    # selects where to sample from the input image, no attention version
    # dims is 2*W*H
    def read(self,x,x_hat,dec_state):
        return torch.cat((x,x_hat),0)

    def write(self,decoding)
        return F.linear(decoding,self.decoder_linear_weights)

    def sampleZ(self,encoding):
        pass

    # takes an input, returns the sequence of outputs, mus, and logvars
    def forward(self,x):
        # flatten x to 1-d
        xview = x.view(x.nelement())
        self.input_size = x.nelement()

        if not self.initialized:
            self.initialize(xview)

        # zero out initial states
        self.zero_states()

        for t in range(0,self.timesteps):
            # Step 1: diff the input against the prev output
            x_hat = xview - F.sigmoid(self.outputs[t])
            # Step 2: read
            r = self.read(xview,x_hat,self.dec_state)
            # Step 3: encoder rnn
            # note the dimensions of r doesn't have to match with the decoding size because
            # we are just concating 2 dim-1 tensors, which is kind of wierd, but ok...
            encoding = self.encoder_rnn.foward(torch.cat((r,self.decoder_rnn.get_hidden_state()), 0))
            # Step 4: sample z
            z = sampleZ(encoding)
            # Step 5: decoder rnn
            decoding = self.decoder_rnn.foward(z)
            # Step 6: write to canvas
            self.outputs[t] = torch.sum(self.outputs[t],self.write(decoding))
        return self.outputs, self.mus, self.logvars

