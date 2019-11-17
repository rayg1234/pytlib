from __future__ import division
from builtins import range
from past.utils import old_div
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from networks.basic_rnn import BasicRNN
from networks.conv_stack import ConvolutionStack,TransposedConvolutionStack
from networks.gaussian_attention_sampler import GaussianAttentionReader,GaussianAttentionWriter
from visualization.image_visualizer import ImageVisualizer
from image.ptimage import PTImage

class AttentionSegmenter(nn.Module):
    def __init__(self,num_classes,inchans=3,att_encoding_size=128,timesteps=10,attn_grid_size=50):
        super(AttentionSegmenter, self).__init__()
        self.num_classes = num_classes
        self.att_encoding_size = att_encoding_size
        self.timesteps = timesteps
        self.attn_grid_size = attn_grid_size
        self.encoder = ConvolutionStack(inchans,final_relu=False,padding=0)
        self.encoder.append(32,3,1)
        self.encoder.append(32,3,2)
        self.encoder.append(64,3,1)
        self.encoder.append(64,3,2)
        self.encoder.append(96,3,1)
        self.encoder.append(96,3,2)

        self.decoder = TransposedConvolutionStack(96,final_relu=False,padding=0)
        self.decoder.append(96,3,2)
        self.decoder.append(64,3,1)
        self.decoder.append(64,3,2)
        self.decoder.append(32,3,1)
        self.decoder.append(32,3,2)
        self.decoder.append(self.num_classes,3,1)

        self.attn_reader = GaussianAttentionReader()
        self.attn_writer = GaussianAttentionWriter()
        self.att_rnn = BasicRNN(hstate_size=att_encoding_size,output_size=5)
        self.register_parameter('att_decoder_weights', None)

    def init_weights(self,hstate):
        if self.att_decoder_weights is None:
            batch_size = hstate.size(0)
            self.att_decoder_weights = nn.Parameter(torch.Tensor(5,old_div(hstate.nelement(),batch_size)))
            stdv = 1. / math.sqrt(self.att_decoder_weights.size(1))
            self.att_decoder_weights.data.uniform_(-stdv, stdv)
        if hstate.data.is_cuda:
            self.cuda()

    def forward(self, x):
        batch_size,chans,height,width = x.size()

        # need to first determine the hidden state size, which is tied to the cnn feature size
        dummy_glimpse = torch.Tensor(batch_size,chans,self.attn_grid_size,self.attn_grid_size)
        if x.is_cuda:
            dummy_glimpse = dummy_glimpse.cuda()
        dummy_feature_map = self.encoder.forward(dummy_glimpse)
        self.att_rnn.forward(dummy_feature_map.view(batch_size,old_div(dummy_feature_map.nelement(),batch_size)))
        self.att_rnn.reset_hidden_state(batch_size,x.data.is_cuda)

        outputs = []
        init_tensor = torch.zeros(batch_size,self.num_classes,height,width)
        if x.data.is_cuda:
            init_tensor = init_tensor.cuda()
        outputs.append(init_tensor) 

        self.init_weights(self.att_rnn.get_hidden_state())

        for t in range(self.timesteps):
            # 1) decode hidden state to generate gaussian attention parameters
            state = self.att_rnn.get_hidden_state()
            gauss_attn_params = F.tanh(F.linear(state,self.att_decoder_weights))

            # 2) extract glimpse
            glimpse = self.attn_reader.forward(x,gauss_attn_params,self.attn_grid_size)

            # visualize first glimpse in batch for all t
            torch_glimpses = torch.chunk(glimpse,batch_size,dim=0)
            ImageVisualizer().set_image(PTImage.from_cwh_torch(torch_glimpses[0].squeeze().data),'zGlimpse {}'.format(t))            

            # 3) use conv stack or resnet to extract features
            feature_map = self.encoder.forward(glimpse)
            conv_output_dims = self.encoder.get_output_dims()[:-1][::-1]
            conv_output_dims.append(glimpse.size())
            # import ipdb;ipdb.set_trace()

            # 4) update hidden state # think about this connection a bit more
            self.att_rnn.forward(feature_map.view(batch_size,old_div(feature_map.nelement(),batch_size)))

            # 5) use deconv network to get partial masks
            partial_mask = self.decoder.forward(feature_map,conv_output_dims)

            # 6) write masks additively to mask canvas
            partial_canvas = self.attn_writer.forward(partial_mask,gauss_attn_params,(height,width))
            outputs.append(torch.add(outputs[-1],partial_canvas))

                # return the sigmoided versions
        for i in range(len(outputs)):
            outputs[i] = torch.sigmoid(outputs[i])
        return outputs