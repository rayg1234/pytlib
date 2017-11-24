from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from networks.conv_stack import ConvolutionStack,TransposedConvolutionStack
from networks.vae import VAE
import torch


# both encodes the image and performs detection on a target box(s)
class EncodingDetector(nn.Module):
    def __init__(self):
        super(EncodingDetector, self).__init__()
        self.vae = VAE()

    # assert the input has two elements, first is the crop, second the full frame
    def forward(self, x):
        assert len(x)==2,'EncodingDetector input must have 2 elements'
        # compute the feature map outputs for both the crop and full
        crop = x[0]
        frame = x[1]
        recon,mu,logvar = self.vae.forward(crop)
        frame_feature_map = self.vae.get_encoder().forward(frame)


        # TODO: this is a dumb way to get the output dims for the deconv
        output_dims = self.convs.get_output_dims()[:-1][::-1]
        output_dims.append(input_dims)
        # print output_dims
        # get outputs from conv and pass them back to deconv
        x = self.tconvs.forward(x,output_dims)
        return F.sigmoid(x)
