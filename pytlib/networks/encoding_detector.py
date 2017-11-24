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
        crop_feature_map = self.vae.get_encoding_feature_map()
        frame_feature_map = self.vae.get_encoder().forward(frame)

        # now compute the convolution of the frame_feature_map against the crop_feature map
        return recon,mu,logvar
