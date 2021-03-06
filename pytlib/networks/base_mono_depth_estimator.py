import torch
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.autograd import Variable
from networks.resnetcnn import ResNetCNN
from networks.unet import UNet
from networks.conv_stack import TransposedConvolutionStack, ConvolutionStack
import numpy as np

class BaseMonoDepthEstimator(nn.Module):
    def __init__(self, inchans=3,nframes=3):
        super(BaseMonoDepthEstimator, self).__init__()
        # 3 parameters for translation and 3 for rotation
        self.inchans = inchans
        self.ego_vector_dim = 6
        self.nframes = nframes
        self.ego_prediction_size = self.ego_vector_dim * (self.nframes -1)

        # Depth network - encoder-decoder combo
        self.unet = UNet(3,1)
        self.final_conv_layer0 = nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.final_conv_layer1 = nn.Conv2d(1, 1, 1, stride=1, padding=0)

        # 2) an ego motion network - use the encoder from (1)
        # and append extra cnns
        self.ego_motion_cnn = ConvolutionStack(self.unet.feature_channels()*nframes)
        self.ego_motion_cnn.append(128,3,2)
        self.ego_motion_cnn.append(64,3,2)
        self.ego_motion_cnn.append(self.ego_prediction_size,1,1)


    def forward(self, x):       
        # for each frame, run through the depth and ego motion networks
        # assume input is BxKxCxWxH
        assert len(x.shape)==5, 'Input must be BxKxCxWxH!'
        assert x.shape[1]==self.nframes, 'Input sequence length must match nframes!, expected {}, found {}'.format(self.nframes, x.shape[1])

        batch_size = x.shape[0]
        unstacked_frames = torch.chunk(x,self.nframes,1)
        unstacked_frames = [torch.squeeze(y,1) for y in unstacked_frames]
        encoded_features = []
        depth_maps = []

        # first predict depth in all frames
        for frame in unstacked_frames:
            depth_map, features = self.unet.forward(frame)
            encoded_features.append(features)
            # depth_map = self.final_conv_layer0(depth_map)
            # depth_map = self.final_conv_layer1(depth_map)
            depth_map = F.sigmoid(depth_map.squeeze(1))
            depth_maps.append(depth_map)

        # next predict ego_motion, stack feature maps
        stacked_features = torch.cat(encoded_features,1)
        ego_motion_features = self.ego_motion_cnn(stacked_features)
        # reduce mean along the spatial dimensions
        ego_vectors = torch.mean(ego_motion_features,[2,3]).reshape(batch_size,self.nframes-1,-1)
        depth_maps = torch.stack(depth_maps,1)

        # frames, ego_vectors, and depth_maps
        return x, ego_vectors, depth_maps
