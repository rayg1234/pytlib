import torch
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.autograd import Variable
from networks.resnetcnn import ResNetCNN
from networks.mlp import MLP
import numpy as np

class BaseMonoDepthEstimator(nn.Module):
    def __init__(self):
        super(BaseMonoDepthEstimator, self).__init__()
        # create 2 networks
        # 1) a depth network - encoder-decoder combo
        self.encoder = ResNetCNN()
        self.decoder = None #placeholder
        # 2) an ego motion network - use the encoder from (1)
        # with an additional FC layer
        # mlp for decoding ego motion, placeholder
        self.ego_motion_decoder = None


    def forward(self, x):   	
        # for each frame, run through the depth and ego motion networks
        # assume input is BxKxCxWxH
        assert len(x.shape)==5, 'Input must be BxKxCxWxH!'
        batch_size = x.shape[0]
        nframes = x.shape[1]
        unstacked_frames = torch.chunk(x,nframes,1)
        unstacked_frames = [torch.squeeze(x,1) for x in unstacked_frames]
        ego_vectors = []
        depth_maps = []
        for frame in unstacked_frames:
        	features = self.encoder.forward(frame)
        	# pass through, replace with actual decoders
        	ego_vector = features
        	depth_map = features
        	ego_vectors.append(ego_vector)
        	depth_maps.append(depth_map)
        # restack these and return
        return torch.stack(ego_vectors,1), torch.stack(depth_maps,1)
