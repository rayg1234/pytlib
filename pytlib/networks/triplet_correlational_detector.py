from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from networks.conv_stack import ConvolutionStack
import torch
import math
from data_loading.sample import Sample

# both encodes the image and performs detection on a target box(s)
class TripletCorrelationalDetector(nn.Module):
    def __init__(self):
        super(TripletCorrelationalDetector, self).__init__()
        self.encoder = ConvolutionStack(3,final_relu=False)
        self.encoder.append(3,3,2)
        self.encoder.append(16,3,2)
        self.encoder.append(32,3,1)
        self.encoder.append(64,3,2)
        self.encoder.append(96,3,1)
        self.crosscor_batchnorm = nn.BatchNorm2d(1)
        self.register_parameter('feature_crop', None)

    # now compute the xcorrelation of these feature maps
    # need to compute these unbatched because we are not using the same filter map for each conv
    def cross_correlation(self,x1,x2,bn=None):
        batch_size = x1.size(0)
        response_maps = []
        for i in range(0,batch_size):
            response = F.conv2d(x1[i,:].unsqueeze(0),x2[i,:].unsqueeze(0))
            response_maps.append(response.squeeze(0))
        rmap = torch.stack(response_maps,0)
        if bn is not None:
            rmap = bn(rmap)
        return rmap

    def forward(self, pos, neg):
        pos_feature_map = self.encoder.forward(pos)
        neg_feature_map = self.encoder.forward(neg)
        # initialize feature_encoding if None
        batch_size = pos.size(0)
        if self.feature_crop is None:
            self.feature_crop = nn.Parameter(torch.Tensor(pos.size()[1:]))
            stdv = 1. / math.sqrt(self.feature_crop.nelement())
            self.feature_crop.data.uniform_(-stdv, stdv)

        crop = self.feature_crop.cuda() if pos.is_cuda else self.feature_crop
        anchor_feature_map = self.encoder.forward(crop.expand(batch_size,*self.feature_crop.size()))
        return anchor_feature_map,pos_feature_map,neg_feature_map

    def infer(self,frame,pos):
        batch_size = frame.size(0)
        if self.feature_crop is None:
            self.feature_crop = nn.Parameter(torch.Tensor(pos.size()[1:]))
        crop = self.feature_crop.cuda() if pos.is_cuda else self.feature_crop
        crop_features = self.encoder.forward(crop.expand(batch_size,*self.feature_crop.size()))
        frame_features = self.encoder.forward(frame)
        pos_features = self.encoder.forward(pos)
        return self.cross_correlation(frame_features,crop_features)