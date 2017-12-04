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
        self.encoder.append(32,3,2)
        self.encoder.append(64,3,2)
        self.encoder.append(96,3,2)
        self.crosscor_batchnorm = nn.BatchNorm2d(1)

    # now compute the xcorrelation of these feature maps
    # need to compute these unbatched because we are not using the same filter map for each conv
    def cross_correlation(self,x1,x2,bn):
        batch_size = x1.size(0)
        response_maps = []
        for i in range(0,batch_size):
            response = F.conv2d(x1[i,:].unsqueeze(0),x2[i,:].unsqueeze(0))
            response_maps.append(response.squeeze(0))
        rmap = torch.stack(response_maps,0)
        if bn is not None:
            rmap = bn(rmap)
        return rmap

    # assert the input has two elements, first is the crop, second the full frame
    def forward(self, anchor, pos, neg):
        # recon,mu,logvar = None,None,None
        anchor_feature_map = self.encoder.forward(anchor)
        pos_feature_map = self.encoder.forward(pos)
        neg_feature_map = self.encoder.forward(neg)
        # anchor_pos_xcor = self.cross_correlation(anchor_feature_map,pos_feature_map,self.crosscor_batchnorm1)
        # anchor_neg_xcor = self.cross_correlation(anchor_feature_map,neg_feature_map,self.crosscor_batchnorm2)
        # frame_feature_map = self.encoder.forward(frame)
        # frame_pos_xcor = self.cross_correlation(frame_feature_map,pos_feature_map,self.crosscor_batchnorm)
        return anchor_feature_map,pos_feature_map,neg_feature_map
