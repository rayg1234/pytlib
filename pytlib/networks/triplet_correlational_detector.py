from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from networks.conv_stack import ConvolutionStack
import torch
import math
from data_loading.sample import Sample

# both encodes the image and performs detection on a target box(s)
class TripletCorrelationalDetector(nn.Module):
    def __init__(self,anchor_size=(3,127,127)):
        super(TripletCorrelationalDetector, self).__init__()
        self.encoder = ConvolutionStack(3,final_relu=False,padding=0)
        self.encoder.append(3,3,2)
        self.encoder.append(16,3,2)
        self.encoder.append(32,3,1)
        self.encoder.append(64,3,2)
        self.encoder.append(128,3,1)
        self.crosscor_batchnorm0 = nn.BatchNorm2d(1)
        # this used to dynamically initialized to deal with runtime cropsizes
        # I will keep it this way incase I want to revisit        
        self.register_parameter('anchor_crop', None)
        self.anchor_size = torch.Size(anchor_size)

    # now compute the xcorrelation of these feature maps
    # need to compute these unbatched because we are not using the same filter map for each conv
    def cross_correlation(self,x1,x2,bn=None,padding=0):
        batch_size = x1.size(0)
        response_maps = []
        for i in range(0,batch_size):
            response = F.conv2d(x1[i,:].unsqueeze(0),x2[i,:].unsqueeze(0),padding=padding)
            response_maps.append(response.squeeze(0))
        rmap = torch.stack(response_maps,0)
        rmap = bn(rmap) if bn is not None else rmap
        return rmap

    def init_anchor(self,is_cuda):
        if self.anchor_crop is None:
            self.anchor_crop = nn.Parameter(torch.Tensor(self.anchor_size))
            stdv = 1. / math.sqrt(self.anchor_crop.nelement())
            self.anchor_crop.data.uniform_(-stdv, stdv)
            if is_cuda:
                self.cuda()        

    def forward(self, pos, neg):
        pos_feature_map = self.encoder.forward(pos)
        neg_feature_map = self.encoder.forward(neg)
        # initialize feature_encoding if None
        batch_size = pos.size(0)
        self.init_anchor(pos.is_cuda)
        anchor = self.anchor_crop.expand(batch_size,*self.anchor_crop.size())
        anchor_feature_map = self.encoder.forward(anchor)

        cxp = self.cross_correlation(pos_feature_map,anchor_feature_map)
        cxn = self.cross_correlation(neg_feature_map,anchor_feature_map)
        return anchor,cxp,cxn

    def infer(self,frame):
        batch_size = frame.size(0)
        self.init_anchor(frame.is_cuda)
        batched_crop = self.anchor_crop.expand(batch_size,*self.anchor_crop.size())
        crop_features = self.encoder.forward(batched_crop)
        frame_features = self.encoder.forward(frame)
        return self.cross_correlation(frame_features,crop_features)

