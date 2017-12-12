from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from networks.conv_stack import ConvolutionStack
import torch
import math
from data_loading.sample import Sample
from networks.vae import VAE

# both encodes the image and performs detection on a target box(s)
class TripletCorrelationalDetector(nn.Module):
    def __init__(self,anchor_size=(127,127)):
        super(TripletCorrelationalDetector, self).__init__()
        self.vae = VAE()
        self.encoder = self.vae.get_encoder()
        self.crosscor_batchnorm0 = nn.BatchNorm2d(1)
        # this used to dynamically initialized to deal with runtime cropsizes
        # I will keep it this way incase I want to revisit        
        self.register_parameter('anchor_crop', None)
        self.anchor_size = torch.Size((3,anchor_size[0],anchor_size[1]))

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

    def forward(self, pos, neg, pos_crop):
        recon,mu,logvar = self.vae.forward(pos_crop)
        pos_feature_map = self.encoder.forward(pos)
        neg_feature_map = self.encoder.forward(neg)
        # initialize feature_encoding if None
        batch_size = pos.size(0)
        self.init_anchor(pos.is_cuda)
        anchor = self.anchor_crop.expand(batch_size,*self.anchor_crop.size())
        anchor_feature_map = self.encoder.forward(anchor)

        cxp = self.cross_correlation(pos_feature_map,anchor_feature_map)
        cxn = self.cross_correlation(neg_feature_map,anchor_feature_map)
        return anchor,cxp,cxn,recon,mu,logvar

    def infer(self,frame,pos_crop):
        # self.vae.forward(pos_crop)
        batch_size = frame.size(0)
        self.init_anchor(frame.is_cuda)
        batched_crop = self.anchor_crop.expand(batch_size,*self.anchor_crop.size())
        crop_features = self.encoder.forward(batched_crop)
        frame_features = self.encoder.forward(frame)
        # posf,negf = self.encoder.forward(pos),self.encoder.forward(neg)
        #self.cross_correlation(posf,crop_features),self.cross_correlation(negf,crop_features)
        return self.cross_correlation(frame_features,crop_features)

