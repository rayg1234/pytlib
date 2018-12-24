import torch.nn as nn
import torch.nn.functional as F
from networks.conv_stack import ConvolutionStack
import torch
import math
from data_loading.sample import Sample
from networks.vae import VAE

class TripletCorrelationalDetector(nn.Module):
    def __init__(self):
        super(TripletCorrelationalDetector, self).__init__()
        self.vae = VAE()
        self.encoder = self.vae.get_encoder()
        self.register_parameter('anchor_feature_map', None)

    # compute the xcorrelation of these feature maps
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

    def forward(self, pos, neg, pos_crop):
        pos_feature_map = self.encoder.forward(pos)
        neg_feature_map = self.encoder.forward(neg)

        recon,mu,logvar = self.vae.forward(pos_crop)
        anchor_feature_map = self.vae.get_encoding_feature_map()

        # save the feature_map for inference, yes this overwrites it every frame
        self.anchor_feature_map = nn.Parameter(anchor_feature_map.data[0])

        cxp = self.cross_correlation(pos_feature_map,anchor_feature_map)
        cxn = self.cross_correlation(neg_feature_map,anchor_feature_map)
        return pos_crop,cxp,cxn,recon,mu,logvar

    def infer(self,frame,random_patch):
        self.vae.forward(random_patch)
        batch_size = frame.size(0)

        if self.anchor_feature_map is None:
            self.anchor_feature_map = nn.Parameter(self.vae.get_encoding_feature_map().data)
        
        batched_feature_map = self.anchor_feature_map.expand(batch_size,*self.anchor_feature_map.size())
        frame_features = self.encoder.forward(frame)
        return self.cross_correlation(frame_features,self.anchor_feature_map)

