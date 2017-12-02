from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from networks.conv_stack import ConvolutionStack
import torch
import math
from data_loading.sample import Sample

# both encodes the image and performs detection on a target box(s)
class CorrelationalDetector(nn.Module):
    def __init__(self):
        super(EncodingDetector, self).__init__()
        self.encoder = ConvolutionStack(3,final_relu=False)
        self.encoder.append(3,3,2)
        self.encoder.append(16,3,2)
        self.encoder.append(64,3,1)
        self.encoder.append(128,3,2)
        self.encoder.append(256,3,1)
        self.crosscor_batchnorm = nn.BatchNorm2d(1)

    # assert the input has two elements, first is the crop, second the full frame
    def forward(self, crop, frame):
        # recon,mu,logvar = None,None,None
        crop_feature_map = self.encoder.forward(crop)
        frame_feature_map = self.encoder.forward(frame)

        # now compute the convolution of the frame_feature_map against the crop_feature map
        # need to compute these unbatched because we are not using the same filter map for each conv
        batch_size = frame_feature_map.size(0)
        response_maps = []
        for i in range(0,batch_size):
            response = F.conv2d(frame_feature_map[i,:].unsqueeze(0),crop_feature_map[i,:].unsqueeze(0))
            response_maps.append(response.squeeze(0))
        rmap = torch.stack(response_maps,0)
        rmap = self.crosscor_batchnorm(rmap)
        return rmap
