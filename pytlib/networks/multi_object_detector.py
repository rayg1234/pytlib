import torch
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.autograd import Variable
from networks.resnetcnn import ResNetCNN

class MultiObjectDetector(nn.Module):
    def __init__(self):
        super(MultiObjectDetector, self).__init__()
        self.feature_map_generator = ResNetCNN()
        self.register_parameter('box_predictor_weights', None)
        self.nboxes_per_pixel = 5

    def __init_weights(self,fmaps):
    	# 1x1 conv weights:
    	# cout x cin x H x W
    	filter_tensor = torch.Tensor(6*self.nboxes_per_pixel,fmaps.shape[1],1,1)
        self.box_predictor_weights = nn.Parameter(filter_tensor)
        stdv = 1. / math.sqrt(self.box_predictor_weights.size(1))
        self.box_predictor_weights.data.uniform_(-stdv, stdv)
        if fmaps.data.is_cuda:
            self.cuda()

    def __box_predictor(self,feature_maps):
    	if self.box_predictor_weights is None:
    		self.__init_weights(feature_maps)
        # Detector head compute
        # Simply use 1x1 convolution to regress down to HxWx(Nx5) result
        # where N is the number of boxes per super pixel
        return F.conv2d(feature_maps, self.box_predictor_weights)


    def forward(self, x):
        # CNN Compute, outputs BCHW order on cudnn
        feature_maps = self.feature_map_generator.forward(x)
        boxes = self.__box_predictor(feature_maps)
        return boxes

