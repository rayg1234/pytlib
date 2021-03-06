from __future__ import division
from past.utils import old_div
import torch
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.autograd import Variable
from networks.resnetcnn import ResNetCNN
from utils.batch_box_utils import rescale_boxes, generate_region_meshgrid, batch_nms
import numpy as np

class MultiObjectDetector(nn.Module):
    def __init__(self, nboxes_per_pixel=5, num_classes=2):
        # num_classes to predict, includes background
        super(MultiObjectDetector, self).__init__()
        self.feature_map_generator = ResNetCNN()
        self.register_parameter('box_predictor_weights', None)
        self.register_parameter('class_predictor_weights', None)
        self.nboxes_per_pixel = nboxes_per_pixel
        self.num_classes = num_classes

    def __init_weights(self,fmaps):
        # 1x1 conv weights:
        # cout x cin x H x W
        filter_tensor = torch.Tensor(4*self.nboxes_per_pixel,fmaps.shape[1],1,1)
        self.box_predictor_weights = nn.Parameter(filter_tensor)
        stdv = 1. / math.sqrt(self.box_predictor_weights.size(1))
        self.box_predictor_weights.data.uniform_(-stdv, stdv)

        filter_tensor = torch.Tensor(self.num_classes*self.nboxes_per_pixel,fmaps.shape[1],1,1)
        self.class_predictor_weights = nn.Parameter(filter_tensor)
        stdv = 1. / math.sqrt(self.class_predictor_weights.size(1))
        self.class_predictor_weights.data.uniform_(-stdv, stdv)

        if fmaps.data.is_cuda:
            self.cuda()

    def __box_predictor(self,feature_maps):
        # Simply use 1x1 convolution to regress down to HxWx(Nx4) result
        # where N is the number of boxes per super pixel
        convoutput = F.conv2d(feature_maps, self.box_predictor_weights)
        new_shape = convoutput.shape[0:1] + torch.Size([-1, self.nboxes_per_pixel]) + convoutput.shape[2:4]
        # rehsape as BxNx4xHxW
        # apply sigmoid to values between 0 and 1?
        # or normalize here
        return torch.reshape(convoutput,new_shape)

    def __class_predictor(self,feature_maps):
        convoutput = F.conv2d(feature_maps, self.class_predictor_weights)
        new_shape = convoutput.shape[0:1] + torch.Size([self.num_classes, self.nboxes_per_pixel]) + convoutput.shape[2:4]
        # dont take softmax here, take logsoftmax in the loss
        return torch.reshape(convoutput,new_shape)

    @classmethod
    def post_process_boxes(cls, original_image, boxes, classes, num_classes):
        # post-process to produce a Nx5 tensor of valid boxes
        assert len(boxes.shape)==4 and len(classes.shape)==4, \
            'boxes and classes must be non-batched tensors of dim 4'
        batch_size = boxes.shape[0]
        softmax_classes = F.softmax(classes,dim=0).flatten(start_dim=1)
        argmax_classes = torch.argmax(softmax_classes,dim=0)
        mask = argmax_classes<num_classes

        # transform from (-1,1) region coordinate system to original image coords
        region_size = old_div(np.array(original_image.shape[1:]),np.array(boxes.shape[2:]))
        hh,ww = generate_region_meshgrid(boxes.shape[2:], region_size, old_div(region_size,2))
        rescaled_box_preds = rescale_boxes(boxes.unsqueeze(0), region_size, [hh,ww])        
        flatten_boxes = rescaled_box_preds.squeeze().flatten(start_dim=1)

        # only select those that are non-background
        valid_boxes = flatten_boxes[:,mask].transpose(0,1)
        valid_classes = argmax_classes[mask]

        nms_boxes, mask = batch_nms(valid_boxes)
        nms_classes = valid_classes[mask]
        return nms_boxes, nms_classes


    def forward(self, x):       
        # CNN Compute, outputs BCHW order on cudnn
        feature_maps = self.feature_map_generator.forward(x)
        if self.class_predictor_weights is None:
            self.__init_weights(feature_maps)         
        boxes = self.__box_predictor(feature_maps)
        classes = self.__class_predictor(feature_maps)
        return x, boxes, classes

