from image.frame import Frame
from image.box import Box
from image.object import Object
from data_loading.samplers.sampler import Sampler
from data_loading.sample import Sample
from image.affine import Affine
from image.image_utils import PIL_to_cudnn_np, scale_np_img
from excepts.general_exceptions import NoFramesException
import numpy as np
import random
import torch
from interface import implements

# the Encoding Detection Sampler is used for a proof of concept for a joint encoding + detection model
# the data contains two elements, an array of crops, and the full detection frame
# the targets contain two elements, the same array of crops, and an array of bounding boxes
# eg:
# Sample
#    data: [a single crop, Full frame]
#    target: [a single crop, a bounding box]
class EncodingDetectionSample(implements(Sampler)):

    def __init__(self,source,params):
        self.source = source
        self.crop_size = params['crop_size']
        self.obj_types = params['obj_types']
        self.frame_ids = []
        #index all the frames that have at least one item we want
        # TODO turn this into a re-usable filter module
        for i,frame in enumerate(self.source):
            crop_objs = filter(lambda x: x.obj_type in self.obj_types,frame.get_objects())
            if(len(crop_objs)>0):
                self.frame_ids.append(i)

        print 'The source has {0} items'.format(len(self.source))
        if len(self.frame_ids)==0:
            raise NoFramesException('No Valid Frames Found!')

        print '{0} frames found'.format(len(self.frame_ids))

    def next(self):
        # 1) pick a random frame:

        # 2) randomly perturb the frame
        
        # 3) produce a crop target (assume there is only one)

        # 4) encode the bounding box

        

        return None