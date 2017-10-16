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

class CropSampler(Sampler):

    def __init__(self,source,params):
        Sampler.__init__(self,source)
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
        # just grab the next random frame
        frame = self.source[random.choice(self.frame_ids)]
        frame_image = frame.get_pil_image()

        crop_objs = filter(lambda x: x.obj_type in self.obj_types,frame.get_objects())
        print 'Num crop objs in sample: {0}'.format(len(crop_objs))

        # randomly grab a crop_obj
        crop = random.choice(crop_objs)
        crop_image = frame_image.crop(crop.box.to_single_array())
        resized_image = crop_image.resize(self.crop_size)
        np_img = scale_np_img(PIL_to_cudnn_np(resized_image),[0,255],[0,1])

        #todo add targets
        sample = Sample(torch.Tensor(np_img.astype(float)))
        return sample