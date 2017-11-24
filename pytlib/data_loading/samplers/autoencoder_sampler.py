from image.frame import Frame
from image.box import Box
from image.object import Object
from data_loading.samplers.sampler import Sampler
from data_loading.sample import AutoEncoderSample
from image.affine import Affine
from excepts.general_exceptions import NoFramesException
from utils.dict_utils import get_deep
from image.ptimage import PTImage,Ordering,ValueClass
from image.random_perturber import RandomPerturber
import numpy as np
import random
import torch
from interface import implements

# This sampler provides images that contains a single item of something
# we want to encode, the target tensor includes both the crop itself and the coordinates
# random perturbations of the crop is an optional parameter
class AutoEncoderSampler(implements(Sampler)):

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
        # just grab the next random frame
        frame = self.source[random.choice(self.frame_ids)]

        # get a random crop object
        crop_objs = filter(lambda x: x.obj_type in self.obj_types,frame.get_objects())
        # print 'Num crop objs in sample: {0}'.format(len(crop_objs))
        crop = random.choice(crop_objs)
        # print 'crop_box: ' + str(crop.box)

        # frame.show_image_with_labels()

        # 1) Randomly perturb crop box (scale and translation)
        transformed_box = RandomPerturber.perturb_crop_box(crop.box,{})

        # 2) Take crop
        affine = Affine()
        scalex = float(self.crop_size[0])/transformed_box.edges()[0]
        scaley = float(self.crop_size[1])/transformed_box.edges()[1]
        affine.append(Affine.translation(-transformed_box.xy_min()))
        affine.append(Affine.scaling((scalex,scaley)))

        transformed_image = affine.apply_to_image(frame.image,self.crop_size) 
        # transformed_image.visualize(title='transformed image')

        # 3) Randomly perturb cropped image (rotation only)

        chw_image = transformed_image.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01)
        # chw_image.visualize(title='chw_image')
        # np_img = scale_np_img(PIL_to_cudnn_np(resized_image),[0,255],[0,1])
        sample = AutoEncoderSample([torch.Tensor(chw_image.get_data().astype(float))],
                                   [torch.Tensor(chw_image.get_data().astype(float))])
        return sample