from image.frame import Frame
from image.box import Box
from image.object import Object
from data_loading.samplers.sampler import Sampler
from data_loading.sample import Sample
from image.affine import Affine
from excepts.general_exceptions import NoFramesException
from utils.dict_utils import get_deep
from image.ptimage import PTImage,Ordering,ValueClass
from image.random_perturber import RandomPerturber
import numpy as np
import random
import torch
from interface import implements

# This is a sample where the image and the target are both images
class AutoEncoderSample(implements(Sample)):
    def __init__(self,data,target):
        self.data = data
        self.target = target
        self.output = None

    def visualize(self,parameters={}):
        image_target = PTImage.from_cwh_torch(self.target[0])
        image_output = PTImage.from_cwh_torch(self.output[0])
        ImageVisualizer().set_image(image_target,parameters.get('title','') + ' : Target')
        ImageVisualizer().set_image(image_output,parameters.get('title','') + ' : Output')

    # specific to the AE sample, the first element of the output has the same shape as the target
    def set_output(self,output):
        assert output[0].size() == self.target[0].size()
        self.output = output

    def get_data(self):
        return self.data

    def get_target(self):
        return self.target

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
        # frame.show_image_with_labels()
        # get a random crop object
        crop_objs = filter(lambda x: x.obj_type in self.obj_types,frame.get_objects())
        # print 'Num crop objs in sample: {0}'.format(len(crop_objs))
        crop = random.choice(crop_objs)
        # print 'crop_box: ' + str(crop.box)

        # frame.show_image_with_labels()

        # 1) Randomly perturb crop box (scale and translation)
        transformed_box = RandomPerturber.perturb_crop_box(crop.box,{})

        # 2) Take crop, todo, change to using center crop to preserve aspect ratio
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
        sample = AutoEncoderSample([torch.Tensor(chw_image.get_data().astype(float))],
                                   [torch.Tensor(chw_image.get_data().astype(float))])
        return sample