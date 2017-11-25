from image.frame import Frame
from image.box import Box
from image.object import Object
from data_loading.samplers.sampler import Sampler
from data_loading.sample import Sample, EncodingDetectionSample
from image.affine import Affine
from image.image_utils import PIL_to_cudnn_np, scale_np_img
from excepts.general_exceptions import NoFramesException
from image.random_perturber import RandomPerturber
import numpy as np
import random
import torch
from interface import implements
from image.ptimage import Ordering,ValueClass

# the Encoding Detection Sampler is used for a proof of concept for a joint encoding + detection model
# the data contains two elements, an array of crops, and the full detection frame
# the targets contain two elements, the same array of crops, and an array of bounding boxes
# eg:
# Sample
#    data: [a single crop, Full frame]
#    target: [a single crop, a bounding box]
class EncodingDetectionSampler(implements(Sampler)):

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
        # 1) pick a random frame and a random crop
        frame = self.source[random.choice(self.frame_ids)]
        # frame.show_image_with_labels()
        # only deal with frames a single sample for now
        assert len(frame.objects)==1, "Frame has no objects!"

        # 2) randomly perturb the frame
        perturbed_frame = RandomPerturber.perturb_frame(frame,{'translation_range':[-0.2,0.2],'scaling_range':[0.9,1.1]})

        # 3) produce a crop target (assume there is only one)
        box = perturbed_frame.objects[0].box
        affine = Affine()
        scalex = float(self.crop_size[0])/box.edges()[0]
        scaley = float(self.crop_size[1])/box.edges()[1]
        affine.append(Affine.translation(-box.xy_min()))
        affine.append(Affine.scaling((scalex,scaley)))
        crop_image = affine.apply_to_image(perturbed_frame.image,self.crop_size)
        # crop_image.visualize(display=True)

        # 4) encode the bounding box
        chw_crop = crop_image.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01)
        chw_frame_img = perturbed_frame.image.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01)    
        data = [torch.Tensor(chw_crop.get_data().astype(float)),
                torch.Tensor(chw_frame_img.get_data().astype(float))]
        target = [torch.Tensor(chw_crop.get_data().astype(float)),
                  torch.Tensor(box.to_single_array().astype(float))]

        return EncodingDetectionSample(data,target)