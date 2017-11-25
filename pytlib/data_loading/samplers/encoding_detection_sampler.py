from image.frame import Frame
from image.box import Box
from image.object import Object
from data_loading.samplers.sampler import Sampler
from data_loading.sample import Sample, EncodingDetectionSample
from image.affine import Affine
from excepts.general_exceptions import NoFramesException
from image.random_perturber import RandomPerturber
from image.affine_transforms import crop_image_resize,resize_image_center_crop
import numpy as np
import random
import torch
import copy
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
        self.frame_size = params['frame_size']
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
        frame.show_image_with_labels('original_frame')
        # only deal with frames a single sample for now
        assert len(frame.objects)==1, "Frame has no objects!"

        # 2) resize frame
        affine_resized_frame = resize_image_center_crop(frame.image,self.frame_size)
        resized_frame = copy.copy(frame)
        resized_frame.image = affine_resized_frame.apply_to_image(resized_frame.image,self.frame_size)
        resized_frame.objects[0].box = affine_resized_frame.apply_to_box(resized_frame.objects[0].box)
        resized_frame.show_image_with_labels('resized_frame')

        # 3) randomly perturb the frame
        # perturbed_frame = RandomPerturber.perturb_frame(frame,{'translation_range':[-0.0,0.0],'scaling_range':[1.0,1.0]})
        perturbed_frame = RandomPerturber.perturb_frame(resized_frame,{'translation_range':[-0.2,0.2],'scaling_range':[0.9,1.1]})
        
        # 4) produce a center crop of the target (assume there is only one)
        box = perturbed_frame.objects[0].box

        affine_crop_image = crop_image_resize(perturbed_frame.image,box,self.crop_size)
        crop_image = affine_crop_image.apply_to_image(perturbed_frame.image,self.crop_size)
        crop_image.visualize(display=True)

        # 5) encode the bounding box
        chw_crop = crop_image.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01)
        chw_frame_img = perturbed_frame.image.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01)    
        data = [torch.Tensor(chw_crop.get_data().astype(float)),
                torch.Tensor(chw_frame_img.get_data().astype(float))]
        target = [torch.Tensor(chw_crop.get_data().astype(float)),
                  torch.Tensor(box.to_single_array().astype(float))]

        return EncodingDetectionSample(data,target)