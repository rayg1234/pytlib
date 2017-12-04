from image.frame import Frame
from image.box import Box
from image.object import Object
from data_loading.sample import Sample
from data_loading.samplers.sampler import Sampler
from image.affine import Affine
from excepts.general_exceptions import NoFramesException
from image.random_perturber import RandomPerturber
from visualization.image_visualizer import ImageVisualizer
from image.affine_transforms import crop_image_resize,resize_image_center_crop
import numpy as np
import random
import torch
from interface import implements
from image.ptimage import Ordering,ValueClass
from image.frame import Frame
import copy

# TODO move to the network definition because this tied to it?
# this is a sample for a complete detection and feature encoding problem
# data: [a crop, the Full frame]
# target: [a crop, a bounding box]
# output: [a crop, a bounding box]
class EncodingDetectionSample(implements(Sample)):
    def __init__(self,data,target):
        self.data = data
        self.target = target
        self.output = None

    def visualize(self,parameters={}):
        image_frame = PTImage.from_cwh_torch(self.data[1])
        image_target = PTImage.from_cwh_torch(self.target[0])
        image_output = PTImage.from_cwh_torch(self.output[0])
        # todo add a coversion from 2d to 3d for visuals
        target_box = Box.tensor_to_box(self.target[1].cpu(),image_frame.get_wh())
        objs = [Object(target_box,0,obj_type='T')]
        frame = Frame.from_image_and_objects(image_frame,objs)

        image_rmap = PTImage.from_2d_wh_torch(F.sigmoid(self.output[3]).data)
        ImageVisualizer().set_image(image_rmap,parameters.get('title','') + ' : RMap')
        ImageVisualizer().set_image(image_target,parameters.get('title','') + ' : Target')
        ImageVisualizer().set_image(image_output,parameters.get('title','') + ' : Output')
        ImageVisualizer().set_image(frame,parameters.get('title','') + ' : Frame')          

    def set_output(self,output):
        self.output = output

    def get_data(self):
        return self.data

    def get_target(self):
        return self.target

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
        # frame.show_image_with_labels('original_frame')
        # just work with the first object for now and assume frame only has 1

        # 2) resize frame
        affine_resized_frame = resize_image_center_crop(frame.image,self.frame_size)
        resized_frame = Frame(frame.image_path,copy.deepcopy(frame.objects))
        resized_frame.image = affine_resized_frame.apply_to_image(resized_frame.image,self.frame_size)
        resized_frame.objects[0].box = affine_resized_frame.apply_to_box(resized_frame.objects[0].box)
        # resized_frame.visualize(title='resized_frame')

        # 3) randomly perturb the frame
        # perturbed_frame = RandomPerturber.perturb_frame(resized_frame,{'translation_range':[-0.0,0.0],'scaling_range':[1.0,1.0]})
        perturbed_frame = RandomPerturber.perturb_frame(resized_frame,{'translation_range':[-0.1,0.1],'scaling_range':[0.9,1.1]})
        # perturbed_frame.show_image_with_labels('perturbed_frame')

        # 4) produce a center crop of the target (assume there is only one)
        box = perturbed_frame.objects[0].box

        affine_crop_image = crop_image_resize(perturbed_frame.image,box,self.crop_size)
        crop_image = affine_crop_image.apply_to_image(perturbed_frame.image,self.crop_size)
        # crop_image.visualize(display=True)

        # 5) encode the bounding box
        chw_crop = crop_image.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01)
        chw_frame_img = perturbed_frame.image.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01)    
        data = [torch.Tensor(chw_crop.get_data().astype(float)),
                torch.Tensor(chw_frame_img.get_data().astype(float))]


        target = [torch.Tensor(chw_crop.get_data().astype(float)),
                  Box.box_to_tensor(box,self.frame_size)]
        
        return EncodingDetectionSample(data,target)