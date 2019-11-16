from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
from image.frame import Frame
from image.box import Box
from image.object import Object
from data_loading.loaders.loader import Loader
from data_loading.sample import Sample
from image.affine import Affine
from excepts.general_exceptions import NoFramesException
from image.random_perturber import RandomPerturber
from visualization.image_visualizer import ImageVisualizer
from image.affine_transforms import crop_image_resize,resize_image_center_crop
import numpy as np
import random
import torch
from interface import implements
from image.ptimage import Ordering,ValueClass,PTImage
from image.frame import Frame
from image.image_utils import generate_response_map_from_boxes
from data_loading.sample import Sample
import copy
import torch.nn.functional as F

class TripletDetectionSample(implements(Sample)):
    def __init__(self,data,target):
        self.data = data
        self.target = target
        self.output = None

    def visualize(self,parameters={}):
        # image_frame = PTImage.from_cwh_torch(self.data[0])
        if parameters.get('mode','train')=='train':
            image_pos = PTImage.from_cwh_torch(self.data[0])
            image_neg = PTImage.from_cwh_torch(self.data[1])
            image_anchor = PTImage.from_cwh_torch(self.output[0])
            image_pos_map = PTImage.from_2d_wh_torch(F.sigmoid(self.output[1]).data)
            image_neg_map = PTImage.from_2d_wh_torch(F.sigmoid(self.output[2]).data)
            image_pos_tar = PTImage.from_2d_wh_torch(self.target[0])
            image_neg_tar = PTImage.from_2d_wh_torch(self.target[1])
            # target_box = Box.tensor_to_box(self.target[0].cpu(),image_pos.get_wh())
            # objs = [Object(target_box,0,obj_type='T')]
            # pos_frame = Frame.from_image_and_objects(image_pos,objs)

            # ImageVisualizer().set_image(image_frame,parameters.get('title','') + ' : Frame')
            ImageVisualizer().set_image(image_anchor,parameters.get('title','') + ' : anchor')
            ImageVisualizer().set_image(image_pos,parameters.get('title','') + ' : pos_frame')
            ImageVisualizer().set_image(image_neg,parameters.get('title','') + ' : neg_frame')
            ImageVisualizer().set_image(image_pos_tar,parameters.get('title','') + ' : pos_target')
            ImageVisualizer().set_image(image_neg_tar,parameters.get('title','') + ' : neg_target')            
            ImageVisualizer().set_image(image_pos_map,parameters.get('title','') + ' : pos_res')
            ImageVisualizer().set_image(image_neg_map,parameters.get('title','') + ' : neg_res')
        else:
            img_frame = PTImage.from_cwh_torch(self.data[0])
            img_frame_xcor = PTImage.from_2d_wh_torch(F.sigmoid(self.output[0]).data)

            # img_pos = PTImage.from_cwh_torch(self.data[1])
            # img_neg = PTImage.from_cwh_torch(self.data[2])
            # image_pos_map = PTImage.from_2d_wh_torch(F.sigmoid(self.output[1]).data)
            # image_neg_map = PTImage.from_2d_wh_torch(F.sigmoid(self.output[2]).data)

            ImageVisualizer().set_image(img_frame,parameters.get('title','') + ' : Frame')
            ImageVisualizer().set_image(img_frame_xcor,parameters.get('title','') + ' : Frame xcor')
            # ImageVisualizer().set_image(img_pos,parameters.get('title','') + ' : pos')
            # ImageVisualizer().set_image(image_pos_map,parameters.get('title','') + ' : pos xcor')
            # ImageVisualizer().set_image(img_neg,parameters.get('title','') + ' : neg')
            # ImageVisualizer().set_image(image_neg_map,parameters.get('title','') + ' : neg xcor')

    def set_output(self,output):
        self.output = output

    def get_data(self):
        return self.data

    def get_target(self):
        return self.target

class TripletDetectionLoader(implements(Loader)):
    def __init__(self,source,crop_size,anchor_size,obj_types=None,mode='train'):
        self.source = source
        self.crop_size = crop_size
        self.obj_types = obj_types
        self.anchor_size = anchor_size
        self.frame_ids = []
        self.perturbations = {'translation_range':[-0.0,0.0],'scaling_range':[2.0,2.0]}
        self.mode = mode
        #index all the frames that have at least one item we want
        # TODO turn this into a re-usable filter module
        if self.mode=='train':
            for i,frame in enumerate(self.source):
                crop_objs = [x for x in frame.get_objects() if x.obj_type in self.obj_types]
                if(len(crop_objs)>0):
                    self.frame_ids.append(i)

            print('The source has {0} items'.format(len(self.source)))
            if len(self.frame_ids)==0:
                raise NoFramesException('No Valid Frames Found!')

            print('{0} frames found'.format(len(self.frame_ids)))

    # find a negative crop in a frame, must not contain an object of interest
    def find_negative_crop(self,frame,objects):
        # pick a random crop, check that it does not overlap with an existing target
        # TODO, this is inefficient, fix this algorithm later
        frame_size = frame.image.get_wh();
        max_attempts = 10
        for i in range(0,max_attempts):
            randcx = random.randrange(old_div(self.crop_size[0],2),frame_size[0]-old_div(self.crop_size[0],2))
            randcy = random.randrange(old_div(self.crop_size[1],2),frame_size[1]-old_div(self.crop_size[1],2))
            new_box = Box(randcx - old_div(self.crop_size[0],2), 
                          randcy - old_div(self.crop_size[1],2),
                          randcx + old_div(self.crop_size[0],2),
                          randcy + old_div(self.crop_size[1],2))
            box_found = all(Box.intersection(x.box,new_box) is None for x in objects)
            if box_found:
                return new_box
        return None

    def load_train(self):
        frame1,frame2,neg_box,pos_box,anchor_box = None,None,None,None,None
        # TODO, this should probably break if never find anything for a while
        while neg_box is None:
            indices = random.sample(self.frame_ids,2)
            frame1,frame2 = [self.source[x] for x in indices]
            frame1_objs = [x for x in frame1.get_objects() if x.obj_type in self.obj_types]
            frame2_objs = [x for x in frame2.get_objects() if x.obj_type in self.obj_types]
            # get random pos boxes
            pos_box = random.choice(frame1_objs).box
            anchor_box = random.choice(frame2_objs).box

            # find random neg crop
            neg_box = self.find_negative_crop(frame1,frame1_objs)

        perturbed_pos_box = RandomPerturber.perturb_crop_box(pos_box,self.perturbations)
        affine_crop0 = crop_image_resize(frame1.image,perturbed_pos_box,self.crop_size)
        pos_crop = affine_crop0.apply_to_image(frame1.image,self.crop_size)

        affine_crop1 = crop_image_resize(frame2.image,anchor_box,self.anchor_size)
        anchor_crop = affine_crop1.apply_to_image(frame2.image,self.anchor_size)

        affine_crop2 = crop_image_resize(frame1.image,neg_box,self.crop_size)
        neg_crop = affine_crop2.apply_to_image(frame1.image,self.crop_size)
        # neg_crop.visualize(display=True,title='neg')

        # now find all the boxes that intersect with the perturbed_pos_box
        intersected_boxes = []
        for obj in [x for x in frame1.get_objects() if x.obj_type in self.obj_types]:
             if Box.intersection(obj.box,perturbed_pos_box) is not None:
                intersected_boxes.append(obj.box)

        intersected_boxes = list([affine_crop0.apply_to_box(x) for x in intersected_boxes])
        # test display
        # disp_frame = Frame.from_image_and_objects(pos_crop,[Object(box_crop)])
        # disp_frame.visualize(display=True,title='pos frame')
        # pos_crop.visualize(display=True,title='pos crop')

        pos = torch.Tensor(pos_crop.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01).get_data().astype(float))
        neg = torch.Tensor(neg_crop.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01).get_data().astype(float))
        anchor = torch.Tensor(anchor_crop.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01).get_data().astype(float))

        # pos_map = generate_response_map_from_boxes(pos_crop.get_hw(),intersected_boxes)
        # PTImage.from_2d_numpy(pos_map).visualize(display=True,title='pos frame')      
        pos_map = torch.Tensor(generate_response_map_from_boxes(pos_crop.get_hw(),intersected_boxes))
        neg_map = torch.Tensor(generate_response_map_from_boxes(pos_crop.get_hw()))

        data = [pos,neg,anchor]
        target = [pos_map,neg_map,anchor]
        return TripletDetectionSample(data,target)

    def load_test(self):
        frame = random.choice(self.source)
        random_t = torch.Tensor(3,self.anchor_size[0],self.anchor_size[1])
        frame_t = torch.Tensor(frame.image.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01).get_data().astype(float))
        data = [frame_t,random_t]
        target = [torch.Tensor(1)] # dummy target
        return TripletDetectionSample(data,target)

    # pick a frame to generate positive and negative crop
    def __next__(self):
        return self.load_train() if self.mode == 'train' else self.load_test()
 

