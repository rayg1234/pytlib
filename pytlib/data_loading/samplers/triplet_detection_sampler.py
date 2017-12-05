from image.frame import Frame
from image.box import Box
from image.object import Object
from data_loading.samplers.sampler import Sampler
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

            # ImageVisualizer().set_image(image_frame,parameters.get('title','') + ' : Frame')  
            ImageVisualizer().set_image(image_pos,parameters.get('title','') + ' : Pos')
            ImageVisualizer().set_image(image_neg,parameters.get('title','') + ' : Neg')
        else:
            img_frame = PTImage.from_cwh_torch(self.data[0])
            img_frame_xcor = PTImage.from_2d_wh_torch(F.sigmoid(self.output[0]).data)
            ImageVisualizer().set_image(img_frame,parameters.get('title','') + ' : Frame')
            ImageVisualizer().set_image(img_frame_xcor,parameters.get('title','') + ' : Frame xcor')

    def set_output(self,output):
        self.output = output

    def get_data(self):
        return self.data

    def get_target(self):
        return self.target

class TripletDetectionSampler(implements(Sampler)):
    def __init__(self,source,params):
        self.source = source
        self.crop_size = params['crop_size']
        self.obj_types = params['obj_types']
        self.frame_ids = []
        self.perturbations = {'translation_range':[-0.1,0.1],'scaling_range':[0.9,1.0]}
        self.mode = params.get('mode','train')
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

    # find a negative crop in a frame, must not contain an object of interest
    def find_negative_crop(self,frame,objects):
        # pick a random crop, check that it does not overlap with an existing target
        # TODO, this is inefficient, fix this algorithm later
        frame_size = frame.image.get_wh();
        max_attempts = 10
        for i in range(0,max_attempts):
            randcx = random.randrange(self.crop_size[0]/2,frame_size[0]-self.crop_size[0]/2)
            randcy = random.randrange(self.crop_size[1]/2,frame_size[1]-self.crop_size[1]/2)
            new_box = Box(randcx - self.crop_size[0]/2, 
                          randcy - self.crop_size[1]/2,
                          randcx + self.crop_size[0]/2,
                          randcy + self.crop_size[1]/2)
            box_found = all(Box.intersection(x.box,new_box) is None for x in objects)
            if box_found:
                return new_box
        return None

    # pick a frame to generate positive and negative crop
    def next(self):
        frame1 = self.source[random.choice(self.frame_ids)]
        # frame1.visualize(display=True)

        frame1_objs = filter(lambda x: x.obj_type in self.obj_types,frame1.get_objects())
        # get random pos boxes
        pos_box = random.choice(frame1_objs).box

        # find random neg crop
        neg_box = self.find_negative_crop(frame1,frame1_objs)
        # its possible that it was impossible to find a neg box, in that case, abandon the sample all together
        if neg_box is None:
            return None

        pos_box = RandomPerturber.perturb_crop_box(pos_box,self.perturbations)
        affine_crop = crop_image_resize(frame1.image,pos_box,self.crop_size)
        pos_crop = affine_crop.apply_to_image(frame1.image,self.crop_size)
        # pos_crop.visualize(display=True,title='pos')

        affine_crop = crop_image_resize(frame1.image,neg_box,self.crop_size)
        neg_crop = affine_crop.apply_to_image(frame1.image,self.crop_size)
        # neg_crop.visualize(display=True,title='neg')

        pos = torch.Tensor(pos_crop.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01).get_data().astype(float))
        neg = torch.Tensor(neg_crop.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01).get_data().astype(float))

        data, target = [],[torch.Tensor(1)]
        if self.mode=='train':
            data = [pos,neg]
        else:
            frame_t = torch.Tensor(frame1.image.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01).get_data().astype(float))
            data = [frame_t,pos]
        return TripletDetectionSample(data,target)

 

