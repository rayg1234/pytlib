from image.frame import Frame
from image.box import Box
from image.object import Object
from data_loading.loaders.loader import Loader
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
from visualization.image_visualizer import ImageVisualizer

class SegmentationSample(implements(Sample)):
    def __init__(self,data,target,class_lookup=dict()):
        self.data = data
        self.target = target
        self.output = None
        # a dictionary of value to name for decoding the class
        self.class_lookup = class_lookup

    def visualize(self,parameters={}):
        pass

    def set_output(self,output):
        self.output = output

    def get_data(self):
        return self.data

    def get_target(self):
        return self.target

# This is for semantic segmentation taskes, not instance segmentation
class SegmentationLoader(implements(Loader)):
    def __init__(self,source,crop_size,max_objects=100,obj_types=None):
        self.source = source
        self.crop_size = crop_size
        self.obj_types = obj_types
        self.max_objects = max_objects
        self.frame_ids = []

        #index all the frames that have at least one item we want
        # TODO turn this into a re-usable filter module
        for i,frame in enumerate(self.source):
            crop_objs = filter(lambda x: not self.obj_types or x.obj_type in self.obj_types,frame.get_objects())
            if(len(crop_objs)>0):
                self.frame_ids.append(i)

        print 'The source has {0} items'.format(len(self.source))
        if len(self.frame_ids)==0:
            raise NoFramesException('No Valid Frames Found!')

        print '{0} frames found'.format(len(self.frame_ids))

    def next(self):
        # 1) pick a random frame
        frame = self.source[random.choice(self.frame_ids)]

        # 2) generate a random perturbation and perturb the frame, this also perturbs the objects including segementation polygons
        perturbed_frame = RandomPerturber.perturb_frame(frame,{})

        # 3) for each object type, produce a merged binary mask over the frame, 
        # this results in a w x h x k target map where k is the number of classes in consideration 
        # for now we will use the pycocotool's merge and polygon mapping functions since they are implemented in c
        # although I prefer to not have this dependency

        return sample





