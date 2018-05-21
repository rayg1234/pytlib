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
from image.polygon import Polygon
from image.affine_transforms import resize_image_center_crop,apply_affine_to_frame
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
        # need to draw the mask layers ontop of the data with transparency
        assert False

    def set_output(self,output):
        self.output = output

    def get_data(self):
        return self.data

    def get_target(self):
        return self.target

# This is for semantic segmentation taskes, not instance segmentation
class SegmentationLoader(implements(Loader)):
    def __init__(self,source,crop_size,max_frames=1e8,obj_types=set()):
        self.source = source
        self.crop_size = crop_size
        self.obj_types_to_ids = dict()
        self.ids_to_obj_types = dict()
        self.frame_ids = []
        self.max_frames = max_frames

        print 'SegmentationLoader: finding valid frames'
        # parallelize this, this is too slow
        for i,frame in enumerate(self.source):
            if len(self.frame_ids)>=self.max_frames:
                break
            valid_obj_count = 0
            for obj in frame.get_objects():
                if obj.obj_type in obj_types or not obj_types:
                    valid_obj_count+=1
                    self.obj_types_to_ids[obj.obj_type]=self.obj_types_to_ids.get(obj.obj_type,len(self.obj_types_to_ids))
                    self.ids_to_obj_types[i]=self.ids_to_obj_types.get(i,obj.obj_type)
            if valid_obj_count>0:
                self.frame_ids.append(i)

        print 'The source has {0} items'.format(len(self.source))
        if len(self.frame_ids)==0:
            raise NoFramesException('No Valid Frames Found!')

        print '{0} frames and {1} classes found'.format(len(self.frame_ids),len(self.obj_types_to_ids))

    def next(self):
        # 1) pick a random frame
        frame = self.source[random.choice(self.frame_ids)]
        # 2) generate a random perturbation and perturb the frame, this also perturbs the objects including segementation polygons
        perturbed_frame = RandomPerturber.perturb_frame(frame,{})
        # 3) scale the perturbed frame to the desired input resolution
        crop_affine = resize_image_center_crop(perturbed_frame.image,self.crop_size)
        perturbed_frame = apply_affine_to_frame(perturbed_frame,crop_affine,self.crop_size)
        # visualize the perturbed_frame along with its perturbed objects and masks here
        # perturbed_frame.visualize(display=True)

        # 3) for each object type, produce a merged binary mask over the frame, 
        # this results in a w x h x k target map where k is the number of classes in consideration 
        # for now we will use the pycocotool's merge and polygon mapping functions since they are implemented in c
        # although I prefer to not have this dependency
        # loop over all object type and create a binary mask for each
        # declare a np array of whk
        masks = np.zeros(perturbed_frame.image.get_hw().tolist()+[len(self.obj_types_to_ids)])
        for k,v in self.obj_types_to_ids.items():
            # a) for all objs in the frame that belong to this type, create a merged mask
            polygons = []
            for obj in perturbed_frame.get_objects():
                if obj.obj_type==k:
                    polygons.extend(obj.polygons)
            masks[:,:,v] = Polygon.create_mask(polygons,perturbed_frame.image.get_wh()[0],perturbed_frame.image.get_wh()[1])

        # 4) create the segmentation sample
        chw_image = perturbed_frame.image.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01)
        # transpose the mask

        chw_mask = np.transpose(masks,axes=(2,0,1))

        # chw_image.visualize(title='chw_image')
        sample = SegmentationSample([torch.Tensor(chw_image.get_data().astype(float))],
                                    [torch.Tensor(chw_mask)],
                                    self.ids_to_obj_types)        

        return sample





