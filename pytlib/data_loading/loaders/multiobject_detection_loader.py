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
from image.image_utils import draw_objects_on_np_image
import numpy as np
import random
import torch
from interface import implements
from visualization.image_visualizer import ImageVisualizer
from networks.multi_object_detector import MultiObjectDetector

class MultiObjectDetectionSample(implements(Sample)):
    def __init__(self,data,target,class_lookup=dict()):
        self.data = data
        self.target = target
        self.output = None
        # a dictionary of value to name for decoding the class
        self.class_lookup = class_lookup

    def __convert_to_objects(self,boxes,classes):
        boxlist = Box.tensor_to_boxes(boxes.cpu())
        objects = []
        for x,y in zip(*(boxlist,classes.cpu().numpy())):
            objects.append(Object(x,0,self.class_lookup[y]))
        return objects

    def visualize(self,parameters={}):
        image_original = PTImage.from_cwh_torch(self.data[0])
        drawing_image = image_original.to_order_and_class(Ordering.HWC,ValueClass.BYTE0255).get_data().copy()

        boxes,classes = self.output[1:]
        # Nx4 boxes and N class tensor 
        valid_boxes, valid_classes = MultiObjectDetector.post_process_boxes(boxes,classes,len(self.class_lookup))
        # convert targets
        real_targets = self.target[0][:,0]>-1
        filtered_targets = self.target[0][real_targets].reshape(-1,self.target[0].shape[1])
        target_boxes = filtered_targets[:,1:]
        target_classes = filtered_targets[:,0]

        draw_objects_on_np_image(drawing_image,self.__convert_to_objects(valid_boxes,valid_classes),color=(0,255,0))   
        draw_objects_on_np_image(drawing_image,self.__convert_to_objects(target_boxes,target_classes),color=(255,0,0))
        ImageVisualizer().set_image(PTImage(drawing_image),parameters.get('title','') + ' : Output')

    def set_output(self,output):
        self.output = output

    def get_data(self):
        return self.data

    def get_target(self):
        return self.target

# loads frames with multiple objects as targets
class MultiObjectDetectionLoader(implements(Loader)):
    def __init__(self,source,crop_size,max_objects=100,obj_types=None):
        self.source = source
        self.crop_size = crop_size
        self.obj_types = set(obj_types)
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

        # 2) generate a random perturbation and perturb the frame
        perturbed_frame = RandomPerturber.perturb_frame(frame,{})

        # 3) encode the objects into targets with size that does not exceed max_objects
        # if there are more objects than max_objects, the remaining ones are dropped.
        # vector of length max_objects
        # each comp vector has the form [class(1),bbox(4)]
        # a padding target is used to represent a non-existent object, this has the form [-1,-1,-1,-1,-1]
        
        # create the padding vector
        class_encoding,class_decoding = dict(),dict()
        padvec = [np.array([-1]*5) for i in range(self.max_objects)]
        for i,obj in enumerate(perturbed_frame.objects[0:min(self.max_objects,len(perturbed_frame.objects))]):
            if obj.obj_type not in self.obj_types:
                continue
            if obj.obj_type not in class_encoding:
                code = len(class_encoding)
                class_encoding[obj.obj_type] = code
                class_decoding[code] = obj.obj_type
            box_coords = obj.box.to_single_array()
            padvec[i] = np.concatenate((np.array([class_encoding[obj.obj_type]]),box_coords),axis=0)

        chw_image = perturbed_frame.image.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01)
        # perturbed_frame.visualize(title='chw_image',display=True)
        sample = MultiObjectDetectionSample([torch.Tensor(chw_image.get_data().astype(float))],
                                            [torch.Tensor(padvec)],
                                            class_decoding)
        return sample





