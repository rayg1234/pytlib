import torch
from interface import Interface, implements
from visualization.image_visualizer import ImageVisualizer
from image.ptimage import PTImage
from image.frame import Frame
from image.image_utils import tensor_to_box
from image.object import Object
# A sample represents a single training example, it can be different for different 
# types of models. 
# It must contain the data in pytorch tensor format to feed as input to some network
# The input data, target and output must all be lists and can have arbitrary length
class Sample(Interface):
    def __init__(self,data,target):
        pass

    def visualize(self,parameters={}):
        pass

    def set_output(self,output):
        pass

    def get_data(self):
        pass

    def get_target(self):
        pass

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
        image_ccmap = PTImage.from_2d_cwh_torch(self.output[3])
        target_box = tensor_to_box(self.target[1].cpu(),image_frame.get_wh())
        output_box = tensor_to_box(self.output[4].cpu(),image_frame.get_wh())
        objs = [Object(target_box,0,obj_type='T'),Object(output_box,1,obj_type='O')]
        frame = Frame.from_image_and_objects(image_frame,objs)
        ImageVisualizer().set_image(image_target,parameters.get('title','') + ' : Target')
        ImageVisualizer().set_image(image_output,parameters.get('title','') + ' : Output')
        ImageVisualizer().set_image(image_ccmap,parameters.get('title','') + ' : CMap')
        ImageVisualizer().set_image(frame,parameters.get('title','') + ' : Frame')          

    def set_output(self,output):
        self.output = output

    def get_data(self):
        return self.data

    def get_target(self):
        return self.target    