import torch
from interface import Interface, implements
from visualization.image_visualizer import ImageVisualizer
from image.ptimage import PTImage

# A sample represents a single training example
# It must contain the data in pytorch tensor format to feed as input to some network
# The input data, target and output must all be lists and can have arbitrary length
# The only restraint is that the target and output must have the same length and
# their tensors should have the same shapes
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
        assert len(data)==1 and len(target)==1
        self.data = data
        self.target = target
        self.output = None

    def visualize(self,parameters={}):
        image_target = PTImage.from_cwh_torch(self.target[0])
        image_output = PTImage.from_cwh_torch(self.output[0])
        ImageVisualizer().set_image(image_target,parameters.get('title','') + ' : Target')
        ImageVisualizer().set_image(image_output,parameters.get('title','') + ' : Output')

    def set_output(self,output):
        assert len(output)==1
        self.output = output

    def get_data(self):
        return self.data

    def get_target(self):
        return self.target

# this is a sample for a complete detection and feature encoding problem
# data: [Array of crops, Full frame]
# target: [Array of crops, Array of bounding boxes]
# output: [Array of crops, Array of bounding boxes]
class DetectionSample(implements(Sample)):
    def __init__(self,data,target):
        self.data = data
        self.target = target
        self.output = None

    def visualize(self,parameters={}):
        pass

    def set_output(self,output):
        self.output = output

    def get_data(self):
        return self.data

    def get_target(self):
        return self.target    