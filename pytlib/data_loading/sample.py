import torch
import torch.nn.functional as F
from interface import Interface, implements

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