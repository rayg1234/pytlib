import torch
import copy
import os
import numpy as np
from PIL import Image
from image.image_utils import PIL_to_cudnn_np, scale_np_img
import matplotlib.pyplot as plt
# This is a general representation of images
# and act as a mediator between different types and storage orders
# here is main use case:
# 1) Load image using PIL to PIL image format
# 2) store image as HWC numpy array
# 3) apply perturbations/affine transforms
# 3) scale and transpose to chw (cudnn format)
# 4) convert to pytorch tensor for NN compute
#
# to take the network output and convert back
# 1) unscale and tranpose to HWC and convert to numpy
# Note on cudnn storage order is BCHW and PIL uses HWC arrays
#
# for Numpy 'C' Style row-major arrays, the first dimension is the
# slowest changing dimension (last is fastest changing), and thus continguous slices of memory is across the last dim
# so for numpy c-style arraynd, we should prefer BCHW for accessing single elements from a batch
# the pytorch tensor should also have this memory layout

class Ordering:
    CHW = 'CHW'
    HWC = 'HWC'

class ValueClass:
    FLOAT01 = {'dtype':'float','range':[0,1]}
    BYTE0255 = {'dtype':'uint8','range':[0,255]}

class PTImage:
    def __init__(self,data=None,pil_image_path='',ordering=Ordering.HWC,vc=ValueClass.BYTE0255,persist=True):
        self.image_path = pil_image_path
        self.ordering = ordering
        self.vc = vc
        self.persist = persist
        self.__data = data # numpy array

    def copy(self):
        copy = PTImage(pil_image_path=self.image_path,ordering=self.ordering,vc=self.vc)
        if self.__data is not None:
            copy.__data = np.copy(self.__data)
        return copy

    def get_data(self):
        if self.__data is None:
            assert os.path.isfile(self.image_path), "cant open file: %s" % self.image_path
            tmp_data = np.asarray(Image.open(self.image_path, 'r'))
            if self.persist:
                self.__data = tmp_data
            return tmp_data
        else:
            return self.__data

    def get_pil_image(self):
        return Image.fromarray(self.get_data())

    def visualize(self,axes=None,display=False,block=True,title='PTImage Visualization'):
        # TODO if already in the right order, don't both converting
        display_img = self.to_order_and_class(Ordering.HWC,ValueClass.BYTE0255)
        cur_ax = None
        if axes is None:
            fig,cur_ax = plt.subplots(1,figsize=(15, 8))
            fig.canvas.set_window_title(title)
        else:
            cur_ax = axes
        cur_ax.imshow(display_img.get_data(), interpolation='nearest', vmin=0, vmax=255)
        if display:
            plt.show(block=block)
        return cur_ax
        
    # makes a copy
    def to_order_and_class(self,new_ordering,new_value_class):
        new_data = None

        if self.ordering == new_ordering:
            new_data = self.get_data()
        elif self.ordering == Ordering.CHW and new_ordering == Ordering.HWC:
            new_data = np.transpose(self.get_data(),axes=(1,2,0))
        elif self.ordering == Ordering.HWC and new_ordering == Ordering.CHW:
            new_data = np.transpose(self.get_data(),axes=(2,0,1))
        else:
            assert False, 'Dont know how to convert to this ordering'

        if self.vc != new_value_class:
            new_data = scale_np_img(new_data,self.vc['range'],new_value_class['range'],new_value_class['dtype'])

        new_img = PTImage(data=new_data,ordering=new_ordering,vc=new_value_class)
        return new_img     

    @classmethod
    def from_cwh_torch(cls,torch_img):
        return cls(data=torch_img.cpu().numpy().squeeze(),ordering=Ordering.CHW,vc=ValueClass.FLOAT01)

