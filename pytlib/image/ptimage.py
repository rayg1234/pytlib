import torch
import copy
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from image.box import Box
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
        transform_image = self.to_order_and_class(Ordering.HWC,ValueClass.BYTE0255)
        return Image.fromarray(transform_image.get_data())

    @staticmethod
    def scale_np_img(image,input_range,output_range,output_type=float):
        assert len(input_range)==2 and len(output_range)==2
        scale = float(output_range[1] - output_range[0])/(input_range[1] - input_range[0])
        offset = output_range[0] - input_range[0]*scale;
        return (image*scale+offset).astype(output_type);

    def visualize(self,axes=None,display=False,title='PTImage Visualization'):
        # TODO if already in the right order, don't both converting
        display_img = self.to_order_and_class(Ordering.HWC,ValueClass.BYTE0255)
        fig,cur_ax = None,None
        if axes is None:
            fig,cur_ax = plt.subplots(1,figsize=(15, 8))
            fig.canvas.set_window_title(title)
        else:
            cur_ax = axes
        # cur_ax.imshow(display_img.get_data())
        cur_ax.imshow(display_img.get_data().squeeze(), vmin=0, vmax=255)
        if display:
            plt.show(block=True)
            plt.close()
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
            new_data = PTImage.scale_np_img(new_data,self.vc['range'],new_value_class['range'],new_value_class['dtype'])

        new_img = PTImage(data=new_data,ordering=new_ordering,vc=new_value_class)
        return new_img

    def get_dims(self):
        return np.array(self.get_data().shape)

    def get_bounding_box(self):
        return Box.from_single_array(np.array([0,0,self.get_wh()[0],self.get_wh()[1]]))

    # get height and width, in that order
    def get_wh(self):
        shape = self.get_data().shape
        if self.ordering == Ordering.CHW:
            return np.array([shape[2],shape[1]])
        else:
            return np.array([shape[1],shape[0]])

    # get height and width, in that order
    def get_hw(self):
        shape = self.get_data().shape
        if self.ordering == Ordering.CHW:
            return np.array([shape[1],shape[2]])
        else:
            return np.array([shape[0],shape[1]])

    @classmethod
    def from_numpy_array(cls,np_array,ordering=Ordering.HWC,vc=ValueClass.BYTE0255):
        return cls(data=np_array,ordering=ordering,vc=vc)

    @classmethod
    def from_pil_image(cls,pil_img):
        return cls(data=np.asarray(pil_img),ordering=Ordering.HWC,vc=ValueClass.BYTE0255)

    @classmethod
    def from_cwh_torch(cls,torch_img):
        return cls(data=torch_img.cpu().numpy(),ordering=Ordering.CHW,vc=ValueClass.FLOAT01)

    @classmethod
    def from_2d_numpy(cls,map2d):
        # assumes img2d has 2 dimensions
        assert len(map2d.shape)==2, 'img2d must have only 2 dimenions, found {}'.format(map2d.shape)
        map3d = np.expand_dims(map2d, axis=0)
        map3d = np.repeat(map3d,3,axis=0)
        # import ipdb;ipdb.set_trace()
        return cls(data=map3d,ordering=Ordering.CHW,vc=ValueClass.FLOAT01)

    @classmethod
    def from_2d_wh_torch(cls,img2d):
        # assumes img2d has 2 dimensions
        map2d = img2d.cpu().numpy().squeeze()
        assert len(map2d.shape)==2, 'img2d must have only 2 dimenions, found {}'.format(map2d.shape)
        map3d = np.expand_dims(map2d, axis=0)
        map3d = np.repeat(map3d,3,axis=0)
        # import ipdb;ipdb.set_trace()
        return cls(data=map3d,ordering=Ordering.CHW,vc=ValueClass.FLOAT01)