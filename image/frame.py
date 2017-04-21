from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.patches as patches

class ImageNotLoadedException(Exception):
    pass

class Frame:
    def __init__(self,image_path='',objects=[]):
        self.image_path = image_path
        self.objects = objects
        self.image = None
        self.nparray = None
    
    # loads image into numpy ndarray with dims (HxWxC) 
    # for Numpy 'C' Style row-major arrays, the first dimension is the
    # slowest changing dimension (last is fastest changing), and thus continguous slices of memory is across the last dim
    # so for numpy c-style arraynd, we should prefer B,H,W,C for accessing single elements from a batch
    # the pytorch tensor should also have this memory layout
    def get_image(self):
        if self.image is None:
            assert os.path.isfile(self.image_path), "cant open file: %s" % self.image_path
            self.image = Image.open(self.image_path, 'r')
            # self.image = np.asarray(pil_im)
        return self.image

    # cudnn storage order is BCHW and numpy to PIL gives WHC arrays
    def get_numpy_image(self):
        if self.image is None:
            self.get_image()
        if self.nparray is None:
            self.nparray = np.transpose(np.asarray(self.image),axes=(2,0,1))
        return self.nparray

    def get_objects(self):
        return self.objects

    def show_raw_image(self):
        raise NotImplementedError

    def show_image(self):
        if self.image is None:
            raise ImageNotLoadedException("Image was not loaded!")
        # Create figure and axes
        fig,ax = plt.subplots(1,figsize=(15, 8))
        ax.imshow(self.image)
        for obj in self.objects:
            rect = patches.Rectangle(obj.box.xy_min(),obj.box.edges()[0],obj.box.edges()[1],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            ax.text(obj.box.xmin, obj.box.ymin, str(obj.unique_id) + ' ' + str(obj.obj_type), 
                color='white', fontsize=12, bbox={'facecolor':'red', 'alpha':0.5, 'pad':2})
        plt.show()