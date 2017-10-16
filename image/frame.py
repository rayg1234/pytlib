from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.patches as patches
from image.image_utils import cudnn_np_to_PIL, PIL_to_cudnn_np

# stores a numpy representation of an image along with all the objects in that image
# default in CUDNN storage order. The image is loaded with PIL.
# note we only store the numpy image and not the PIL image, this makes some processes like
# visualization slower but makes the interface simpler.
class Frame:
    def __init__(self,image_path='',objects=[]):
        self.image_path = image_path
        self.objects = objects
        # lazy instantiation
        self.image = None
    
    # cudnn storage order is BCHW and PIL uses WHC arrays
    #
    # for Numpy 'C' Style row-major arrays, the first dimension is the
    # slowest changing dimension (last is fastest changing), and thus continguous slices of memory is across the last dim
    # so for numpy c-style arraynd, we should prefer BCHW for accessing single elements from a batch
    # the pytorch tensor should also have this memory layout
    def get_image(self):
        if self.image is None:
            assert os.path.isfile(self.image_path), "cant open file: %s" % self.image_path
            self.image = PIL_to_cudnn_np(Image.open(self.image_path, 'r'))
            # self.image = np.asarray(pil_im)
        return self.image

    # convert image back to PIL format (WHC)
    def get_pil_image(self):
        if self.image is None:
            self.get_image()    
        return cudnn_np_to_PIL(self.image)

    def get_objects(self):
        return self.objects

    def show_raw_image(self):
        fig,ax = plt.subplots(1,figsize=(15, 8))
        ax.imshow(self.get_pil_image())
        plt.show()

    def show_image(self):
        if self.image is None:
            self.get_image()

        # Create figure and axes
        fig,ax = plt.subplots(1,figsize=(15, 8))
        ax.imshow(self.get_pil_image())
        for obj in self.objects:
            rect = patches.Rectangle(obj.box.xy_min(),obj.box.edges()[0],obj.box.edges()[1],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            ax.text(obj.box.xmin, obj.box.ymin, str(obj.unique_id) + ' ' + str(obj.obj_type), 
                color='white', fontsize=12, bbox={'facecolor':'red', 'alpha':0.5, 'pad':2})
        plt.show()