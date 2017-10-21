from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.patches as patches
from image.image_utils import cudnn_np_to_PIL, PIL_to_cudnn_np
from image.ptimage import PTImage,Ordering,ValueClass

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

    def get_image(self):
        if self.image is None:
            self.image = PTImage(pil_image_path=self.image_path)
        return self.image

    def get_objects(self):
        return self.objects

    def show_raw_image(self):
        self.image.visualize()

    def show_image_with_labels(self):
        fig,ax = plt.subplots(1,figsize=(15, 8))
        ax.imshow(self.get_image().to_order_and_class(Ordering.WHC,ValueClass.BYTE0255).data)
        for obj in self.objects:
            rect = patches.Rectangle(obj.box.xy_min(),obj.box.edges()[0],obj.box.edges()[1],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            coord_string = str([int(round(x)) for x in obj.box.to_single_array()])
            ax.text(obj.box.xmin, obj.box.ymin, str(obj.unique_id) + ' ' + str(obj.obj_type) + ' ' + coord_string, 
                color='white', fontsize=12, bbox={'facecolor':'red', 'alpha':0.5, 'pad':2})
        plt.show()