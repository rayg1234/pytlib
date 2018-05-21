from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import copy
from image.ptimage import PTImage,Ordering,ValueClass

# stores a numpy representation of an image along with all the objects in that image
# default in CUDNN storage order. The image is loaded with PIL.
# note we only store the numpy image and not the PIL image, this makes some processes like
# visualization slower but makes the interface simpler.
class Frame:
    def __init__(self,image_path='',objects=[]):
        self.image_path = image_path
        self.objects = objects
        self.image = PTImage(pil_image_path=self.image_path,persist=False)

    @classmethod
    def from_image_and_objects(cls,ptimage,objects=[]):
        frame = cls('',objects)
        frame.image = ptimage
        return frame

    def get_objects(self):
        return self.objects

    def show_raw_image(self):
        self.image.visualize('frame image')

    def visualize(self,axes=None,display=False,title='Frame Visualization'):
        axes = self.image.visualize(axes=axes,title=title,display=False)
        # draw objects
        for obj in self.objects:
            rect = patches.Rectangle(obj.box.xy_min(),obj.box.edges()[0],obj.box.edges()[1],linewidth=1,edgecolor='r',facecolor='none')
            poly_objs = [patches.Polygon(x.data) for x in obj.polygons]
            polys = PatchCollection(poly_objs)
            axes.add_patch(rect)
            axes.add_collection(polys)
            coord_string = str([int(round(x)) for x in obj.box.to_single_array()])
            axes.text(obj.box.xmin, obj.box.ymin, str(obj.unique_id) + ' ' + str(obj.obj_type) + ' ' + coord_string, 
                color='white', fontsize=12, bbox={'facecolor':'red', 'alpha':0.5, 'pad':2})
        if display:
            plt.show(block=True)
            plt.close()