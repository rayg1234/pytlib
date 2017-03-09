from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.patches as patches

class Frame:
    def __init__(self,image_path='',objects=[]):
        self.image_path = image_path
        self.objects = objects
        self.image = None
        
    def load_image(self):
        if not self.image:
            assert os.path.isfile(self.image_path), "cant open file: %s" % self.image_path
            pil_im = Image.open(self.image_path, 'r')
            self.image = np.asarray(pil_im)
        
    def show_raw_image(self):
        pass

    def show_image(self):
        # Create figure and axes
        fig,ax = plt.subplots(1,figsize=(15, 8))
        ax.imshow(self.image)
        for obj in self.objects:
            rect = patches.Rectangle(obj.box.xy_min(),obj.box.edges()[0],obj.box.edges()[1],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            ax.text(obj.box.xmin, obj.box.ymin, str(obj.unique_id) + ' ' + str(obj.obj_type), 
                color='white', fontsize=12, bbox={'facecolor':'red', 'alpha':0.5, 'pad':2})
        plt.show()