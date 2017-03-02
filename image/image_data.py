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
        
    def LoadImage(self):
        assert os.path.isfile(self.image_path), "cant open file: %s" % self.image_path
        pil_im = Image.open(self.image_path, 'r')
        self.image = np.asarray(pil_im)
        
    def ShowRawImage(self):
        pass

    def ShowImage(self):
        # Create figure and axes
        fig,ax = plt.subplots(1,figsize=(15, 8))
        ax.imshow(self.image)
        for obj in self.objects:
            rect = patches.Rectangle(obj.box.XYmin(),obj.box.Edges()[0],obj.box.Edges()[1],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
        plt.show()
        
class Box:
    def __init__(self,xmin,xmax,ymin,ymax):
        assert xmax>xmin and ymax>ymin, "xmax>xmin and ymax>ymin!"
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax
    
    @classmethod
    def FromDoubleArray(cls,box):
        return cls(box[0][0],box[0][1],box[1][0],box[1][1])
    
    def XYmin(self):
        return (self.xmin,self.ymin)
    
    def Edges(self):
        return (self.xmax-self.xmin,self.ymax-self.ymin)
    
    def Area(self):
        return (self.xmax-self.xmin)*(self.ymax-self.ymin)
        
# and Object contains an id,box,and type
class Object:
    def __init__(self,box,unique_id=-1,obj_type='head'):
        self.box = box
        self.unique_id = unique_id
        self.obj_type = obj_type
        
    @classmethod
    def FromDict(cls,objdict):
        #print objdict
        box = Box(objdict['box']['min'][0],objdict['box']['max'][0],objdict['box']['min'][1],objdict['box']['max'][1])
        uid = objdict['unique_identifier']
        obj_type = objdict['attributes']['type']
        return cls(box,uid,obj_type)
        
