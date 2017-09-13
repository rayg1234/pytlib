from scipy.ndimage.interpolation import affine_transform
from image.box import Box
import numpy as np
import math

# class to chain affine transforms
# if several affines are chained
# ie: x.append(t1); x.append(t2),
# then t1 is applied first followed by t2
class Affine:

    @classmethod
    def translation(cls,x):
        assert len(x)==2,'translation must be a 2-vector!'
        return np.array([[1,0,x[0]],[0,1,x[1]],[0,0,1]])

    @classmethod
    def scaling(cls,x):
        assert len(x)==2,'scaling must be a 2-vector!'
        return np.array([[x[0],0,0],[0,x[1],0],[0,0,1]])

    @classmethod
    def rotation(cls,x):
        return np.array([[math.cos(x),-math.sin(x),0],[math.sin(x),math.cos(x),0],[0,0,1]])

    @classmethod
    def identity(cls):
        return np.array([[1,0,0],[0,1,0],[0,0,1]])

    @classmethod
    def from_box(cls,box):
        mins = box.xy_min()
        affine = Affine()
        affine.append(Affine.translation(mins))
        return affine

    def __init__(self):
        self.transform = self.identity()
        self.inverse = self.identity()

    # matrix input has to be non-singular
    def append(self,matrix):
        self.transform = np.dot(matrix,self.transform)
        self.inverse = np.dot(self.inverse,np.linalg.inv(matrix))

    def apply_to_coords(self,input):
        return np.dot(self.transform,input)

    def unapply_to_coords(self,input):
        return np.dot(self.inverse,input)

    def apply_to_image(self,image,output_size=[]):
        # either use scipy here or implement apply/interpolate scheme

        pass

    def unapply_to_image(self,image):
        # use scipy here
        pass


