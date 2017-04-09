from scipy.ndimage.interpolation import affine_transform
import numpy as np
import math

# class to chain affine transforms
# if several affines are chained
# ie: x.append(t1); x.append(t2),
# then t2 is applied first followed by t1
class Affine:

    @classmethod
    def translation(cls,a,b):
        return np.array([[1,0,a],[0,1,b],[0,0,1]])
  
    @classmethod
    def scaling(cls,a,b):
        return np.array([[a,0,0],[0,b,0],[0,0,1]])

    @classmethod
    def rotation(cls,x):
        return np.array([[math.cos(x),-math.sin(x),0],[math.sin(x),math.cos(x),0],[0,0,1]])

    @classmethod
    def identity(cls):
        return np.array([[1,0,0],[0,1,0],[0,0,1]])

    def __init__(self):
        self.transform = self.identity()
        self.inverse = self.identity()

    # matrix input has to be non-singular
    def append(self,matrix):
        self.transform = np.dot(self.transform,matrix)
        self.inverse = np.dot(np.linalg.inv(matrix),self.inverse)

    def apply_to_coords(self,input):
        return np.dot(self.transform,input)

    def unapply_to_coords(self,input):
        return np.dot(self.inverse,input)

    def apply_to_image(self,image):
        # use scipy here
        pass

    def unapply_to_image(self,image):
        # use scipy here
        pass


