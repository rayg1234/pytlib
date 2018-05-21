from pycocotools import mask 
import numpy as np

# This mostly wraps pycocotools mask functions for encoding and decoding polygons
# having this wrapper will allow us to make a more general polygon class later
class Polygon:
    # the data is just a 2D numpy array of points 
    def __init__(self,data):
        assert data.ndim==2
        self.data = data

    # append an extra row of 1's for affine transforms
    def augmented_matrix(self):
        return np.hstack((self.data,np.ones((self.data.shape[0],1)))).T

    @classmethod
    def from_augmented_matrix(cls,poly):
        return cls(poly[0:2,:].T)

    # merge a bunch of polygons to create a binary mask, currently use the pycocotools to do this
    @classmethod
    def create_mask(cls,polygons,width,height):
        poly1d = [list(x.data.reshape(x.data.size)) for x in polygons]
        rles = mask.frPyObjects(poly1d, height, width)
        rle = mask.merge(rles)
        m = mask.decode(rle)
        return m



