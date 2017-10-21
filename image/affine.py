from scipy.ndimage.interpolation import affine_transform
from scipy.ndimage.interpolation import map_coordinates
from image.box import Box
from image.ptimage import PTImage,Ordering,ValueClass
import numpy as np
import math

# class to chain affine transforms
# if several affines are chained
# ie: x.append(t1); x.append(t2),
# then t1 is applied first followed by t2

# TODO, rename to ImageTransform, add intensity transform as well
class Affine:

    # Note These matrices have x and y interchanged from coordinate space
    # y is mapped to rows in this system (first dimension of HWC numpy arrays)
    # and x is mapped to cols
    @classmethod
    def translation(cls,x):
        assert len(x)==2,'translation must be a 2-vector!'
        return np.array([[1,0,x[1]],[0,1,x[0]],[0,0,1]])

    @classmethod
    def scaling(cls,x):
        assert len(x)==2,'scaling must be a 2-vector!'
        return np.array([[x[1],0,0],[0,x[0],0],[0,0,1]])

    @classmethod
    def rotation(cls,x):
        return np.array([[math.cos(x),math.sin(x),0],[-math.sin(x),math.cos(x),0],[0,0,1]])

    @classmethod
    def identity(cls):
        return np.array([[1,0,0],[0,1,0],[0,0,1]])

    @classmethod
    def from_box(cls,box):
        mins = box.xy_min()
        affine = Affine()
        affine.append(Affine.translation(mins))
        return affine

    def __init__(self,store_original=False,order=3):
        self.transform = self.identity()
        self.inverse = self.identity()
        self.store_original = store_original
        self.original_image = None
        self.interp_order = order

    # matrix input has to be non-singular
    def append(self,matrix):
        self.transform = np.dot(matrix,self.transform)
        self.inverse = np.dot(self.inverse,np.linalg.inv(matrix))

    def apply_to_coords(self,input):
        return np.dot(self.transform,input)

    def unapply_to_coords(self,input):
        return np.dot(self.inverse,input)

    # optionally store the original images
    # either use scipy here or implement apply/interpolate scheme

    def apply_to_image(self,image,_output_size):
        output_size = _output_size[::-1]
        if self.store_original:
            self.original_image = image
        assert image.ordering == Ordering.WHC, 'Ordering must be WHC to apply the affine transform!'
        image.get_data()
        newimage = PTImage(data=np.empty(output_size + [image.data.shape[2]],dtype=image.vc['dtype']),ordering=Ordering.WHC,vc=image.vc)
        # print self.transform
        # print self.inverse
        # print output_size

        # for i in range(0,image.data.shape[2]):
        #     newimage.data[:,:,i] = affine_transform(image.data[:,:,i],
        #                                             self.inverse[0:2,0:2],
        #                                             offset=-self.transform[0:2,2],
        #                                             output_shape=output_size).astype(image.vc['dtype'])
        
        # scipy's affine_transform sucks, it only accepts 2x2 affine matrices and you have tos specify the offset from the input
        # using my own affine

        # 1) first create an augmented matrix of 3 x (m*n) output points 
        px,py = np.mgrid[0:output_size[0]:1,0:output_size[1]:1]
        points = np.c_[px.ravel(), py.ravel()]
        points_aug = np.concatenate((points,np.ones((points.shape[0],1))),axis=1)

        # 2) next apply the inverse transform to find the input points to sample at
        inv_points = np.dot(self.inverse,points_aug.T)

        # 3) use map_coordinates to do a interpolation on the input image at the required points
        for i in range(0,image.data.shape[2]):
            newimage.data[:,:,i] = map_coordinates(image.data[:,:,i],inv_points[0:2,:],order=self.interp_order).reshape(output_size)

        return newimage


    def unapply_to_image(self,image):
        # use scipy here
        if self.store_original and self.original_image is not None:
            return original_image
        else:
            assert False, 'inverse affine to image not allowed here'


