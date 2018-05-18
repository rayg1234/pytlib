from scipy.ndimage.interpolation import affine_transform
from scipy.ndimage.interpolation import map_coordinates
from image.box import Box
from image.ptimage import PTImage,Ordering,ValueClass
from image.polygon import Polygon
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

    def __init__(self,order=3):
        self.transform = self.identity()
        self.inverse = self.identity()
        self.interp_order = order

    # matrix input has to be non-singular
    def append(self,matrix):
        self.transform = np.dot(matrix,self.transform)
        self.inverse = np.dot(self.inverse,np.linalg.inv(matrix))

    def apply_to_box(self,box):
        transformed_box = np.dot(self.transform,box.augmented_matrix())
        return Box.from_augmented_matrix(transformed_box)

    def apply_to_polygons(self,polygons):
        transformed_polys = []
        for p in polygons:
            transformed_polygon = np.dot(self.transform,p.augmented_matrix())
            transformed_polys.append(Polygon.from_augmented_matrix(transformed_polygon))
        return transformed_polys

    def unapply_to_box(self,box):
        transformed_box = np.dot(self.inverse,box.augmented_matrix())
        return Box.from_augmented_matrix(transformed_box)

    # optionally store the original images
    # either use scipy here or implement apply/interpolate scheme

    # Note for applying affine to HWC numpy arrays
    # we need have x and y interchanged from coordinate space
    def apply_to_image(self,image,_output_size):
        output_size = [int(_output_size[0]),int(_output_size[1])]
        inverse_transform = self.inverse.copy()
        inverse_transform[0:2,0:2] = self.inverse[1:None:-1,1:None:-1]
        inverse_transform[0:2,2] = self.inverse[1:None:-1,2]

        assert image.ordering == Ordering.HWC, 'Ordering must be HWC to apply the affine transform!'
        img_data = image.get_data()
        # check if image only has 1 channel, duplicate the channels
        if len(img_data.shape)==2:
            img_data = np.stack((img_data,)*3,axis=2)
        assert len(img_data.shape)==3, 'Input image must have 3 channels! found {}'.format(image_data.shape)
        newimage = PTImage(data=np.empty([output_size[0],output_size[1],img_data.shape[2]],dtype=image.vc['dtype']),ordering=Ordering.HWC,vc=image.vc)
        newimage_data = newimage.get_data()
        # print self.inverse
        # print inverse_transform
        # print output_size

        # for i in range(0,image.data.shape[2]):
        #     newimage.data[:,:,i] = affine_transform(image.data[:,:,i],
        #                                             self.inverse[0:2,0:2],
        #                                             offset=-self.transform[0:2,2],
        #                                             output_shape=output_size).astype(image.vc['dtype'])
        
        # scipy's affine_transform sucks, it only accepts 2x2 affine matrices and 
        # you have to specify the offset from the input using my own affine
        # Going to use map_coordinates apply the affine and interpolation separately

        # 1) first create an augmented matrix of 3 x (m*n) output points 
        px,py = np.mgrid[0:output_size[0]:1,0:output_size[1]:1]
        points = np.c_[px.ravel(), py.ravel()]
        points_aug = np.concatenate((points,np.ones((points.shape[0],1))),axis=1)

        # 2) next apply the inverse transform to find the input points to sample at
        inv_points = np.dot(inverse_transform,points_aug.T)

        # 3) use map_coordinates to do a interpolation on the input image at the required points
        for i in range(0,img_data.shape[2]):
            newimage_data[:,:,i] = map_coordinates(img_data[:,:,i],inv_points[0:2,:],order=self.interp_order).reshape(output_size)

        return newimage

    def unapply_to_image(self,image):
        assert False, 'inverse affine to image not allowed here'


