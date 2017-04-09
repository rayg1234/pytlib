from transformer import Transformer
from image.frame import Frame
from image.box import Box
from image.object import Object
from data_loading.sample import Sample
from image.affine import Affine
import numpy as np

import torch

class NoFramesException(Exception):
    pass

class CropTransformer(Transformer):
    def __init__(self,crop_size,obj_types=set()):
        self.crop_size = crop_size
        self.obj_types = obj_types

    def transform(self,frames):
        if not len(frames):
            raise NoFramesException('No frames given to the transformer!')

        frames[0].load_image()
        # frame numpy ndarray with dims (HxWxC)
        frame_shape = np.asarray(frames[0].image).shape

        # data = np.array((len(crop_objs),self.crop_size[0],self.crop_size[1],3),dtype=float,order='C')
        # targets = np.array((len(crop_objs),self.crop_size[0],self.crop_size[1],3),dtype=float,order='C')
        data,targets = None,None

        count = 0
        for frame in frames:
            frame.load_image()
            crop_objs = filter(lambda x: x.obj_type in self.obj_types,frame.objects)
            print 'Num crop objs in sample: {0}'.format(len(crop_objs))

            for crop in crop_objs:
                # crop and resize
                crop_image = frame.image.crop(crop.box.to_single_array())
                resized_image = crop_image.resize(self.crop_size)

                affine = Affine()
                scalex = float(self.crop_size[0])/crop.box.edges()[0]
                scaley = float(self.crop_size[1])/crop.box.edges()[1]
                affine.append(Affine.scaling(scalex,scaley))
                affine.append(Affine.translation(crop.box.xy_min()[0],crop.box.xy_min()[1]))
                # import pdb;pdb.set_trace()
                transformed_crop_box = Box.from_augmented_matrix(affine.apply_to_coords(crop.box.augmented_matrix()))

                np_img = np.asarray(resized_image)
                if data is not None:
                    # stack crop on top of tensor along first aWWxis
                    reshaped_img = np_img.reshape([1]+list(np_img.shape))
                    data = np.concatenate((data,reshaped_img),axis=0)
                    targets = np.vstack((targets,transformed_crop_box.to_single_np_array()))
                else:
                    # new data
                    data = np.asarray(resized_image)
                    data = data.reshape([1]+list(data.shape))
                    targets = transformed_crop_box.to_single_np_array()

        print 'targets shape {0}'.format(targets.shape)
        print 'data shape {0}'.format(data.shape)

        # import pdb;pdb.set_trace()
        sample = Sample(torch.Tensor(data.astype(float)),torch.Tensor(targets.astype(float)))
        # create sample tensor to return
        # sample.data = torch.Tensor(B,frame_shape[0],frame_shape[1],frame_shape[2])
        # sample.target = torch.Tensor(B,4)

        return sample
