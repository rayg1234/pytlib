from transformer import Transformer
from image.frame import Frame
from data_loading.sample import Sample
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
        frame_shape = frames[0].image.shape
        
        data = None
        count = 0
        for f in frames:
            frame.load_image()
            crop_objs = filter(lambda x: x.obj_type in self.obj_types,f.objs)
            for crop in crop_objs:
                # crop and resize
                crop_image = frame.image.crop(crop.box.to_single_array())
                resized_image = crop_image.resize(self.crop_size)
                if data:
                    # stack crop on top of tensor along first axis
                    resized_image = resized_image.reshape([1]+list(resized_image.shape))
                    data = data.concatenate((data,resized_image),axis=0)
                else:
                    # new data
                    data = np.array(resized_image)
                    data = data.reshape([1]+list(data.shape))

        # need to also transform target into the correct coordinates

        sample = Sample()
        # create sample tensor to return
        sample.data = torch.Tensor(B,frame_shape[0],frame_shape[1],frame_shape[2])
        sample.target = None

        return sample
