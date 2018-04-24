from interface import Interface, implements
from data_loading.sources.source import Source
from torchvision import datasets
from image.box import Box
from image.frame import Frame
from image.object import Object
from image.ptimage import PTImage
import numpy as np

# this is just a wrapper around pytorch's mnist data loader
class MNISTSource(implements(Source)):
    def __init__(self,path,train=True, download=False):
        self.path = path
        self.train = train
        self.dataset = datasets.MNIST(path, train=train, download=download)
        self.cur = 0

    def __getitem__(self,index):
        pil_img,label = self.dataset[index]
        # assert 2D here
        np_arr = np.asarray(pil_img)
        np_arr = np.expand_dims(np_arr, axis=2)
        # create the PTImage, and object that span the frame
        # add extra channel dimension

        ptimage = PTImage.from_numpy_array(np_arr)
        obj = Object(Box(0,0,pil_img.size[0],pil_img.size[1]))
        frame = Frame.from_image_and_objects(ptimage,[obj])        
        return frame

    def next(self):
        if self.cur >= len(self.dataset):
            raise StopIteration
        else:
            frame = self.__getitem__(self.cur)
            self.cur+=1
            return frame

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self 