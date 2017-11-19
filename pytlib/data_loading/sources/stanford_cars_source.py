from interface import Interface, implements
from image.box import Box
from image.frame import Frame
from image.object import Object
from data_loading.sources.source import Source
import scipy.io
import os

class StanfordCarsSource(implements(Source)):
    def __init__(self,cars_dir,labels_mat):
        self.frames = []
        self.__load_frames(cars_dir,labels_mat)
        self.cur = 0

    def __load_frames(self,cars_dir,labels_mat):
        print 'Loading Stanford Cars Frames'
        labels = scipy.io.loadmat(labels_mat)['annotations'][0]
        # load frames with labels
        for label in labels:
            xmin, ymin, xmax, ymax, _, path = label
            box = Box(float(xmin[0][0]),float(ymin[0][0]),float(xmax[0][0]),float(ymax[0][0]))
            obj = Object(box,obj_type='car')
            image_path = os.path.join(cars_dir,path[0])
            self.frames.append(Frame(image_path,[obj]))

    def next(self):
        if self.cur >= len(self.frames):
            raise StopIteration
        else:
            ret = self.frames[self.cur]
            self.cur+=1
            return ret

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.frames)

    def __getitem__(self,index):
        return self.frames[index]

