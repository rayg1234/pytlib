from interface import Interface, implements
from data_loading.sources.source import Source
from torchvision import datasets
from image.box import Box
from image.frame import Frame
from image.object import Object
from image.ptimage import PTImage
from pycocotools import COCO,mask 
import numpy as np

# wrapper around torchvisions coco dataloader, note there is no download option since the coco set is huge
class COCOSource(implements(Source)):
    def __init__(self,root,annotation_file,train=True):
        self.path = path
        self.train = train
        self.coco = COCO(annotation_file)
        self.dataset = datasets.CocoDetection(root,annotation_file,transform=None,target_transform=None)
        self.cur = 0

    def __getitem__(self,index):
        image,labels = self.dataset[index]
        # assert 2D here
        np_arr = np.asarray(pil_img)
        ptimage = PTImage.from_numpy_array(np_arr)
        objects = []
        for t in labels:
            box = Box.from_xywh(t['bbox'])
            obj_type = self.coco.loadCats(t['category_id'])['name']
            polygon = t['segmentation'] if 'segmentation' in t else None
            # reshape to 2d poly, assume its convex hull?
            if polygon:
                polygon = np.array(polygon).reshape((int(len(polygon)/2), 2))
            obj.Append(Object(box,obj_type=obj_type,polygon=polygon))
        frame = Frame.from_image_and_objects(ptimage,objects)        
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