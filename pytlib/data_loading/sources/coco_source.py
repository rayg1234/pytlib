from interface import Interface, implements
from data_loading.sources.source import Source
from torchvision import datasets
from image.box import Box
from image.frame import Frame
from image.object import Object
from image.ptimage import PTImage
from image.polygon import Polygon
from pycocotools.coco import COCO
import numpy as np

# wrapper around torchvisions coco dataloader, note there is no download option since the coco set is huge
class COCOSource(implements(Source)):
    def __init__(self,root,annotation_file,train=True):
        self.train = train
        self.coco = COCO(annotation_file)
        self.dataset = datasets.CocoDetection(root,annotation_file,transform=None,target_transform=None)
        self.cur = 0

    def __getitem__(self,index):
        image,labels = self.dataset[index]
        np_arr = np.asarray(image)
        ptimage = PTImage.from_numpy_array(np_arr)
        objects = []
        for t in labels:
            box = Box.from_xywh(t['bbox'])
            obj_type = self.coco.loadCats([t['category_id']])[0]['name']
            # convert segmentation to polygon using the pycocotools
            # note the segmentation could in one of several formats, for example the custom coco RLE,
            # to convert the RLE back to polygon is bit of a pain so I will just ignore those right now
            # according the COCO site, most of the data is in polygon form (not sure why theres a discrepency?)
            # and I'd rather not store 2D binary masks with every object.
            polygon = t.get('segmentation')
            # reshape to 2d poly, assume its convex hull?
            polys = []
            if polygon and isinstance(polygon,list):
                for seg in polygon:
                    polys.append(Polygon(np.array(seg).reshape((int(len(seg)/2), 2))))
            objects.append(Object(box,obj_type=obj_type,polygons=polys))
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