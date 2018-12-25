from configuration.train_configuration import TrainConfiguration
from data_loading.sources.coco_source import COCOSource
from data_loading.sources.kitti_source import KITTISource
from data_loading.loaders.multiobject_detection_loader import MultiObjectDetectionLoader
from data_loading.loaders.multi_loader import MultiLoader
import torch.optim as optim
import torch.nn as nn
from networks.multi_object_detector import MultiObjectDetector
from loss_functions.multi_object_detector_loss import multi_object_detector_loss
import random

def get_loader():
    source = KITTISource('/home/ray/Data/KITTI/training',max_frames=1)
    return MultiObjectDetectionLoader(source,crop_size=[255,255],obj_types=['Car'],max_objects=100)

# def get_loader():
#     root = '/home/ray/Data/COCO/val2017'
#     annos = '/home/ray/Data/COCO/annotations/instances_val2017.json'
#     source = COCOSource(root,annos)
#     return MultiObjectDetectionLoader(source,crop_size=[255,255],obj_types=['Car'],max_objects=100)

loader = (get_loader,dict())
# loader = (MultiLoader,dict(loader=get_loader,loader_args=dict(),num_procs=10))
model = (MultiObjectDetector,dict())
optimizer = (optim.Adam,dict(lr=1e-3))
loss = multi_object_detector_loss
train_config = TrainConfiguration(loader,optimizer,model,loss,True)
