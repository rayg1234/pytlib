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

classes = ['Car','Cyclist','Pedestrian']
def get_loader():
    source = KITTISource('/home/ray/Data/KITTI/training',max_frames=10000)
    return MultiObjectDetectionLoader(source,crop_size=[1024,320],obj_types=classes,max_objects=100)

# loader = (get_loader,dict())
loader = (MultiLoader,dict(loader=get_loader,loader_args=dict(),num_procs=8))
model = (MultiObjectDetector,dict(nboxes_per_pixel=5, num_classes=len(classes)+1))
optimizer = (optim.Adam,dict(lr=1e-4))
loss = multi_object_detector_loss
train_config = TrainConfiguration(loader,optimizer,model,loss,True)
