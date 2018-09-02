from configuration.train_configuration import TrainConfiguration
from data_loading.sources.coco_source import COCOSource 
from data_loading.loaders.semantic_segmentation_loader import SegmentationLoader
from data_loading.loaders.multi_loader import MultiLoader
from networks.attention_segmenter import AttentionSegmenter
from loss_functions.segmenter_loss import RecurrentSegmeterLoss
import torch.optim as optim
import torch.nn as nn
import random

def get_loader(mode='train'):
    root = '/home/ray/Data/COCO/val2017'
    annos = '/home/ray/Data/COCO/annotations/instances_val2017.json'
    source = COCOSource(root,annos)
    return SegmentationLoader(source,max_frames=2,crop_size=[255,255],obj_types=['person'])

# loader = (get_loader,dict())
attn_grid_size=50
loader = (MultiLoader,dict(loader=get_loader,loader_args=dict(),num_procs=1))
model = (AttentionSegmenter,dict(num_classes=1,timesteps=3,attn_grid_size=attn_grid_size))
optimizer = (optim.Adam,dict(lr=1e-3))
loss = RecurrentSegmeterLoss(attn_grid_size)
train_config = TrainConfiguration(loader,optimizer,model,loss,cuda=True)
