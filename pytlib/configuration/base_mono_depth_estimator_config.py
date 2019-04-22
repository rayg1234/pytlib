from configuration.train_configuration import TrainConfiguration
from data_loading.sources.kitti_source import KITTISource
from data_loading.loaders.sequence_video_loader import SequenceVideoLoader
from data_loading.loaders.multi_loader import MultiLoader
import torch.optim as optim
import torch.nn as nn
from networks.base_mono_depth_estimator import BaseMonoDepthEstimator
import random

def get_loader():
    source = KITTISource('/home/ray/Data/KITTI/training',max_frames=10)
    return SequenceVideoLoader(source,crop_size=[1024,320])

loader = (get_loader,dict())
# loader = (MultiLoader,dict(loader=get_loader,loader_args=dict(),num_procs=8))
model = (BaseMonoDepthEstimator,dict())
optimizer = (optim.Adam,dict(lr=1e-4))
loss = None # placeholder
train_config = TrainConfiguration(loader,optimizer,model,loss,True)
