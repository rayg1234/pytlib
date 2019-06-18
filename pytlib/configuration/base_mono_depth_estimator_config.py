from configuration.train_configuration import TrainConfiguration
from data_loading.sources.kitti_source import KITTISource
from data_loading.loaders.sequence_video_loader import SequenceVideoLoader
from data_loading.loaders.multi_loader import MultiLoader
import torch.optim as optim
import torch.nn as nn
from networks.base_mono_depth_estimator import BaseMonoDepthEstimator
from loss_functions.mono_depth_loss import mono_depth_loss
import random

def get_loader():
    source = KITTISource('/home/ray/Data/KITTI/tracking/training',max_frames=10000)
    return SequenceVideoLoader(source,crop_size=[512,160])
    # return SequenceVideoLoader(source,crop_size=[1024,320])

# loader = (get_loader,dict())
loader = (MultiLoader,dict(loader=get_loader,loader_args=dict(),num_procs=8))
model = (BaseMonoDepthEstimator,dict())
optimizer = (optim.Adam,dict(lr=2e-5))
loss = mono_depth_loss # placeholder
train_config = TrainConfiguration(loader,optimizer,model,loss,True)
