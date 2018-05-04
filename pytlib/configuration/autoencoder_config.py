from configuration.train_configuration import TrainConfiguration
from data_loading.sources.kitti_source import KITTISource
from data_loading.loaders.autoencoder_loader import AutoEncoderLoader
from data_loading.loaders.multi_loader import MultiLoader
import torch.optim as optim
import torch.nn as nn
from networks.autoencoder import AutoEncoder
import random

def get_loader():
    source = KITTISource('/home/ray/Data/KITTI/training',max_frames=10000)
    return AutoEncoderLoader(source,crop_size=[255,255],obj_types=['Car'])

loader = (MultiLoader,dict(loader=get_loader,loader_args=dict(),num_procs=10))
model = (AutoEncoder,dict())
optimizer = (optim.Adam,dict(lr=1e-3))
loss = nn.BCELoss()
train_config = TrainConfiguration(loader,optimizer,model,loss,True)
