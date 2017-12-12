from configuration.train_configuration import TrainConfiguration
from data_loading.sources.kitti_source import KITTISource
from data_loading.samplers.autoencoder_sampler import AutoEncoderSampler
from data_loading.samplers.multi_sampler import MultiSampler
import torch.optim as optim
import torch.nn as nn
from networks.autoencoder import AutoEncoder
import random

# define these things here
def get_sampler():
    source = KITTISource('/home/ray/Data/KITTI/training',max_frames=10000)
    sampler_params = {'crop_size':[255,255],'obj_types':['Car']}
    return AutoEncoderSampler(source,sampler_params)

loader = (MultiSampler,dict(loader=get_sampler,loader_args=dict(),num_procs=10))
model = (AutoEncoder,_)
optimizer = (optim.Adam,dict(lr=1e-3))
loss = nn.BCELoss()
train_config = TrainConfiguration(loader,optimizer,model,loss,True)
