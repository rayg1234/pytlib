from configuration.train_configuration import TrainConfiguration
from data_loading.sampler_factory import SamplerFactory
from data_loading.multi_sampler import MultiSampler
import torch.optim as optim
import torch.nn as nn
from networks.autoencoder import AutoEncoder
import random

# define these things here
use_cuda = True
loader = MultiSampler(SamplerFactory.GetAESampler,dict(source='/home/ray/Data/KITTI/training',max_frames=6000,crop_size=[100,100]),num_procs=10)
model = AutoEncoder()

# want to do this before constructing optimizer according to pytroch docs
if use_cuda:
	model.cuda()
# optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
optimizer = optim.Adam(model.parameters(),lr=1e-3)
loss = nn.BCELoss()

train_config = TrainConfiguration(loader,optimizer,model,loss,use_cuda)
