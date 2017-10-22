from configuration.train_configuration import TrainConfiguration
from data_loading.sampler_factory import SamplerFactory
import torch.optim as optim
import torch.nn as nn
from networks.autoencoder import AutoEncoder
import random

# define these things here
use_cuda = True
random.seed(1234)
loader = SamplerFactory.GetAESampler('/home/ray/Data/KITTI/training',max_frames=200,crop_size=[100,100])
model = AutoEncoder()

# want to do this before constructing optimizer according to pytroch docs
if use_cuda:
	model.cuda()
# optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
optimizer = optim.Adam(model.parameters(),lr=1e-3)
loss = nn.BCELoss()

train_config = TrainConfiguration(loader,optimizer,model,loss,use_cuda)