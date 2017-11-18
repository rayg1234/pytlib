from configuration.train_configuration import TrainConfiguration
from data_loading.sampler_factory import SamplerFactory
from data_loading.samplers.multi_sampler import MultiSampler
import torch.optim as optim
import torch.nn as nn
from networks.autoencoder import AutoEncoder

# define these things here
use_cuda = False
loader = SamplerFactory.GetAESampler(source='/home/ray/Data/KITTI/training',max_frames=200,crop_size=[100,100])
model = AutoEncoder()

# want to do this before constructing optimizer according to pytroch docs
if use_cuda:
	model.cuda()
optimizer = optim.Adam(model.parameters(),lr=1e-3)
loss = nn.BCELoss()

train_config = TrainConfiguration(loader,optimizer,model,loss,use_cuda)
