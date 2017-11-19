from configuration.train_configuration import TrainConfiguration
from data_loading.sources.stanford_cars_source import StanfordCarsSource
from data_loading.samplers.multi_sampler import MultiSampler
from data_loading.samplers.autoencoder_sampler import AutoEncoderSampler
import torch.optim as optim
import torch.nn as nn
from networks.vae import VAE
from loss_functions.vae_loss import vae_loss
import random

# define these things here
use_cuda = True

def getSampler():
	source = StanfordCarsSource(cars_dir='/home/ray/Data/StanfordCars/cars_train',
								labels_mat='/home/ray/Data/StanfordCars/devkit/cars_train_annos.mat')
	return AutoEncoderSampler(source,{'crop_size':[100,100],'obj_types':'car'})

# todo, replace module based random seed
# loader = getSampler()
loader = MultiSampler(getSampler,dict(),num_procs=10)
model = VAE(encoding_size=128,training=True)

# want to do this before constructing optimizer according to pytroch docs
if use_cuda:
	model.cuda()
# optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
optimizer = optim.Adam(model.parameters(),lr=1e-3)
loss = vae_loss

train_config = TrainConfiguration(loader,optimizer,model,loss,use_cuda)
