from configuration.train_configuration import TrainConfiguration
from data_loading.sources.stanford_cars_source import StanfordCarsSource
from data_loading.samplers.multi_sampler import MultiSampler
from data_loading.samplers.autoencoder_sampler import AutoEncoderSampler
import torch.optim as optim
import torch.nn as nn
from networks.vae import VAE
from loss_functions.vae_loss import vae_loss
import random

def get_sampler():
	source = StanfordCarsSource(cars_dir='/home/ray/Data/StanfordCars/cars_train',
								labels_mat='/home/ray/Data/StanfordCars/devkit/cars_train_annos.mat')
	return AutoEncoderSampler(source,{'crop_size':[100,100],'obj_types':'car'})
loader = (MultiSampler,dict(loader=get_sampler,loader_args=dict(),num_procs=10))
model = (VAE,dict(encoding_size=128,training=True))
optimizer = (optim.Adam,dict(lr=1e-3))
loss = vae_loss
train_config = TrainConfiguration(loader,optimizer,model,loss,use_cuda)
