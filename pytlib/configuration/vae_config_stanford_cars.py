from configuration.train_configuration import TrainConfiguration
from data_loading.sources.stanford_cars_source import StanfordCarsSource
from data_loading.loaders.multi_loader import MultiLoader
from data_loading.loaders.autoencoder_loader import AutoEncoderLoader
import torch.optim as optim
import torch.nn as nn
from networks.vae import VAE
from loss_functions.vae_loss import vae_loss
import random

def get_loader():
	source = StanfordCarsSource(cars_dir='/home/ray/Data/StanfordCars/cars_train',
								labels_mat='/home/ray/Data/StanfordCars/devkit/cars_train_annos.mat')
	return AutoEncoderLoader(source,{'crop_size':[100,100],'obj_types':'car'})
loader = (MultiLoader,dict(loader=get_loader,loader_args=dict(),num_procs=10))
model = (VAE,dict(encoding_size=128,training=True))
optimizer = (optim.Adam,dict(lr=1e-3))
loss = vae_loss
train_config = TrainConfiguration(loader,optimizer,model,loss,True)
