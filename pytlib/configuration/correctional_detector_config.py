from configuration.train_configuration import TrainConfiguration
from data_loading.sources.stanford_cars_source import StanfordCarsSource
from data_loading.samplers.multi_sampler import MultiSampler
from data_loading.samplers.encoding_detection_sampler import EncodingDetectionSampler
import torch.optim as optim
import torch.nn as nn
from networks.correlational_detector import CorrelationalDetector
from loss_functions.response_map_loss import response_map_loss
import random

# define these things here
use_cuda = True

def get_sampler():
	source = StanfordCarsSource(cars_dir='/home/ray/Data/StanfordCars/cars_train',
								labels_mat='/home/ray/Data/StanfordCars/devkit/cars_train_annos.mat')
	return EncodingDetectionSampler(source,{'crop_size':[127,127],'frame_size':[255,255],'obj_types':'car'})

# todo, replace module based random seed
# loader = get_sampler()
loader = MultiSampler(get_sampler,dict(),num_procs=8)
model = CorrelationalDetector()

# want to do this before constructing optimizer according to pytroch docs
if use_cuda:
	model.cuda()
# optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
optimizer = optim.Adam(model.parameters(),lr=1e-3)
# todo the loss function doesnt match with the samplers args right now
loss = response_map_loss

train_config = TrainConfiguration(loader,optimizer,model,loss,use_cuda)
