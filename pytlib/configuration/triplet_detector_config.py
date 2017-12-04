from configuration.train_configuration import TrainConfiguration
from data_loading.samplers.multi_sampler import MultiSampler
from data_loading.sources.kitti_source import KITTISource
from data_loading.samplers.triplet_detection_sampler import TripletDetectionSampler
import torch.optim as optim
import torch.nn as nn
from networks.triplet_correlational_detector import TripletCorrelationalDetector
from loss_functions.triplet_correlation_loss import triplet_correlation_loss
import random

# define these things here
use_cuda = True
def get_sampler():
    source = KITTISource('/home/ray/Data/KITTI/training',max_frames=6000)
    sampler_params = {'crop_size':[100,100],'obj_types':['Car']}
    return TripletDetectionSampler(source,sampler_params)

# loader = get_sampler()
loader = MultiSampler(get_sampler,dict(),num_procs=8)
model = TripletCorrelationalDetector()

# want to do this before constructing optimizer according to pytroch docs
if use_cuda:
	model.cuda()
# optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
optimizer = optim.Adam(model.parameters(),lr=1e-4)
loss = triplet_correlation_loss

train_config = TrainConfiguration(loader,optimizer,model,loss,use_cuda)
