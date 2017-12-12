from configuration.train_configuration import TrainConfiguration
from configuration.test_configuration import TestConfiguration
from data_loading.samplers.multi_sampler import MultiSampler
from data_loading.sources.kitti_source import KITTISource
from data_loading.samplers.triplet_detection_sampler import TripletDetectionSampler
import torch.optim as optim
import torch.nn as nn
from networks.triplet_correlational_detector import TripletCorrelationalDetector
from loss_functions.triplet_correlation_loss import triplet_correlation_loss,triplet_correlation_loss2
import random

# define these things here
def get_sampler(mode):
    source = KITTISource('/home/ray/Data/KITTI/training',max_frames=10)
    sampler_params = {'crop_size':[255,255],'anchor_size':[127,127],'obj_types':['Car'],'mode':mode}
    return TripletDetectionSampler(source,sampler_params)

loader_test = (MultiSampler,dict(loader=get_sampler,loader_args=dict(mode='test'),num_procs=1))
loader_train = (MultiSampler,dict(loader=get_sampler,loader_args=dict(mode='train'),num_procs=10))
model = (TripletCorrelationalDetector,dict(anchor_size=(127,127)))
optimizer = (optim.Adam,dict(lr=1e-3))
loss = triplet_correlation_loss2
train_config = TrainConfiguration(loader_train,optimizer,model,loss,cuda=True)
test_config = TestConfiguration(loader_test,model,loss,cuda=True)
