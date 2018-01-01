from configuration.train_configuration import TrainConfiguration
from configuration.test_configuration import TestConfiguration
from data_loading.loaders.multi_loader import MultiLoader
from data_loading.loaders.triplet_detection_loader import TripletDetectionLoader
from data_loading.sources.kitti_source import KITTISource
import torch.optim as optim
import torch.nn as nn
from networks.triplet_correlational_detector import TripletCorrelationalDetector
from loss_functions.triplet_correlation_loss import triplet_correlation_loss,triplet_correlation_loss2
import random

def get_loader(mode):
    path = '/home/ray/Data/KITTI/testing' if mode=='test' else '/home/ray/Data/KITTI/training'
    source = KITTISource(path,max_frames=10000)
    loader_params = {'crop_size':[255,255],'anchor_size':[127,127],'obj_types':['Car'],'mode':mode}
    return TripletDetectionLoader(source,loader_params)

loader_test = (MultiLoader,dict(loader=get_loader,loader_args=dict(mode='test'),num_procs=1))
loader_train = (MultiLoader,dict(loader=get_loader,loader_args=dict(mode='train'),num_procs=10))
model = (TripletCorrelationalDetector,dict(anchor_size=(127,127)))
optimizer = (optim.Adam,dict(lr=1e-4))
loss = triplet_correlation_loss2
train_config = TrainConfiguration(loader_train,optimizer,model,loss,cuda=True)
test_config = TestConfiguration(loader_test,model,loss,cuda=True)
