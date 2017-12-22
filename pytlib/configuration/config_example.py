from configuration.train_configuration import TrainConfiguration
from data_loading.sources.kitti_source import KITTISource
from data_loading.loaders.autoencoder_loader import AutoEncoderLoader
from data_loading.loaders.multi_loader import MultiLoader
import torch.optim as optim
import torch.nn as nn
from networks.autoencoder import AutoEncoder

def get_loader():
    source = KITTISource('/home/ray/Data/KITTI/training',max_frames=10)
    loader_params = {'crop_size':[255,255],'obj_types':['Car']}
    return AutoEncoderLoader(source,loader_params)

# loader + args for the loader
loader = (MultiLoader,dict(loader=get_loader,loader_args=dict(),num_procs=2))

# model + args for the model
model = (AutoEncoder,dict())

# optimizer + args
optimizer = (optim.Adam,dict(lr=1e-3))

# the loss function
loss = nn.BCELoss()

# create the config object, this object is then used by the trainer to access all the above
# components for training a model
train_config = TrainConfiguration(loader,optimizer,model,loss,False)

