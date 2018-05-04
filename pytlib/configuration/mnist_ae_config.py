from configuration.train_configuration import TrainConfiguration
from data_loading.sources.mnist_source import MNISTSource
from data_loading.loaders.autoencoder_loader import AutoEncoderLoader
from data_loading.loaders.multi_loader import MultiLoader
import torch.optim as optim
import torch.nn as nn
from networks.autoencoder import AutoEncoder
import random

def get_loader():
    source = MNISTSource('mnist_example',download=True)
    return AutoEncoderLoader(source,crop_size=[28,28])

loader = (get_loader,dict())
# loader = (MultiLoader,dict(loader=get_loader,loader_args=dict(),num_procs=16))
model = (AutoEncoder,dict(inchans=1))
optimizer = (optim.Adam,dict(lr=1e-3))
loss = nn.BCELoss()
train_config = TrainConfiguration(loader,optimizer,model,loss,cuda=False)
