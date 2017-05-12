from configuration.train_configuration import TrainConfiguration
from data_loading.data_loader import DataLoaderFactory
import torch.optim as optim
import torch.nn as nn
from networks.autoencoder import AutoEncoder

# define these things here
loader = DataLoaderFactory.GetKITTILoader()
model = AutoEncoder()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss = nn.BCELoss()

train_config = TrainConfiguration(loader,optimizer,model,loss)
