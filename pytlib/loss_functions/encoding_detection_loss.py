import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.logger import Logger
from loss_functions.vae_loss import vae_loss
from loss_functions.response_map_loss import response_map_loss
from visualization.image_visualizer import ImageVisualizer
import numpy as np
from image.ptimage import PTImage

def encoding_detection_loss(reconstruction,mu,logvar,rmap,target_recon,target_coord):
    # compute both the vae loss and the detection loss
    # TODO, these losses need to be weighted
    rloss = response_map_loss(rmap,target_coord)
    vloss = vae_loss(reconstruction,mu,logvar,target_recon)
    Logger().set('loss_component.response_loss',rloss.data.cpu()[0])
    Logger().set('loss_component.vloss',vloss.data.cpu()[0])
    return rloss + vloss 