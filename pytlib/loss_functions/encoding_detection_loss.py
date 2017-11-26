from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.logger import Logger
from loss_functions.vae_loss import vae_loss

def encoding_detection_loss(reconstruction,mu,logvar,ccmap,target_recon,target_coord):
    # compute both the vae loss and the detection loss
    # placeholder, just return vae loss for now
    return vae_loss(reconstruction,mu,logvar,target_recon)
