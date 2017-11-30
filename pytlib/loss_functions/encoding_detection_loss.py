from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.logger import Logger
from loss_functions.vae_loss import vae_loss

def box_loss(output_box,target_box):
	return F.smooth_l1_loss(output_box,target_box)

def encoding_detection_loss(reconstruction,mu,logvar,rmap,out_coords,target_recon,target_coord):
    # compute both the vae loss and the detection loss
    # TODO, these losses need to be weighted
    bloss = box_loss(out_coords,target_coord)
    vloss = vae_loss(reconstruction,mu,logvar,target_recon)
    Logger().set('loss_component.box_loss',bloss.data.cpu()[0])
    Logger().set('loss_component.vloss',vloss.data.cpu()[0])
    return bloss + vloss 