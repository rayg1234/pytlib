import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.logger import Logger
from loss_functions.vae_loss import vae_loss
from visualization.image_visualizer import ImageVisualizer
import numpy as np
from image.ptimage import PTImage

def response_map_loss(rmap,target_box):
    # first turn the box into a binary response map
    binarized_target_map = np.zeros(rmap.size())
    for k,tnp in enumerate(target_box.data.cpu().numpy()):
        tnp_x,tnp_y = np.array([tnp[0],tnp[2]]),np.array([tnp[1],tnp[3]])
        bounds_x = np.clip(np.round(tnp_x*rmap.data.size(2)).astype(int),0,rmap.data.size(2))
        bounds_y = np.clip(np.round(tnp_y*rmap.data.size(3)).astype(int),0,rmap.data.size(3))
        binarized_target_map[k,0,bounds_y[0]:bounds_y[1],bounds_x[0]:bounds_x[1]]=1
        # image_tmap = PTImage.from_2d_wh_torch(torch.Tensor(binarized_target_map[k,:,:,:]))
        # ImageVisualizer().set_image(image_tmap,'TMap {}'.format(k))

    target_map_var = Variable(torch.Tensor(binarized_target_map))
    target_map_var = target_map_var.cuda() if rmap.is_cuda else target_map_var
    return F.binary_cross_entropy_with_logits(rmap,target_map_var)

def encoding_detection_loss(reconstruction,mu,logvar,rmap,target_recon,target_coord):
    # compute both the vae loss and the detection loss
    # TODO, these losses need to be weighted
    rloss = response_map_loss(rmap,target_coord)
    vloss = vae_loss(reconstruction,mu,logvar,target_recon)
    Logger().set('loss_component.response_loss',rloss.data.cpu()[0])
    Logger().set('loss_component.vloss',vloss.data.cpu()[0])
    return rloss + vloss 