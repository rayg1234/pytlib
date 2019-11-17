from __future__ import division
from builtins import range
from past.utils import old_div
import torch
import torch.nn.functional as F
from utils.logger import Logger
from loss_functions.response_map_loss import response_map_loss
from loss_functions.vae_loss import vae_loss
import numpy as np

# TODO, alot of these ops could be inplaced to reduce memory use and improve compute
def pearson_correlation_loss(x1,x2,eps=1e-6):
    assert len(x1.size()) == 1 and len(x2.size()) ==1 , 'Sizes must be the same and one-dimensional' 
    m1 = torch.mean(x1,0,keepdim=True)
    m2 = torch.mean(x2,0,keepdim=True)
    c1 = (torch.add(x1,-m1)).div(m1+eps)
    c2 = (torch.add(x2,-m2)).div(m2+eps)
    numerator = torch.sum(c1.mul(c2),0,keepdim=True)
    denom1 = torch.sqrt(torch.sum(c1.mul(c1),0,keepdim=True))
    denom2 = torch.sqrt(torch.sum(c2.mul(c2),0,keepdim=True))
    cor = numerator.div( torch.add(denom1.mul(denom2),eps))
    return cor

def triplet_correlation_loss(anchor,pos,neg,dummy_target,margin=1.0,eps=1e-6):
    # for now, debatch this computation, to batch properly need to figure out how to broadcast in torch
    batch_size = anchor.size(0)
    pcps,pcns = [],[]
    ne = old_div(anchor.nelement(),batch_size)
    for i in range(0,batch_size):
        pcps.append(pearson_correlation_loss(anchor[i,:].view(ne),pos[i,:].view(ne)))
        pcns.append(pearson_correlation_loss(anchor[i,:].view(ne),neg[i,:].view(ne)))
    pcp = torch.stack(pcps,0)
    pcn = torch.stack(pcns,0)
    dist_hinge = torch.clamp(margin + pcn - pcp, min=0.0)
    loss = torch.mean(dist_hinge)
    return loss

def triplet_correlation_loss2(anchor,pcp,pcn,recon_output,mu,logvar,pos_map,neg_map,recon_target):
    # down sample the map (or upsample the response)
    # use avgpool + rounding
    # take the ratio of the spatial extends

    vloss = vae_loss(recon_output,mu,logvar,recon_target)
    pool_kernel = old_div(np.array(pos_map.size()[1:]),np.array(pcp.size()[-2:]))
    pos_map_resized = torch.round(F.avg_pool2d(pos_map,tuple(pool_kernel)))
    neg_map_resized = torch.round(F.avg_pool2d(neg_map,tuple(pool_kernel)))

    nloss = F.binary_cross_entropy_with_logits(pcn.squeeze(),neg_map_resized.squeeze())
    ploss = F.binary_cross_entropy_with_logits(pcp.squeeze(),pos_map_resized.squeeze())
    # nloss = response_map_loss(pcn.squeeze())
    # ploss = response_map_loss(pcp.squeeze(),boxes)
    Logger().set('loss_component.anchor_mean',anchor.data.mean().item())
    Logger().set('loss_component.anchor_std',anchor.data.std().item())
    Logger().set('loss_component.ploss2',ploss.data.cpu().item())
    Logger().set('loss_component.nloss2',nloss.data.cpu().item())
    Logger().set('loss_component.vloss',vloss.data.cpu().item())
    return ploss+nloss+vloss