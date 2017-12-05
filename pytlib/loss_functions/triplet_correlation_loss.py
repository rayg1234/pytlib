from torch.autograd import Variable
import torch
import torch.nn.functional as F
from utils.logger import Logger

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
    ne = anchor.nelement()/batch_size
    for i in range(0,batch_size):
        pcps.append(pearson_correlation_loss(anchor[i,:].view(ne),pos[i,:].view(ne)))
        pcns.append(pearson_correlation_loss(anchor[i,:].view(ne),neg[i,:].view(ne)))
    pcp = torch.stack(pcps,0)
    pcn = torch.stack(pcns,0)
    dist_hinge = torch.clamp(margin + pcn - pcp, min=0.0)
    loss = torch.mean(dist_hinge)
    return loss

def triplet_correlation_loss2(anchor,pos,neg,dummy_target,margin=1.0,eps=1e-6):
    # for now, debatch this computation, to batch properly need to figure out how to broadcast in torch
    batch_size = anchor.size(0)
    pcps,pcns = [],[]
    ne = anchor.nelement()/batch_size
    for i in range(0,batch_size):
        pcps.append(F.conv2d(anchor[i,:].unsqueeze(0),pos[i,:].unsqueeze(0),padding=3))
        pcns.append(F.conv2d(anchor[i,:].unsqueeze(0),neg[i,:].unsqueeze(0),padding=3))
    pcp = torch.stack(pcps,0)
    pcn = torch.stack(pcns,0)
    pos_tensor = Variable(torch.ones(pcp.size()))
    neg_tensor = Variable(torch.zeros(pcn.size()))
    pos_tensor = pos_tensor.cuda() if anchor.is_cuda else pos_tensor
    neg_tensor = neg_tensor.cuda() if anchor.is_cuda else neg_tensor
    ploss = F.binary_cross_entropy_with_logits(pcp,pos_tensor)
    nloss = F.binary_cross_entropy_with_logits(pcn,neg_tensor)
    Logger().set('loss_component.ploss',ploss.data.cpu()[0])
    Logger().set('loss_component.nloss',nloss.data.cpu()[0])    
    return ploss+nloss