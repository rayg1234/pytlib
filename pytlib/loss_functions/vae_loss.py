from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.logger import Logger

# adapted from https://github.com/pytorch/examples/blob/master/vae/main.py
def KLD_gaussian(mu,logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def vae_loss(reconstruction,mu,logvar,targets):
    BCE = F.binary_cross_entropy(reconstruction, targets)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = KLD_gaussian(mu,logvar)
    # Normalise by same number of elements as in reconstruction
    # assume bchw format
    total_elements = reduce((lambda x, y: x * y), reconstruction.size())
    KLD /= total_elements
    Logger().set('loss_component.variance_mean',logvar.exp().data.mean())
    Logger().set('loss_component.mu_mean',mu.data.mean())
    Logger().set('loss_component.reconstruction_mean',reconstruction.data.mean())
    Logger().set('loss_component.reconstruction_std',reconstruction.data.std())
    Logger().set('loss_component.KLD',KLD.data.cpu()[0])
    Logger().set('loss_component.BCE',BCE.data.cpu()[0])
    return BCE + KLD

# for DRAW model https://arxiv.org/pdf/1502.04623.pdf
def sequence_vae_loss(recs,mus,logvars,target):
    assert len(recs)>0 and len(mus)==len(logvars), "sequence_vae_loss: dimensions don't match"
    # import ipdb;ipdb.set_trace()
    BCE = F.binary_cross_entropy(recs[-1], target)
    KLD = KLD_gaussian(mus[0],logvars[0])
    for t in range(1,len(mus)):
        KLD = torch.add(KLD, KLD_gaussian(mus[t],logvars[t]))

    total_elements = recs[-1].nelement()
    total_elements *= len(mus)
    KLD /= total_elements

    Logger().set('loss_component.reconstruction_mean',recs[-1].data.mean())
    Logger().set('loss_component.reconstruction_std',recs[-1].data.std())    
    Logger().set('loss_component.KLD',KLD.data.cpu()[0])
    Logger().set('loss_component.BCE',BCE.data.cpu()[0])    
    return BCE + KLD

