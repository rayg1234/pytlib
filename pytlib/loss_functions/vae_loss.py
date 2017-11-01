from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.logger import Logger

# adapted from https://github.com/pytorch/examples/blob/master/vae/main.py
def vae_loss(outputs,targets):
    recon_x = outputs[0]
    mu = outputs[1]
    logvar = outputs[2]
    BCE = F.binary_cross_entropy(recon_x, targets)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    # assume bchw format
    total_elements = reduce((lambda x, y: x * y), recon_x.size())
    KLD /= total_elements
    Logger().set('loss_component.variance_mean',logvar.exp().data.mean())
    Logger().set('loss_component.mu_mean',mu.data.mean())
    Logger().set('loss_component.reconstruction_mean',recon_x.data.mean())
    Logger().set('loss_component.reconstruction_std',recon_x.data.std())
    Logger().set('loss_component.KLD',KLD.data.cpu()[0])
    Logger().set('loss_component.BCE',BCE.data.cpu()[0])
    return BCE + KLD