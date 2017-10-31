from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

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
    KLD /= recon_x.size(0) * 784

    return BCE + KLD