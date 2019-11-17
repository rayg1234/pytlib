from __future__ import division
from builtins import range
from builtins import object
from past.utils import old_div
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

class GaussianAttentionGenerator(object):

    # here the params expect to be in the range from -1 to 1 to be kind of bounded within the image 
    @staticmethod
    def generate_filter_params(sampling_params,image_w,image_h,grid_size):
        _gx,_gy,log_sigma2,log_delta,loggamma = sampling_params.split(1,1)
        gx=old_div((image_w+1),2)*(_gx+1)
        gy=old_div((image_h+1),2)*(_gy+1)
        sigma2=torch.exp(log_sigma2)
        delta=old_div((max(image_w,image_h)-1),(grid_size-1))*torch.exp(log_delta)       
        gamma=torch.exp(loggamma)
        return gx,gy,sigma2,delta,gamma

    # generate two sets of filterbanks 
    # 1) batch x N x W (Fx)
    # 2) batch x N x H (Fy)
    @staticmethod
    def generate_filter_matrices(gx,gy,sigma2,delta,image_w,image_h,grid_size,minclamp=1e-8,maxclamp=1e8):
        N = grid_size
        grid_points = torch.arange(0,N).view((1,N,1))
        a = torch.arange(0,image_w).view((1,1,-1))
        b = torch.arange(0,image_h).view((1,1,-1))
        if gx.data.is_cuda:
            grid_points = grid_points.cuda()
            a = a.cuda()
            b = b.cuda()

        # gx is Bx1, grid is (1xNx1), so this is a broadcast op -> BxNx1
        mux = gx.view((-1,1,1)) + (grid_points - old_div(N,2) - 0.5) * delta.view((-1,1,1))
        muy = gy.view((-1,1,1)) + (grid_points - old_div(N,2) - 0.5) * delta.view((-1,1,1))

        s2 = sigma2.view((-1,1,1))
        fx = torch.exp(old_div(-(a-mux).pow(2),(2*s2)))
        fy = torch.exp(old_div(-(b-muy).pow(2),(2*s2)))
        # normalize
        fx = old_div(fx,torch.clamp(torch.sum(fx,2,keepdim=True),minclamp,maxclamp))
        fy = old_div(fy,torch.clamp(torch.sum(fy,2,keepdim=True),minclamp,maxclamp))
        return fx,fy        

# this differentialable gaussian attention module samples from 
# an input (c,h,w) and an 5 element params vector interpreted as 
# (_gx,_gy,log_sigma2,log_delta,loggamma)
# using the same equations as the DRAW paper
# The module outputs a (c,n,n) sized sample where n is the gridsize
# the sampling is identifical across channels dimension
class GaussianAttentionReader(nn.Module):
    def __init__(self):
        super(GaussianAttentionReader, self).__init__()

    def forward(self,x,sampling_params,grid_size):
        # assume nchw
        batch_size,chans,height,width = x.size()

        # 1) generate the filter params from the encoding_vector
        gx,gy,sigma2,delta,gamma = GaussianAttentionGenerator.generate_filter_params(sampling_params,width,height,grid_size)

        # 2) generate the filter matrices from the attention params
        fx,fy = GaussianAttentionGenerator.generate_filter_matrices(gx,gy,sigma2,delta,width,height,grid_size)

        # 3) apply fx and fy to get generate the glimpse, apply to each channel separately since bmm is only for 3D
        outputs = []
        for i in range(chans):
            o = gamma.view(-1,1,1)*torch.bmm(torch.bmm(fy,x[:,i,:,:]),torch.transpose(fx,1,2))
            outputs.append(o.view(batch_size,1,grid_size,grid_size))
        return torch.cat(outputs,dim=1)

# like the AttentionReader, this uses gaussian sampling to 
# write an input patch x and some geometric parameters 'encoding' and
# writes to a large canvas given by the canvas size
class GaussianAttentionWriter(nn.Module):
    def __init__(self):
        super(GaussianAttentionWriter, self).__init__()

    def forward(self,x,sampling_params,canvas_size):
        # assume nchw
        batch_size,chans,grid_size,_ = x.size()
        canvas_h,canvas_w = canvas_size

        # 1) generate the filter params from the encoding_vector
        gx,gy,sigma2,delta,gamma = GaussianAttentionGenerator.generate_filter_params(sampling_params,canvas_w,canvas_h,grid_size)

        # 2) generate the filter matrices from the attention params
        fx,fy = GaussianAttentionGenerator.generate_filter_matrices(gx,gy,sigma2,delta,canvas_w,canvas_h,grid_size)

        # 3) apply fx and fy to each channel and then write to a larger canvas
        outputs = []
        for i in range(chans):
            o = (old_div(1,gamma)).view(-1,1,1)*torch.bmm(torch.bmm(fy.transpose(1,2),x[:,i,:,:]),fx)
            outputs.append(o.view(batch_size,1,canvas_h,canvas_w))
        return torch.cat(outputs,dim=1)    