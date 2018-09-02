import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.gaussian_attention_sampler import GaussianAttentionReader,GaussianAttentionWriter
from visualization.image_visualizer import ImageVisualizer
from image.ptimage import PTImage

class RecurrentSegmeterLoss(nn.Module):

    def __init__(self,grid_size):
        super(RecurrentSegmeterLoss, self).__init__()
        self.grid_size = grid_size

    def forward(self,output_array,gauss_attn_params_all,partial_maskes,target_masks):
        attn_reader = GaussianAttentionReader()
        attn_writer = GaussianAttentionWriter()
        batch_size,_,height,width  = output_array[-1].size()

        # gauss_attn_params = torch.chunk(gauss_attn_params_all,batch_size,dim=0)[0]
        BCE_total = F.binary_cross_entropy(output_array[-1], target_masks)

        for i,params in enumerate(gauss_attn_params_all):
            param_clone = params.clone().detach()
            # import ipdb;ipdb.set_trace()
            param_clone[:,4] = 0
            partial_mask_target = attn_reader.forward(target_masks,param_clone,self.grid_size).round()
            # full_mask = attn_writer.forward(partial_mask,param_clone,(height,width))
            # full_mask = full_mask.detach()
            # import ipdb;ipdb.set_trace()

            partial_mask_target_batched = torch.chunk(partial_mask_target,batch_size,dim=0)
            ImageVisualizer().set_image(PTImage.from_2d_wh_torch(partial_mask_target_batched[0].squeeze().data),'tGlimpse {}'.format(i))     
            BCE_glimpse = F.binary_cross_entropy(F.sigmoid(partial_maskes[i]), partial_mask_target)
            BCE_total = torch.add(BCE_total, BCE_glimpse)
        return BCE_total