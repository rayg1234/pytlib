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

    def forward(self,output_array,gauss_attn_params_all,target_masks):
        attn_reader = GaussianAttentionReader()
        attn_writer = GaussianAttentionWriter()
        batch_size,_,height,width  = output_array[-1].size()

        # gauss_attn_params = torch.chunk(gauss_attn_params_all,batch_size,dim=0)[0]
        BCE_total = F.binary_cross_entropy(output_array[-1], target_masks)
        # for i,params in enumerate(gauss_attn_params_all[:-1]):
        #     partial_mask = attn_reader.forward(target_masks,params,self.grid_size)
        #     full_mask = attn_writer.forward(partial_mask,params,(height,width))
        #     full_mask = full_mask.detach()

        #     full_mask_batched = torch.chunk(full_mask,batch_size,dim=0)
        #     ImageVisualizer().set_image(PTImage.from_2d_wh_torch(full_mask_batched[0].squeeze().data),'tGlimpse {}'.format(i))     
        #     BCE_glimpse = F.binary_cross_entropy(output_array[i], full_mask)
        #     BCE_total = torch.add(BCE_total, BCE_glimpse)
        return BCE_total