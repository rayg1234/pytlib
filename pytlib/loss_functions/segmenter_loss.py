import torch
import torch.nn as nn
import torch.nn.functional as F

def recurrent_segmenter_loss(output_array,target_masks):
    BCE = F.binary_cross_entropy(output_array[-1], target_masks)
    return BCE