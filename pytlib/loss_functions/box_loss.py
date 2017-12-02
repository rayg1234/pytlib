import torch
import torch.nn.functional as F

def box_loss(output_box,target_box):
    return F.smooth_l1_loss(output_box,target_box)