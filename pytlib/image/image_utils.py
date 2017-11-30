import numpy as np
import torch
from image.box import Box

def scale_np_img(image,input_range,output_range,output_type=float):
    assert len(input_range)==2 and len(output_range)==2
    scale = float(output_range[1] - output_range[0])/(input_range[1] - input_range[0])
    offset = output_range[0] - input_range[0]*scale;
    return (image*scale+offset).astype(output_type);

def box_to_tensor(box,frame_size):
    # normalize box coord to between 0 and 1
    box_array = box.scale(1/np.array(frame_size,dtype=float)).to_single_array().astype(float)
    return torch.Tensor(box_array)

def tensor_to_box(tensor,frame_size):
    assert tensor.size()==torch.Size([4]), 'tensor must of size 4 got {}'.format(tensor.size())
    return Box.from_single_array(tensor.numpy()).scale(frame_size)