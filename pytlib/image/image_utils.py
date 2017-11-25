import numpy as np

def scale_np_img(image,input_range,output_range,output_type=float):
    assert len(input_range)==2 and len(output_range)==2
    scale = float(output_range[1] - output_range[0])/(input_range[1] - input_range[0])
    offset = output_range[0] - input_range[0]*scale;
    return (image*scale+offset).astype(output_type);
