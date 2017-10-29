from PIL import Image
import numpy as np

# without considering batching we have the following main types of data
# pil array (compatible with hwc numpy)
# hwc numpy array
# chw numpy array
# chw torch tensor 


# Turns 3D PIL images to np_arrays of dims CHW
def PIL_to_cudnn_np(image):
    return np.transpose(np.asarray(image),axes=(2,0,1))

# Turns 3D CHW images (could have an extra batch dim of 1) into PIL images
def cudnn_np_to_PIL(np_array):
    hwc = np.transpose(np_array.squeeze(),axes=(1,2,0))
    return Image.fromarray(np.uint8(hwc))

def scale_np_img(image,input_range,output_range,output_type=float):
    assert len(input_range)==2 and len(output_range)==2
    scale = float(output_range[1] - output_range[0])/(input_range[1] - input_range[0])
    offset = output_range[0] - input_range[0]*scale;
    return (image*scale+offset).astype(output_type);
