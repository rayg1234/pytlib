from PIL import Image
import numpy as np

# Turns 3D PIL images to np_arrays of dims CHW
def PIL_to_cudnn_np(image):
    return np.transpose(np.asarray(image),axes=(2,0,1))

# Turns 3D CHW images (could have an extra batch dim of 1) into PIL images
def cudnn_np_to_PIL(np_array):
    hwc = np.transpose(np_array.squeeze(),axes=(1,2,0))
    return Image.fromarray(np.uint8(hwc))