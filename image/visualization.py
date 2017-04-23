from PIL import Image
import numpy as np
from image.image_utils import cudnn_np_to_PIL
import matplotlib.pyplot as plt
import math

def visualize_pil_array(images,max_cols=5):
    rows = int(math.ceil(float(len(images)) / max_cols))
    cols = max_cols if len(images) > max_cols else len(images)
    fig,axes = plt.subplots(rows,cols,figsize=(15, 8))
    if rows==1:
        axes = [axes]
    lin_idx = 0

    for i in range(0,rows):
        for j in range(0,cols):
            lidx = np.ravel_multi_index((i,j),(rows,cols))
            if lidx<len(images):
                axes[i][j].imshow(images[lidx])
            else:
                axes[i][j].axis('off')

# tensor is the of the form BCHW
def tensor_to_pil_image_array(py_tensor):
    bd = 0
    nparray = py_tensor.numpy()
    # loop over batch dimension
    splitarr = np.split(nparray,nparray.shape[bd],axis=bd)
    images = []
    for img in splitarr:
        images.append(cudnn_np_to_PIL(img))
    return images
