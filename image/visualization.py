import numpy as np
from image.image_utils import cudnn_np_to_PIL
import matplotlib.pyplot as plt
import math
from PIL import Image


# image needs to be hwc ordering
def visualize_np_image(image):
    plt.imshow(image, interpolation='nearest')
    plt.show()

def visualize_pil_array(images,max_cols=5,title='title',block=True):
    rows = int(math.ceil(float(len(images)) / max_cols))
    cols = max_cols if len(images) > max_cols else len(images)
    fig,axes = plt.subplots(rows,cols,figsize=(15, 8))
    fig.canvas.set_window_title(title)
    print rows, cols
    if rows==1 and cols==1:
        axes = [[axes]]
    elif rows==1 and cols>1:
        axes = [axes]
    lin_idx = 0

    for i in range(0,rows):
        for j in range(0,cols):
            lidx = np.ravel_multi_index((i,j),(rows,cols))
            if lidx<len(images):
                axes[i][j].imshow(images[lidx])
            else:
                axes[i][j].axis('off')
    plt.show(block=block)

# tensor is the of the form BHWC
def tensor_to_pil_image_array(py_tensor):
    batch_dimension = 0
    nparray = py_tensor.numpy()
    # loop o1ver batch dimension
    splitarr = np.split(nparray,nparray.shape[batch_dimension],axis=batch_dimension)
    images = []
    for img in splitarr:
        images.append(cudnn_np_to_PIL(img))
    return images
