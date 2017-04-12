import numpy as np
from PIL import Image
from image.box import Box

# a sample should contain both the data and the optional targets
# this is in pytorch tensor form
class Sample:
  def __init__(self,data,target=None):
    self.data = data
    self.target = target

  # these following functions only apply to CropObject samples, todo: move
  def data_to_pil_images(self):
    bd = 0
    nparray = sample.data.numpy()
    # loop over batch dimension
    splitarr = np.split(nparray,nparray.shape[bd],axis=bd)
    images = []
    for img in splitarr:
        images.append(Image.fromarray(np.uint8(img.squeeze())))
    return images

  def target_to_boxes(self):
    target_arr = sample.target.numpy()
    split_target_arr = np.split(target_arr,target_arr.shape[bd],axis=bd)
    boxes = []
    for targ in split_target_arr:
        boxes.append(Box.from_single_array(targ.squeeze()))
    return boxes

  def draw_data(self,max_cols=5):
    rows = int(math.ceil(float(len(images)) / max_cols))
    cols = max_cols if len(images) > max_cols else len(images)
    images = self.data_to_pil_images()
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