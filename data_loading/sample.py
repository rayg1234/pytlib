import numpy as np
from PIL import Image
from image.box import Box
import math
from image.visualization import tensor_to_pil_image_array


# a sample should contain both the data and the optional targets
# this is in pytorch tensor form
class Sample:
  def __init__(self,data,target=None):
      self.data = data
      self.target = target

  # these following functions only apply to CropObject samples, todo: move
  # to helper  class
  def data_to_pil_images(self):
      return tensor_to_pil_image_array(self.data)

  def target_to_boxes(self):
    target_arr = self.target.numpy()
    split_target_arr = np.split(target_arr,target_arr.shape[bd],axis=bd)
    boxes = []
    for targ in split_target_arr:
        boxes.append(Box.from_single_array(targ.squeeze()))
    return boxes

