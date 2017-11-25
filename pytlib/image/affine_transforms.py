from image.affine import Affine
import numpy as np
    
# take crop from the crop_box and then resize to the output_resolution
def crop_image_resize(ptimage,crop_box,output_resolution):
    affine = Affine()
    scale = min(float(output_resolution[0])/crop_box.edges()[0],float(output_resolution[1])/crop_box.edges()[1])
    crop_offset = crop_box.xy_min() + 0.5*(np.array(crop_box.edges())-np.array(output_resolution)/scale)
    affine.append(Affine.translation(-crop_offset))
    affine.append(Affine.scaling([scale,scale]))
    # print affine.transform
    return affine

# resize the image to the output_resolution, only in 1 dimension
# to preserve aspect ratio, then center crop in the second dimension
def resize_image_center_crop(ptimage,output_resolution):
    affine = Affine()
    # calculate scale and crop offsets
    wh = ptimage.get_wh()
    scale = min(float(output_resolution[0])/wh[0],float(output_resolution[1])/wh[1])
    offset = 0.5*(scale*np.array(wh)-np.array(output_resolution))
    affine.append(Affine.scaling([scale,scale]))
    affine.append(Affine.translation(-offset))
    return affine
