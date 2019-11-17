from __future__ import division
from past.utils import old_div
from image.affine import Affine
from image.frame import Frame
from image.box import Box
import numpy as np
import copy
    
# take crop from the crop_box and then resize to the output_resolution
def crop_image_resize(ptimage,crop_box,output_resolution):
    affine = Affine()
    scale = min(float(output_resolution[0])/crop_box.edges()[0],float(output_resolution[1])/crop_box.edges()[1])
    crop_offset = crop_box.xy_min() + 0.5*(np.array(crop_box.edges())-old_div(np.array(output_resolution),scale))
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
    scale = max(float(output_resolution[0])/wh[0],float(output_resolution[1])/wh[1])
    offset = 0.5*(scale*np.array(wh)-np.array(output_resolution))
    affine.append(Affine.scaling([scale,scale]))
    affine.append(Affine.translation(-offset))
    return affine

def apply_affine_to_frame(frame,affine,output_size):
    perturbed_frame = Frame(frame.image_path)
    perturbed_frame.image = affine.apply_to_image(frame.image,output_size)
    for i,obj in enumerate(frame.objects):
        # filter out completely out of bound objects
        perturbed_obj_box = affine.apply_to_box(obj.box)
        perturbed_polygons = affine.apply_to_polygons(obj.polygons)
        if Box.intersection(perturbed_obj_box,perturbed_frame.image.get_bounding_box()) is not None:
            obj_copy = copy.deepcopy(obj)
            obj_copy.box = perturbed_obj_box
            obj_copy.polygons = perturbed_polygons
            perturbed_frame.objects.append(obj_copy)
    return perturbed_frame         