from utils.dict_utils import get_deep
from image.affine import Affine
from image.box import Box
import random
import numpy as np

# Given an image and a tight crop of an object
# generate a new 'loose' crop around the object with the new target
# coordinates of the crop, optionally apply some perturbations to it

# img 100x100, box is 10x10 @ 2,2, this is a translation of 2x2 and output domain of 10x10
# crop_params:
# crop_range: [1.1,1.3] 110-130 percent of original bounding box
# scale_range: [0.9,1.1] 90-110 percent of original size
# translation_range: [-0.2,0.2] translation applied after scale
# rotation: tbd

# returns both the new image and the new crop box
def generate_new_crop(image,
                      crop_box,
                      crop_params,
                      random_seed=1234):
    random.seed(random_seed)
    affine = Affine.from_box(crop_box)
    output_dims = get_deep(crop_params,'output_dims')
    assert output_dims is not None,'output dims not specified'

    # affine transform order
    # 1) randomly generate crop upscaling using crop_scale_range (assert > 1)
    # this is a center scale
    crop_scale_range = get_deep(crop_params,'crop_scale_range',default=[1.1,1.3])
    crop_scale = random.random()*(crop_scale_range[1]-crop_scale_range[0]) + crop_scale_range[0]
    shift = -crop_box.edges()*crop_scale/2
    affine.append(Affine.translation(shift))

    # 2) add translation perturbation
    translate = get_deep(crop_params,'translation_range',[0.0,0.0])
    tx = random.random()*(translate[1]-translate[0])+translate[0]
    ty = random.random()*(translate[1]-translate[0])+translate[0]
    translation_range = np.array([tx,ty])*(crop_box.edges())
    affine.append(Affine.translation(translation_range))

    # 3) add scaling perturbation
    scale = get_deep(crop_params,'scaling_range',[1.0,1.0])
    sx = random.random()*(scale[1]-scale[0])+scale[0]
    sy = random.random()*(scale[1]-scale[0])+scale[0]
    scaling_range = np.array([sx,sy])*(crop_box.edges())
    affine.append(Affine.scaling(scaling_range))

    # 4) apply rescale to output dims

    # now apply the transformation to both the image and coords using the output dims as boundary
    transformed_box = affine.apply_to_coords(crop_box.augmented_matrix())
    print 'transformed_box: '+ str(transformed_box)

    transformed_box = Box.from_augmented_matrix(transformed_box).bound(Box.from_edges(output_dims))
    print 'transformed_box: '+ str(transformed_box)

    # now apply to image

    return transformed_box
