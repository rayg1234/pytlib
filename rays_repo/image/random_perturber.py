from utils.dict_utils import get_deep
from image.affine import Affine
from image.box import Box
import random
import numpy as np

# class for generating random affine transformations on boxes and images
class RandomPerturber:

    # scale and translation
    @staticmethod
    def perturb_crop_box(params,crop_box):
        affine = Affine()

        # translate to center coordinates
        affine.append(Affine.translation(-crop_box.center()))

        translate = get_deep(params,'translation_range',[-0.1,0.1])
        tx = random.random()*(translate[1]-translate[0])+translate[0]
        ty = random.random()*(translate[1]-translate[0])+translate[0]
        translation_range = np.array([tx,ty])*(-crop_box.edges())
        affine.append(Affine.translation(translation_range))

        scale = get_deep(params,'scaling_range',[0.8,1.2])
        sx = random.random()*(scale[1]-scale[0])+scale[0]
        sy = random.random()*(scale[1]-scale[0])+scale[0]
        scaling_range = np.array([sx,sy])
        affine.append(Affine.scaling(scaling_range))

        # translate back 
        affine.append(Affine.translation(crop_box.center()))

        transformed_box = affine.apply_to_box(crop_box)
        return transformed_box

    # scale, translation, and rotation
    @staticmethod
    def perturb_image(params,image,boxes=[]):
        pass