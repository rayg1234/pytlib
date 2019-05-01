from utils.dict_utils import get_deep
from image.affine import Affine
from image.box import Box
from image.frame import Frame
import random
import numpy as np
import copy

# class for generating random affine transformations on boxes and images
class RandomPerturber:

    @staticmethod
    def generate_random_affine(center,edges,params):
        affine = Affine()

        # translate to center coordinates
        affine.append(Affine.translation(-center))

        translate = get_deep(params,'translation_range',[-0.1,0.1])
        tx = random.random()*(translate[1]-translate[0])+translate[0]
        ty = random.random()*(translate[1]-translate[0])+translate[0]
        translation_range = np.array([tx,ty])*(-edges)
        affine.append(Affine.translation(translation_range))

        scale = get_deep(params,'scaling_range',[0.9,1.4])
        scale = random.random()*(scale[1]-scale[0])+scale[0]
        scaling_range = np.array([scale,scale])
        affine.append(Affine.scaling(scaling_range))

        # translate back 
        affine.append(Affine.translation(center))
        return affine

    # scale and translation
    @staticmethod
    def perturb_crop_box(crop_box,params):
        rand_affine = RandomPerturber.generate_random_affine(crop_box.center(),crop_box.edges(),params)
        transformed_box = rand_affine.apply_to_box(crop_box)
        return transformed_box

    # scale, translation, (and rotation, TODO)
    # returns a new frame
    @staticmethod
    def perturb_frame(frame,params):
        dims = frame.image.get_hw()
        rand_affine = RandomPerturber.generate_random_affine(dims/2,dims,params)
        perturbed_frame = Frame(frame.image_path)
        perturbed_frame.image = rand_affine.apply_to_image(frame.image,dims)
        perturbed_frame.calib_mat = rand_affine.apply_to_matrix(frame.calib_mat)
        for i,obj in enumerate(frame.objects):
            # filter out completely out of bound objects
            perturbed_obj_box = rand_affine.apply_to_box(obj.box)
            perturbed_polygons = rand_affine.apply_to_polygons(obj.polygons)
            if Box.intersection(perturbed_obj_box,perturbed_frame.image.get_bounding_box()) is not None:
                obj_copy = copy.deepcopy(obj)
                obj_copy.box = perturbed_obj_box
                obj_copy.polygons = perturbed_polygons
                perturbed_frame.objects.append(obj_copy)
        return perturbed_frame

