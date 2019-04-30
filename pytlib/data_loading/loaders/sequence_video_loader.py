from data_loading.loaders.loader import Loader
from data_loading.sample import Sample
from interface import implements
from image.random_perturber import RandomPerturber
from image.affine_transforms import resize_image_center_crop,apply_affine_to_frame
from image.ptimage import PTImage,Ordering,ValueClass
import numpy as np
import random
import torch

class SequenceVideoSample(implements(Sample)):
    def __init__(self,data,target):
        self.data = data
        self.target = target
        self.output = None

    def visualize(self,parameters={}):
        # visualizes a sequence
        pass

    def set_output(self,output):
        self.output = output

    def get_data(self):
        return self.data

    def get_target(self):
        return self.target

class SequenceVideoLoader(implements(Loader)):
    def __init__(self,source,crop_size,num_frames=3):
        self.source = source
        self.crop_size = crop_size
        self.num_frames = num_frames

    def next(self):
        # randomly pick 3 frames in a row
        num_frames_in_src = len(self.source)
        print("Number of frames in src {}".format(num_frames_in_src))

        # 1) choose the first frame from 0 -> N-2
        frames = []
        first_frame = random.randint(0,num_frames_in_src - 2)
        for i in range(0,self.num_frames):
            frames.append(self.source[first_frame+i])

        # 2) generate a random perturbation and perturb all the frames
        perturb_params = {'translation_range':[-0.1,0.1],
                          'scaling_range':[0.9,1.1]}
        perturbed_frames = []
        # TODO: also apply perturbation to instrincs
        for f in frames:
            perturbed_frame = RandomPerturber.perturb_frame(f,perturb_params)
            crop_affine = resize_image_center_crop(perturbed_frame.image,self.crop_size)
            output_size = [self.crop_size[1],self.crop_size[0]]
            perturbed_frame = apply_affine_to_frame(perturbed_frame,crop_affine,output_size)
            perturbed_frames.append(perturbed_frame)
            # perturbed_frame.visualize(title='chw_image',display=True)

        # 3) prepare tensors
        # -make a tensor with a stack of 3 frame 
        # -add the calibration to the target
        input_tensors = []
        for f in perturbed_frames:
            img = f.image.to_order_and_class(Ordering.CHW,ValueClass.FLOAT01)
            input_tensors.append(torch.Tensor(img.get_data().astype(float)))


        # the input is now 3xCxWxH
        sample = SequenceVideoSample([torch.stack(input_tensors,dim=0)],
                                     [torch.Tensor(0)])
        return sample





