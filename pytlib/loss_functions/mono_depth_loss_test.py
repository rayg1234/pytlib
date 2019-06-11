import unittest
import torch
from loss_functions.mono_depth_loss import process_single_batch
from utils.test_utils import near_tensor_equality
import numpy as np

class TestMonoDepthLoss(unittest.TestCase):

    # if disp_map are 0s and ego transform is identity
    # then we should expect to get the same image back
    def test_process_single_batch_no_motion(self):
        image_0 = torch.Tensor([[[0,0],[2,2]],
                                [[1,1],[3,3]],
                                [[2,2],[4,4]]])
        ego_motion_vectors = [torch.Tensor([0,0,0,0,0,0])] 
        disp_map = torch.Tensor([[0,0],[0,0]])
        calib_0 = torch.Tensor([[50,0,20],
                                [0,60,10],
                                [0,0,1]])
        image_input = torch.stack([image_0,image_0],dim=0)
        disp_input = torch.stack([disp_map,disp_map],dim=0)
        calib_input = torch.stack([calib_0,calib_0],dim=0)
        loss,out_images = process_single_batch(image_input,ego_motion_vectors,disp_input,calib_input)
        self.assertTrue(near_tensor_equality(image_input,out_images[0]))
        self.assertTrue(near_tensor_equality(loss,torch.Tensor([0])))

    # when depth is large, then small motions should be irrelevant
    # and therefore reprojection error is small
    def test_process_single_batch_motion_large_depth(self):
        image_0 = torch.Tensor([[[0,0],[2,2]],
                                [[1,1],[3,3]],
                                [[2,2],[4,4]]])
        ego_motion_vectors = [torch.Tensor([10,10,10,0,0,0])] 
        disp_map_0 = torch.Tensor([[1e-8,1e-8],[1e-8,1e-8]])
        calib_0 = torch.Tensor([[50,0,20],
                                [0,60,10],
                                [0,0,1]])
        image_input = torch.stack([image_0,image_0],dim=0)
        disp_input = torch.stack([disp_map_0,disp_map_0],dim=0)
        calib_input = torch.stack([calib_0,calib_0],dim=0)
        loss,out_images = process_single_batch(image_input,ego_motion_vectors,disp_input,calib_input)
        self.assertTrue(near_tensor_equality(image_0,out_images[0],tol=1e-2))
        self.assertTrue(near_tensor_equality(loss,torch.Tensor([0])))

if __name__ == '__main__':
    unittest.main()