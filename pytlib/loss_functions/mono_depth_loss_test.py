import unittest
import torch
from loss_functions.mono_depth_loss import process_single_batch
from utils.test_utils import near_tensor_equality

class TestMonoDepthLoss(unittest.TestCase):

    # if depth maps are 1s and ego transform is identity
    # then we should expect to get the same image back
    def test_process_single_batch(self):
        image_0 = torch.Tensor([[[0,0],[2,2]],
                                [[1,1],[3,3]],
                                [[2,2],[4,4]]])
        image_1 = torch.Tensor([[[0,0],[2,2]],
                                [[1,1],[3,3]],
                                [[2,2],[4,4]]])
        ego_motion_vectors = [torch.Tensor([0,0,0,0,0,0])] 
        depth_map_0 = torch.Tensor([[1,1],[1,1]])
        depth_map_1 = torch.Tensor([[1,1],[1,1]])
        calib_0 = torch.Tensor([[1,0,0],[0,1,0],[0,0,1]])
        calib_1 = torch.Tensor([[1,0,0],[0,1,0],[0,0,1]])
        
        image_input = torch.stack([image_0,image_1],dim=0)
        depth_input = torch.stack([depth_map_0,depth_map_1],dim=0)
        calib_input = torch.stack([calib_0,calib_1],dim=0)
        loss,out_images = process_single_batch(image_input,ego_motion_vectors,depth_input,calib_input)
        self.assertTrue(near_tensor_equality(image_1,out_images[0]))
        print loss,out_images



if __name__ == '__main__':
    unittest.main()