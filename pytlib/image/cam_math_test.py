from __future__ import division
from __future__ import print_function
from past.utils import old_div
import unittest
import torch
from image.cam_math import image_to_cam,euler_to_mat,six_dof_vec_to_matrix,cam_to_image
from utils.test_utils import near_tensor_equality
import numpy as np

class TestCamMath(unittest.TestCase):

    def test_image_to_cam_base(self):
        image = torch.Tensor([[[0,0],[2,2]],
                              [[1,1],[3,3]],
                              [[2,2],[4,4]]])
        disp = torch.Tensor([[1,1],
                              [5,5]])
        intrin = torch.Tensor([[1,0,0],
                               [0,1,0],
                               [0,0,1]])

        output_coords = image_to_cam(image,disp,intrin)
        expected_output_coords = torch.Tensor([[0., 1., 0., 0.2],
                                               [0., 0., 0.2, 0.2],
                                               [1., 1., 0.2, 0.2]])
        self.assertTrue(near_tensor_equality(expected_output_coords,output_coords))

    def test_euler_to_mat_base(self):
        # test base case: no rotation
        r = torch.Tensor([0,0,0])
        mat = euler_to_mat(r)
        expected_mat = torch.Tensor([[1,0,0],
                                     [0,1,0],
                                     [0,0,1]])
        self.assertTrue(near_tensor_equality(mat,expected_mat))

    def test_euler_to_mat_rot_x(self):
        # rotate around x by pi
        r = torch.Tensor([np.pi,0,0])
        mat = euler_to_mat(r)
        print(mat)
        expected_mat = torch.Tensor([[1,0,0],
                                     [0,-1,0],
                                     [0,0,-1]])
        self.assertTrue(near_tensor_equality(mat,expected_mat))

    def test_euler_to_mat_rot_x(self):
        # rotate around z by pi/2
        r = torch.Tensor([0,0,old_div(np.pi,2)])
        mat = euler_to_mat(r)
        print(mat)
        expected_mat = torch.Tensor([[0,-1,0],
                                     [1,0,0],
                                     [0,0,1]])
        self.assertTrue(near_tensor_equality(mat,expected_mat))

    def test_six_dof_vec_to_matrix_base(self):
        vec6 = torch.Tensor([0,0,0,0,0,0])
        mat = six_dof_vec_to_matrix(vec6)
        print(mat)
        expected_mat = torch.Tensor([[1,0,0,0],
                                     [0,1,0,0],
                                     [0,0,1,0],
                                     [0,0,0,1]])
        self.assertTrue(near_tensor_equality(mat,expected_mat))

    def test_six_dof_vec_to_matrix_simple(self):
        vec6 = torch.Tensor([1,5,10,np.pi,0,0])
        mat = six_dof_vec_to_matrix(vec6)
        print(mat)
        expected_mat = torch.Tensor([[1,0,0,1],
                                     [0,-1,0,5],
                                     [0,0,-1,10],
                                     [0,0,0,1]])
        self.assertTrue(near_tensor_equality(mat,expected_mat))

    def test_cam_to_image(self):
        image = torch.Tensor([[[0,0],[2,2]],
                              [[1,1],[3,3]],
                              [[2,2],[4,4]]])   
        proj_mat = torch.Tensor([[1,0,0,0],
                                 [0,1,0,0],
                                 [0,0,1,0],
                                 [0,0,0,1]])
        cam_coords = torch.Tensor([[0,1,0,1],
                                   [0,0,1,1],
                                   [1,1,1,1],
                                   [1,1,1,1]])            
        output_image,_ = cam_to_image(proj_mat, cam_coords, image)
        self.assertTrue(near_tensor_equality(output_image,image))

    def test_cam_to_image_non_square(self):
        image = torch.Tensor([[[0,0,0],
                               [0,5,0]],
                              [[1,6,1],
                               [1,1,1]],
                              [[2,7,2],
                               [2,2,2]]])   
        proj_mat = torch.Tensor([[1,0,0,0],
                                 [0,1,0,0],
                                 [0,0,1,0],
                                 [0,0,0,1]])                                         
        cam_coords = torch.Tensor([[0,1,2,0,1,2],
                                   [0,0,0,1,1,1],
                                   [1,1,1,1,1,1],
                                   [1,1,1,1,1,1]])             
        output_image,_ = cam_to_image(proj_mat, cam_coords, image)
        self.assertTrue(near_tensor_equality(output_image,image))

    def test_cam_to_image_mask(self):
        image = torch.Tensor([[[0,0,0],
                               [0,5,0]],
                              [[1,6,1],
                               [1,1,1]],
                              [[2,7,2],
                               [2,2,2]]])   
        proj_mat = torch.Tensor([[1,0,0,0],
                                 [0,1,0,0],
                                 [0,0,1,0],
                                 [0,0,0,1]])                                         
        cam_coords = torch.Tensor([[0,1,2,0,1,2],
                                   [1,1,1,2,2,2],
                                   [1,1,1,1,1,1],
                                   [1,1,1,1,1,1]])             
        output_image,mask = cam_to_image(proj_mat, cam_coords, image)
        expected_mask = torch.Tensor([[1., 1., 1.],
                                      [0., 0., 0.]])
        expected_out = torch.Tensor([[[0., 5., 0.],
                                      [0., 0., 0.]],
                                     [[1., 1., 1.],
                                      [0., 0., 0.]],
                                     [[2., 2., 2.],
                                      [0., 0., 0.]]])
        self.assertTrue(near_tensor_equality(output_image,expected_out))
        self.assertTrue(near_tensor_equality(mask,expected_mask))


if __name__ == '__main__':
    unittest.main()