import unittest
import torch
from image.cam_math import image_to_cam,euler_to_mat,six_dof_vec_to_matrix
from utils.test_utils import near_tensor_equality
import numpy as np

class TestCamMath(unittest.TestCase):

    def test_image_to_cam_base(self):
        image = torch.Tensor([[[0,0],[2,2]],
                              [[1,1],[3,3]],
                              [[2,2],[4,4]]])
        depth = torch.Tensor([[1,1],
                              [5,5]])
        intrin = torch.Tensor([[1,0,0],
                               [0,1,0],
                               [0,0,1]])

        output_coords, output_colors = image_to_cam(image,depth,intrin)
        expected_output_coords = torch.Tensor([[0., 0., 5., 5.],
                                               [0., 1., 0., 5.],
                                               [1., 1., 5., 5.]])
        expected_output_colors = torch.Tensor([[0., 0., 2., 2.],
                                               [1., 1., 3., 3.],
                                               [2., 2., 4., 4.]])
        self.assertTrue(near_tensor_equality(expected_output_coords,output_coords))
        self.assertTrue(near_tensor_equality(expected_output_colors,output_colors))

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
        print mat
        expected_mat = torch.Tensor([[1,0,0],
                                     [0,-1,0],
                                     [0,0,-1]])
        self.assertTrue(near_tensor_equality(mat,expected_mat))

    def test_euler_to_mat_rot_x(self):
        # rotate around z by pi/2
        r = torch.Tensor([0,0,np.pi/2])
        mat = euler_to_mat(r)
        print mat
        expected_mat = torch.Tensor([[0,-1,0],
                                     [1,0,0],
                                     [0,0,1]])
        self.assertTrue(near_tensor_equality(mat,expected_mat))

    def test_six_dof_vec_to_matrix_base(self):
        vec6 = torch.Tensor([0,0,0,0,0,0])
        mat = six_dof_vec_to_matrix(vec6)
        print mat
        expected_mat = torch.Tensor([[1,0,0,0],
                                     [0,1,0,0],
                                     [0,0,1,0],
                                     [0,0,0,1]])
        self.assertTrue(near_tensor_equality(mat,expected_mat))

    def test_six_dof_vec_to_matrix_simple(self):
        vec6 = torch.Tensor([1,5,10,np.pi,0,0])
        mat = six_dof_vec_to_matrix(vec6)
        print mat
        expected_mat = torch.Tensor([[1,0,0,1],
                                     [0,-1,0,5],
                                     [0,0,-1,10],
                                     [0,0,0,1]])
        self.assertTrue(near_tensor_equality(mat,expected_mat))

if __name__ == '__main__':
    unittest.main()