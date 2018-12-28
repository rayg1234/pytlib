import unittest
import torch
from loss_functions.multi_object_detector_loss import batch_box_intersection, batch_box_area, batch_box_IOU

class TestMultiObjectDetectorLoss(unittest.TestCase):

    def near_equality(self,m1,m2):
        return torch.all(torch.lt(torch.abs(torch.add(m1, -m2)), 1e-4))

    def test_batch_box_IOU_edge_cases(self):
        t1 = torch.Tensor([[0,0,0,0]])
        t2 = torch.Tensor([[0,0,0,0]])
        ious = batch_box_IOU(t1,t2)
        expected_matrix = torch.Tensor([[0.]])
        self.assertTrue(self.near_equality(ious,expected_matrix))

    def test_batch_box_IOU(self):
        t1 = torch.Tensor([[0,0,1,1],
                           [2,2,8,8],
                           [-1,-1,0,0]])
        t2 = torch.Tensor([[0,0,1,1],
                           [3,3,9,9]])
        ious = batch_box_IOU(t1,t2)
        expected_matrix = torch.Tensor([[1.0000, 0.0000],
                                        [0.0000, 0.5319],
                                        [0.0000, 0.0000]])
        self.assertTrue(self.near_equality(ious,expected_matrix))

    def test_batch_box_intersection_edge_cases(self):
        # no intersection
        t1 = torch.Tensor([[0,0,1,1]])
        t2 = torch.Tensor([[3,4,9,10],
                           [3,3,9,9]])
        intersection_matrix = batch_box_intersection(t1,t2)
        expected_matrix = torch.Tensor([[ 0.],
                                        [ 0.]])
        self.assertTrue(torch.all(torch.eq(intersection_matrix,expected_matrix)))

        # negative intersection
        t1 = torch.Tensor([[0,0,-1,-1]])
        t2 = torch.Tensor([[-3,-4,9,10],
                           [-3,-3,9,9]])
        intersection_matrix = batch_box_intersection(t1,t2)
        expected_matrix = torch.Tensor([[ 0.],
                                        [ 0.]])
        self.assertTrue(torch.all(torch.eq(intersection_matrix,expected_matrix)))

    def test_batch_box_intersection(self):
        t1 = torch.Tensor([[0,0,1,1],
                           [2,2,8,8]])
        t2 = torch.Tensor([[0,0,1,1],
                           [3,3,9,9]])
        intersection_matrix = batch_box_intersection(t1,t2)
        expected_matrix = torch.Tensor([[ 1.,  0.],
                                        [ 0., 25.]])
        self.assertTrue(torch.all(torch.eq(intersection_matrix,expected_matrix)))

    def test_batch_box_area(self):
        boxes = torch.Tensor([[0,0,1,1],
                              [2,2,8,8],
                              [0,0,0,0]])
        areas = batch_box_area(boxes)
        expected_areas = torch.tensor([[1.,36.,0.]]).transpose(0,1)
        self.assertTrue(self.near_equality(areas,expected_areas))

if __name__ == '__main__':
    unittest.main()