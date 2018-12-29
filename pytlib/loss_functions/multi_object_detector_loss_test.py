import unittest
import torch
from loss_functions.multi_object_detector_loss import batch_box_intersection, batch_box_area, batch_box_IOU, assign_targets
from utils.test_utils import near_tensor_equality

class TestMultiObjectDetectorLoss(unittest.TestCase):

    def test_assign_targets(self):
        preds = torch.Tensor([[0,0,1,1],
                              [2,2,4,4]]).unsqueeze(0)
        targets = torch.Tensor([[3,3,4,4],
                                [0,0,1,1]]).unsqueeze(0)
        matches = assign_targets(preds,targets)
        expected_matches = ([[0, 0], [1, 0]], [[0, 0], [0, 1]])
        self.assertEqual(matches,expected_matches)

    def test_batch_box_IOU_edge_cases(self):
        t1 = torch.Tensor([[0,0,0,0]])
        t2 = torch.Tensor([[0,0,0,0]])
        ious = batch_box_IOU(t1,t2)
        expected_matrix = torch.Tensor([[0.]])
        self.assertTrue(near_tensor_equality(ious,expected_matrix))

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
        self.assertTrue(near_tensor_equality(ious,expected_matrix))

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
        self.assertTrue(near_tensor_equality(areas,expected_areas))

if __name__ == '__main__':
    unittest.main()