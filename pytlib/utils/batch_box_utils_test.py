import unittest
import torch
from utils.batch_box_utils import batch_box_intersection, batch_box_area, batch_box_IOU, rescale_boxes, euc_distance_cost, batch_nms
from utils.test_utils import near_tensor_equality

class TestBatchBoxUtils(unittest.TestCase):

    def test_rescale_boxes_no_offset(self):
        boxes = torch.Tensor([[0,0,1,1],
                              [2,2,4,4]])
        scale = [1,2]
        scaled_boxes = rescale_boxes(boxes, scale)
        expected_boxes = torch.Tensor([[0.,0.,2.,1.],
                                       [4.,2.,8.,4.]])
        # print expected_boxes
        self.assertTrue(near_tensor_equality(scaled_boxes,expected_boxes))

    def test_rescale_boxes_with_offset(self):
        boxes = torch.Tensor([[0,0,1,1],
                              [2,2,4,4]])
        scale = [1,2]
        offset = [40,30]
        scaled_boxes = rescale_boxes(boxes, scale, offset)
        expected_boxes = torch.Tensor([[30.,40.,32.,41.],
                                       [34.,42.,38.,44.]])
        # print scaled_boxes
        self.assertTrue(near_tensor_equality(scaled_boxes,expected_boxes))

    def test_batch_box_IOU_edge_cases(self):
        t1 = torch.Tensor([[0,0,0,0]])
        t2 = torch.Tensor([[0,0,0,0]])
        ious = batch_box_IOU(t1,t2)
        expected_matrix = torch.Tensor([[0.]])
        self.assertTrue(near_tensor_equality(ious,expected_matrix))

    def test_euc_distance_cost(self):
        t1 = torch.Tensor([[0,0,1,1],
                           [2,2,8,8],
                           [-1,-1,0,0]])
        t2 = torch.Tensor([[0,0,1,1],
                           [3,3,9,9]])
        distmat = euc_distance_cost(t1,t2)
        expected_matrix = torch.Tensor([[ 0.0000, 60.5000],
                                        [40.5000,  2.0000],
                                        [ 2.0000, 84.5000]])
        self.assertTrue(near_tensor_equality(distmat,expected_matrix))

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

    def test_batch_nms_a(self):
        boxes = torch.Tensor([[0,0,1,1],
                              [0,0,1,1],
                              [0,0,5,5],
                              [1,1,6,6],
                              [3,3,10,10],
                              [3,3,6,6],
                              [0,0,1,1]])

        expected_result = torch.Tensor([[0,0,1,1],
                                     [0,0,5,5],
                                     [1,1,6,6],
                                     [3,3,10,10],
                                     [3,3,6,6]])
        # thresh=0.5        
        results, _ = batch_nms(boxes)
        self.assertTrue(torch.all(torch.eq(results,expected_result)))

    def test_batch_nms_b(self):
        boxes = torch.Tensor([[0,0,1,1],
                              [0,0,1,1],
                              [0,0,5,5],
                              [1,1,6,6],
                              [3,3,10,10],
                              [3,3,6,6],
                              [0,0,1,1]])

        expected_result = torch.Tensor([[0,0,1,1],
                                        [0,0,5,5]])
        results, _ = batch_nms(boxes,thresh=0.1)
        self.assertTrue(torch.all(torch.eq(results,expected_result)))

if __name__ == '__main__':
    unittest.main()