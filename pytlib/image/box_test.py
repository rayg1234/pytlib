import unittest
import torch
from image.box import Box

class TestBox(unittest.TestCase):

	def test_tensor_to_boxes(self):
		box0 = [0,0,1,1]
		box1 = [2,2,3,3]
		t1 = torch.Tensor([box0,box1])
		boxes = Box.tensor_to_boxes(t1)
		tboxes0 = boxes[0].to_single_array()
		tboxes1 = boxes[1].to_single_array()
		self.assertTrue(all(box0[i]==tboxes0[i] for i in range(len(tboxes0))))
		self.assertTrue(all(box1[i]==tboxes1[i] for i in range(len(tboxes1))))