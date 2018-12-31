import unittest
import torch
from loss_functions.multi_object_detector_loss import assign_targets

class TestMultiObjectDetectorLoss(unittest.TestCase):

    def test_assign_targets(self):
        preds = torch.Tensor([[0,0,1,1],
                              [2,2,4,4]]).unsqueeze(0)
        targets = torch.Tensor([[3,3,4,4],
                                [0,0,1,1]]).unsqueeze(0)
        pred_inds,target_inds = assign_targets(preds,targets)
        # print pred_inds, target_inds
        expected_pred_inds = [[0, 0], [0, 1]]
        expected_target_inds = [[0, 0], [1, 0]]
        self.assertEqual(pred_inds,expected_pred_inds)
        self.assertEqual(target_inds,expected_target_inds)

if __name__ == '__main__':
    unittest.main()