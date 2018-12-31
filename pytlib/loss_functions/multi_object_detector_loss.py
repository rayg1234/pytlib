import torch
import scipy.optimize
import torch.nn.functional as F
from loss_functions.box_loss import box_loss
from utils.logger import Logger
from utils.batch_box_utils import rescale_boxes, euc_distance_cost, generate_region_meshgrid
import numpy as np

def preprocess_targets_and_preds(targets, box_preds, class_preds, original_image):
    # 1) prepare dummy masks
    # 2) separate box targets
    # 3) separate class targets, no need for 1-hot encoding, apply NLL loss directly
    batch_size = original_image.shape[0]

    dummy_mask = targets[:,:,0]>-1
    dummy_masks = [x.squeeze() for x in torch.chunk(dummy_mask,targets.shape[0])]
    box_targets = targets[:,:,1:]
    class_targets = targets[:,:,0]

    # scale predictions by visual regions
    # use a meshgrid, for example we have 2x2 visual regions with 32pix strides
    # then the grid looks like
    # yy = [[16,16],[48,48]]
    # xx = [[16,48],[16,48]]
    # we use this meshgrid to create an offset for the original predictions
    # so that a 0,0 prediction is exactly at the center of that region
    # and -1,1 are the edges of this region
    # use original_image.shape/cnn feature map shape to approximate the vision region size
    # TODO, check this works for non-multiple sizes of cnn reduction factor
    region_size = np.array(original_image.shape[2:])/np.array(box_preds.shape[3:])
    hh,ww = generate_region_meshgrid(box_preds.shape[3:], region_size, region_size/2)
    rescaled_box_preds = rescale_boxes(box_preds, region_size, [hh,ww])
    rescaled_box_preds_flat = rescaled_box_preds.flatten(start_dim=2).transpose(1,2)

    box_targets_flat = box_targets.reshape(batch_size,-1,4)

    num_classes = class_preds.shape[1]
    class_preds_flatten_hw = class_preds.flatten(start_dim=2).transpose(1,2)
    logsoftmax_preds = F.log_softmax(class_preds_flatten_hw,dim=2)
    return rescaled_box_preds_flat, box_targets_flat, logsoftmax_preds, class_targets, dummy_masks

def assign_targets(box_preds, box_targets, dummy_target_masks=None):
    if dummy_target_masks is None:
        dummy_target_masks = [torch.ones_like(box_targets[0,:,0],dtype=torch.uint8)]*box_targets.shape[0]

    # first construct cost function using IOU
    # for each box in box_targets, create IOU cost against a row in box_preds
    assert len(box_preds.shape)==3 and len(box_targets.shape)==3, 'boxes must be BxNx4'
    assert len(dummy_target_masks)==box_preds.shape[0], 'dummy_masks must match batch size'
    # explicit loop over batches
    pred_indices,target_indices = [[],[]],[[],[]]
    for i in range(0,box_targets.shape[0]):
        # TODO, dont use IOU cost, use distance based cost
        cost = euc_distance_cost(box_preds[i,:,:],box_targets[i,dummy_target_masks[i],:])
        # cost = 1 - batch_box_IOU(box_preds[i,:,:],box_targets[i,dummy_target_masks[i],:])
        original_target_indices = (dummy_target_masks[i]!=0).nonzero().squeeze(1).cpu().numpy()
        # next use scipy's hungarian to create the assignment
        row_inds, col_inds = scipy.optimize.linear_sum_assignment(cost.detach().cpu().numpy())
        pred_indices[0].extend([i]*len(row_inds)) # batch index
        pred_indices[1].extend(row_inds)
        target_indices[0].extend([i]*len(col_inds)) # batch index
        original_col_indices = [original_target_indices[k] for k in col_inds]
        target_indices[1].extend(original_col_indices) #original target indices        
    return pred_indices,target_indices

def multi_object_detector_loss(original_image, 
                               box_preds, 
                               class_preds, 
                               targets, 
                               pos_to_neg_class_weight_ratio=3,
                               class_loss_weight=1.0,
                               box_loss_weight=1.0):
    # 1) preprocess targets
    # p_box_targets: BxNx4
    # p_box_preds: BxNx4
    # p_class_preds: BxNxC
    # p_class_targets: BxNx1
    # dummy_masks: list, batch items of N
    p_box_preds, p_box_targets, p_class_preds, p_class_targets, dummy_target_masks = \
        preprocess_targets_and_preds(targets, box_preds, class_preds, original_image)

    # 2) globally assign targets against predictions
    pred_indices, target_indices = assign_targets(p_box_preds, p_box_targets, dummy_target_masks)

    targets_exist = p_box_targets[target_indices].numel()
    # 3) only targets that have been assigned gets a box loss
    total_box_loss = 0
    if targets_exist:
        total_box_loss = box_loss(p_box_preds[pred_indices], p_box_targets[target_indices])
        Logger().set('loss_component.total_box_loss',total_box_loss.mean().item())

    # 4) all targets get classification loss
    # TODO, move this into a function with a unit test
    positive_class_loss = 0
    Logger().set('loss_component.positve_class_targets_size',p_class_targets[target_indices].shape[0])
    if targets_exist:
        positive_class_loss += F.nll_loss(p_class_preds[pred_indices], p_class_targets[target_indices].long())
        Logger().set('loss_component.positive_class_loss',positive_class_loss.mean().item())
    
    mask = torch.ones_like(p_class_preds,dtype=torch.uint8)
    mask[pred_indices] = 0
    neg_preds = torch.masked_select(p_class_preds,mask).reshape(-1,2)
    neg_targets = neg_preds.new_ones(neg_preds.shape[0],dtype=torch.long)*(p_class_preds.shape[2]-1)
    Logger().set('loss_component.negative_class_targets_size',neg_targets.flatten().shape[0])
    negative_class_loss = F.nll_loss(neg_preds,neg_targets,reduction='sum')
    Logger().set('loss_component.negative_class_loss',negative_class_loss.mean().item())
    total_class_loss = pos_to_neg_class_weight_ratio/(1.+pos_to_neg_class_weight_ratio)*positive_class_loss \
        + 1/(1.+pos_to_neg_class_weight_ratio)*negative_class_loss       
    # 5) total loss = w0*class_loss + w1*box_loss
    Logger().set('loss_component.total_class_loss',total_class_loss.mean().item())
    total_loss = class_loss_weight*total_class_loss + box_loss_weight*total_box_loss
    
    total_positive_targets = torch.sum(F.softmax(class_preds.flatten(start_dim=2).transpose(1,2),dim=2)[:,:,0]>0.5)
    total_negative_targets = torch.sum(F.softmax(class_preds.flatten(start_dim=2).transpose(1,2),dim=2)[:,:,1]>0.5)
    Logger().set('loss_component.total_negative_targets',total_negative_targets.item())
    Logger().set('loss_component.total_positive_targets',total_positive_targets.item())
    return total_loss