import torch
import scipy.optimize
import torch.nn.functional as F
from loss_functions.box_loss import box_loss
from utils.logger import Logger

def normalize_boxes(original_image, boxes):
    # use input dimensions to normalize boxes between 0 and 1
    # assume boxes are BxNx4
    assert len(boxes.shape)==3 and boxes.shape[2]==4, 'boxes must a BxNx4 tensor'
    height = original_image.shape[2]
    width = original_image.shape[3]
    boxes[:,:,[0,2]] = boxes[:,:,[0,2]]/width
    boxes[:,:,[1,3]] = boxes[:,:,[1,3]]/height
    return boxes

def preprocess_targets(targets):
    # 1) prune dummy targets, remove all rows with [-1]*5
    # 2) separate box targets
    # 3) separate class targets, no need for 1-hot encoding, apply NLL loss directly
    # dummy targets have the first column set to -1
    condition = targets[:,:,0]>-1
    filtered_targets = targets[condition].reshape(targets.shape[0],-1,targets.shape[2])
    box_targets = filtered_targets[:,:,1:]
    class_targets = filtered_targets[:,:,0]
    return box_targets, class_targets

def batch_box_intersection(t1,t2):
    assert len(t1.shape)==2 and len(t2.shape)==2 and \
        t1.shape[1]==4 and t2.shape[1]==4, 'boxes must be Nx4 tensors'
    x_min1,y_min1,x_max1,y_max1 = torch.chunk(t1,4,dim=1)
    x_min2,y_min2,x_max2,y_max2 = torch.chunk(t2,4,dim=1)
    y_mins = torch.max(y_min1,y_min2.transpose(0,1))
    y_maxs = torch.min(y_max1,y_max2.transpose(0,1))
    intersect_heights = torch.max(torch.zeros_like(y_maxs), y_maxs - y_mins)
    x_mins = torch.max(x_min1,x_min2.transpose(0,1))
    x_maxs = torch.min(x_max1,x_max2.transpose(0,1))
    intersect_widths = torch.max(torch.zeros_like(x_maxs), x_maxs - x_mins)
    result = intersect_heights * intersect_widths 
    return result

def batch_box_area(boxes):
    assert len(boxes.shape)==2 and boxes.shape[1]==4, 'boxes must be Nx4 tensors'
    x_min,y_min,x_max,y_max = torch.chunk(boxes,4,dim=1)
    area = ((x_max - x_min)*(y_max - y_min))
    return area

def batch_box_IOU(t1,t2):
    intersections = batch_box_intersection(t1,t2)
    width,height = intersections.shape
    areas1 = batch_box_area(t1)
    areas2 = batch_box_area(t2)
    areas1_expanded = areas1.expand(-1,height)
    areas2_expanded = areas2.transpose(0,1).expand(width,-1)
    unions = (areas1_expanded + areas2_expanded - intersections)
    iou = torch.where(
        torch.eq(intersections, 0.0),
        torch.zeros_like(intersections), torch.div(intersections, unions))        
    return iou

def assign_targets(box_preds, box_targets):
    # first construct cost function using IOU
    # for each box in box_targets, create IOU cost against a row in box_preds
    assert len(box_preds.shape)==3 and len(box_targets.shape)==3, 'boxes must be BxNx4'
    # explicit loop over batches
    pred_indices,target_indices = [[],[]],[[],[]]
    for i in range(0,box_targets.shape[0]):
        iou_cost = 1 - batch_box_IOU(box_preds[i,:,:],box_targets[i,:,:])
        # next use scipy's hungarian to create the assignment
        row_inds, col_inds = scipy.optimize.linear_sum_assignment(iou_cost.detach().cpu().numpy())
        pred_indices[0].extend([i]*len(col_inds))
        pred_indices[1].extend(col_inds)
        target_indices[0].extend([i]*len(row_inds))
        target_indices[1].extend(row_inds)        
    return pred_indices,target_indices

def multi_object_detector_loss(original_image, 
                               box_preds, 
                               class_preds, 
                               targets, 
                               pos_to_neg_class_weight_ratio=3.0,
                               class_loss_weight=1.0,
                               box_loss_weight=1.0):
    batch_size = original_image.shape[0]
    # 1) preprocess targets
    # box_targets: BxNx4
    # class_targets: BxNx1
    box_targets, class_targets = preprocess_targets(targets)

    # 2) normalize box targets and inputs coordinate system 0-1
    normalized_box_targets = normalize_boxes(original_image, box_targets.view(batch_size,-1,4))
    box_preds_flatten_hw = box_preds.flatten(start_dim=2)
    normalized_box_preds = normalize_boxes(original_image, box_preds_flatten_hw.view(batch_size,-1,4))

    # 2) globally assign targets against predictions
    pred_indices,target_indices = assign_targets(normalized_box_targets, normalized_box_preds)

    # 3) only targets that have been assigned gets a box loss
    total_box_loss = box_loss(normalized_box_preds[pred_indices], normalized_box_targets[target_indices])

    # 4) all targets get classification loss
    num_classes = class_preds.shape[1]
    class_preds_flatten_hw = class_preds.flatten(start_dim=2).view(batch_size,-1,num_classes)
    softmax_preds = F.log_softmax(class_preds_flatten_hw,dim=2)
    positive_class_loss = F.nll_loss(softmax_preds[pred_indices],class_targets.flatten().long())
    mask = torch.ones_like(softmax_preds,dtype=torch.uint8)
    mask[pred_indices] = 0
    neg_preds = torch.masked_select(softmax_preds,mask).reshape(-1,2)
    neg_targets = neg_preds.new_ones(neg_preds.shape[0],dtype=torch.long)*(num_classes-1)
    negative_class_loss = F.nll_loss(neg_preds,neg_targets)
    total_class_loss = pos_to_neg_class_weight_ratio/(1.+pos_to_neg_class_weight_ratio)*positive_class_loss \
        + 1./(1+pos_to_neg_class_weight_ratio)*negative_class_loss

    # 5) total loss = w0*class_loss + w1*box_loss
    Logger().set('loss_component.positive_class_loss',positive_class_loss.mean().item())
    Logger().set('loss_component.negative_class_loss',negative_class_loss.mean().item())
    Logger().set('loss_component.total_box_loss',total_box_loss.mean().item())
    Logger().set('loss_component.total_class_loss',total_class_loss.mean().item())
    total_loss = class_loss_weight*total_class_loss + box_loss_weight*total_box_loss
    return total_loss