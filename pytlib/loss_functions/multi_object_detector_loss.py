import torch

def normalize_boxes(original_image, boxes):
    # use input dimensions to normalize boxes between 0 and 1
    # assume boxes are BxNx4
    assert len(boxes.shape)==3 and boxes.shape[2]==4, 'boxes must a BxNx4 tensor'
    height = original_image.shape[2]
    width = original_image.shape[3]
    new_boxes = boxes.new_tensor(boxes.data)
    new_boxes[:,:,[0,2]] = boxes[:,:,[0,2]]/width
    new_boxes[:,:,[1,3]] = boxes[:,:,[1,3]]/height
    return new_boxes

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

    # next use scipy's hungarian to create the assignment
    pass

def multi_object_detector_loss(original_image, 
                               box_preds, 
                               class_preds, 
                               targets, 
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
    matches = assign_targets(normalized_box_targets, normalized_box_preds)

    # 3) only targets that have been assigned gets a box loss

    # 4) all targets get classification loss

    # 5) total loss = w0*class_loss + w1*box_loss
    total_loss = class_loss_weight*class_loss + box_loss_weight*box_loss
    return total_loss