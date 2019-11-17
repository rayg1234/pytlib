from __future__ import division
from builtins import range
from past.utils import old_div
import torch
import torch.nn.functional as F
import numpy as np

def rescale_boxes(boxes, scale, offset=[0,0]):
    assert boxes.shape[1]==4, 'dim[1] of boxes must the box dimensions, instead got {}'.format(boxes.shape)
    height,width = scale
    new_boxes = boxes.clone()
    new_boxes[:,[1,3]] = boxes[:,[1,3]]*height + offset[0]
    new_boxes[:,[0,2]] = boxes[:,[0,2]]*width + offset[1]
    return new_boxes

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

def batch_nms(boxes,thresh=0.5):
    assert len(boxes.shape)==2 and boxes.shape[1]==4, 'boxes must be Nx4 tensors'
    #compute IOU of every pair of boxes
    # NxN grid of ious
    ious = batch_box_IOU(boxes,boxes)
    #loop over every row in upper triagle, find row with IOU greater than T, mark that index
    indices = []
    for i in range(0,boxes.shape[0]):
        idxs = (ious[i,i+1:]>=thresh).nonzero() + i+1
        indices.append(idxs)
    mask = torch.ones_like(boxes[:,0],dtype=torch.bool)
    if indices:
        all_unique_indices = torch.unique(torch.cat(indices))
        mask[all_unique_indices] = 0
    return boxes[mask], mask

def euc_distance_cost(boxes1,boxes2):
    assert len(boxes1.shape)==2 and boxes1.shape[1]==4, 'boxes must be Nx4 tensors'
    assert len(boxes2.shape)==2 and boxes2.shape[1]==4, 'boxes must be Nx4 tensors'
    # center squared cost: ((xmax1+xmin1)/2 - (xmax2+xmin2)/2))^2
    x_min1,y_min1,x_max1,y_max1 = torch.chunk(boxes1,4,dim=1)
    x_min2,y_min2,x_max2,y_max2 = torch.chunk(boxes2,4,dim=1)
    cx1 = old_div((x_max1+x_min1),2)
    cx2 = old_div((x_max2+x_min2),2)
    cy1 = old_div((y_max1+y_min1),2)
    cy2 = old_div((y_max2+y_min2),2)
    xx1,xx2 = torch.meshgrid(cx1.squeeze(),cx2.squeeze())
    yy1,yy2 = torch.meshgrid(cy1.squeeze(),cy2.squeeze())
    distx = (xx2-xx1)*(xx2-xx1)
    disty = (yy2-yy1)*(yy2-yy1)
    return distx+disty

def generate_region_meshgrid(num_regions,region_size,region_offsets):
    hh,ww = torch.meshgrid(torch.arange(0,num_regions[0]),torch.arange(0,num_regions[1]))
    hh = (hh*region_size[0]+region_offsets[0]).cuda().float()
    ww = (ww*region_size[1]+region_offsets[1]).cuda().float()
    return hh,ww
