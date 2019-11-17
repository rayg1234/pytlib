from __future__ import division
from builtins import range
from past.utils import old_div
import torch
import torch.nn.functional as F
from image.cam_math import image_to_cam,cam_to_image,six_dof_vec_to_matrix
from visualization.image_visualizer import ImageVisualizer
from image.ptimage import PTImage
from utils.logger import Logger
from loss_functions.ssim_loss import ssim
import numpy as np

#TODO: this is for initial dev only, make the entire loss function batched
def process_single_batch(original_images,ego_motion_vectors,disp_maps,calib_frames,batch_number=0,mask_loss_factor=0.1):
    cam_coords = []
    num_frames = calib_frames.shape[0]
    Logger().set('loss_component.disp_maps_mean',disp_maps.data.mean().item())
    Logger().set('loss_component.disp_maps_min',disp_maps.data.min().item())
    Logger().set('loss_component.disp_maps_max',disp_maps.data.max().item())
    Logger().set('loss_component.ego_motion_vectors[0]',
        np.array2string(ego_motion_vectors[0].detach().cpu().numpy()))
    # step 1) Use inverse cam_matrix and depth to convert
    # frame 1,2,3 into camera coordinates    
    for i in range(0,num_frames):
        cam_coords.append(image_to_cam(original_images[i],disp_maps[i],calib_frames[i]))
    transforms = []
    # step 2) Generate transformation matrix from ego_motion_vectors
    for i in range(0,num_frames-1):
        # fake_ego_motion_vec = torch.zeros_like(ego_motion_vectors[i])
        transforms.append(six_dof_vec_to_matrix(ego_motion_vectors[i]))

    # step 3) Transform Frame i (cam_coords) -> Frame i+1(cam_coords) 
    # Then construct a new 2D image using new projection matrix
    total_re_loss = torch.zeros([],dtype=original_images.dtype,device=original_images.device)
    total_ssim_loss = torch.zeros([],dtype=original_images.dtype,device=original_images.device)
    total_mask_loss =  torch.zeros([],dtype=original_images.dtype,device=original_images.device)
    out_images = []
    for i in range(0,num_frames-1):
        # augment cam coords with row of 1's to 4D vecs
        ones_row = torch.ones_like(cam_coords[i])[0,:].unsqueeze(0)
        augmented_vecs = torch.cat((cam_coords[i],ones_row),dim=0)
        cur_frame_coords = torch.matmul(transforms[i],augmented_vecs)
        intrin_filler_right = torch.zeros(3,dtype=original_images.dtype,device=original_images.device).unsqueeze(1)
        intrin_filler_bottom = torch.zeros(4,dtype=original_images.dtype,device=original_images.device).unsqueeze(0)
        intrin_filler_bottom[0,3] = 1
        hom_calib = torch.cat((calib_frames[i],intrin_filler_right),dim=1)
        hom_calib = torch.cat((hom_calib,intrin_filler_bottom),dim=0)
        warped_image, mask = cam_to_image(hom_calib,cur_frame_coords,original_images[i])
        out_images.append(warped_image)
        # compare warped_image to next real image
        # don't use 0 pixels for loss
        ptimage = PTImage.from_cwh_torch(warped_image)
        ptmask = PTImage.from_2d_wh_torch(mask)
        orig_image = PTImage.from_cwh_torch(original_images[i])
        # ImageVisualizer().set_image(orig_image,'original_images {}'.format(i))
        ImageVisualizer().set_image(ptimage,'warped_image {}-{}'.format(batch_number,i))
        ImageVisualizer().set_image(ptmask,'mask {}-{}'.format(batch_number,i))
        Logger().set('loss_component.mask_mean.{}-{}'.format(batch_number,i),mask.mean().data.item())    

        masked_warp_image = warped_image.unsqueeze(0) * mask
        masked_gt_image = original_images[i+1].unsqueeze(0) * mask
        re_loss = F.smooth_l1_loss(masked_warp_image,masked_gt_image,reduction='none')
        # add loss to prevent mask from going to 0
        # total_mask_loss += mask_loss_factor*F.smooth_l1_loss(mask, torch.ones_like(mask))
        total_re_loss += re_loss.mean()
        total_ssim_loss += old_div((1-ssim(masked_warp_image,masked_gt_image)),2)

    Logger().set('loss_component.mask_loss.{}'.format(batch_number),total_mask_loss.data.item())    
    Logger().set('loss_component.batch_re_loss.{}'.format(batch_number),total_re_loss.data.item())
    Logger().set('loss_component.batch_ssim_loss.{}'.format(batch_number),total_ssim_loss.data.item())
    return total_re_loss+total_ssim_loss+total_mask_loss,out_images

def mono_depth_loss(original_images,ego_motion_vectors,disp_maps,calib_frames):
    batch_size = calib_frames.shape[0]
    num_frames = calib_frames.shape[1]
    assert num_frames==3, 'Currently only support 3-sequence frames!'
    assert len(original_images.shape)==5, 'Image shape should be BxKxCxHxW'
    assert len(ego_motion_vectors.shape)==3, 'Ego vector shape should be Bx(K-1)x6'
    assert ego_motion_vectors.shape[2]==6, 'Ego vector shape should be Bx(K-1)x6'
    assert len(disp_maps.shape)==4, 'Depth map should BxKxHxW'


    # loop over all batches manually for now
    # TODO: change all helpers here to handle batches
    loss = torch.zeros([],dtype=original_images.dtype,device=original_images.device)
    for b in range(0,batch_size):
        single_batch_loss = process_single_batch(original_images[b,:],
                                                 ego_motion_vectors[b,:],
                                                 disp_maps[b,:],
                                                 calib_frames[b,:],
                                                 b)
        loss+=single_batch_loss
    return loss