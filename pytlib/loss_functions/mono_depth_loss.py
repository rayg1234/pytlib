import torch
import torch.nn.functional as F
from image.cam_math import image_to_cam,cam_to_image,six_dof_vec_to_matrix
from visualization.image_visualizer import ImageVisualizer
from image.ptimage import PTImage

#TODO: this is for initial dev only, make the entire loss function batched
def process_single_batch(original_images,ego_motion_vectors,depth_maps,calib_frames):
    cam_coords = []
    num_frames = calib_frames.shape[0]

    # step 1) Use inverse cam_matrix and depth to convert
    # frame 1,2,3 into camera coordinates    
    for i in range(0,num_frames):
        cam_coords.append(image_to_cam(original_images[i],depth_maps[i],calib_frames[i]))

    transforms = []
    # step 2) Generate transformation matrix from ego_motion_vectors
    for i in range(0,num_frames-1):
        # fake_ego_motion_vec = torch.zeros_like(ego_motion_vectors[i])
        transforms.append(six_dof_vec_to_matrix(ego_motion_vectors[i]))

    # step 3) Transform Frame i (cam_coords) -> Frame i+1(cam_coords) 
    # Then construct a new 2D image using new projection matrix
    total_loss = torch.zeros([],dtype=original_images.dtype,device=original_images.device)
    warped_images = []
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
        warped_images.append(warped_image)
        # compare warped_image to next real image
        # don't use 0 pixels for loss
        ptimage = PTImage.from_cwh_torch(warped_image)
        ptmask = PTImage.from_2d_wh_torch(mask)
        orig_image = PTImage.from_cwh_torch(original_images[i])
        ImageVisualizer().set_image(orig_image,'original_images {}'.format(i))
        ImageVisualizer().set_image(ptimage,'warped_image {}'.format(i))
        ImageVisualizer().set_image(ptmask,'mask {}'.format(i))
        loss = F.smooth_l1_loss(warped_image,original_images[i+1],reduction='none')
        loss = loss * mask
        total_loss+=loss.mean()
    return total_loss, warped_images

def mono_depth_loss(original_images,ego_motion_vectors,depth_maps,calib_frames):
    batch_size = calib_frames.shape[0]
    num_frames = calib_frames.shape[1]
    assert num_frames==3, 'Currently only support 3-sequence frames!'
    assert len(original_images.shape)==5, 'Image shape should be BxKxCxHxW'
    assert len(ego_motion_vectors.shape)==3, 'Ego vector shape should be Bx(K-1)x6'
    assert ego_motion_vectors.shape[2]==6, 'Ego vector shape should be Bx(K-1)x6'
    assert len(depth_maps.shape)==4, 'Depth map should BxKxHxW'


    # loop over all batches manually for now
    # TODO: change all helpers here to handle batches
    loss = torch.zeros([],dtype=original_images.dtype,device=original_images.device)
    for b in range(0,batch_size):
        single_batch_loss, _ = process_single_batch(original_images[b,:],
                                                    ego_motion_vectors[b,:],
                                                    depth_maps[b,:],
                                                    calib_frames[b,:])
        loss+=single_batch_loss
    return loss