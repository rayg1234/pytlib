import torch
import torch.nn.functional as F
from image.cam_math import image_to_cam

def mono_depth_loss(original_images, ego_motion_vectors,depth_maps,calib_frames):
    batch_size = calib_frames.shape[0]
    assert calib_frames.shape[1]==3, 'Currently only support 3-sequence frames!'
    assert len(original_images.shape)==5, 'Image shape should be BxKxCxHxW'
    assert len(ego_motion_vectors.shape)==3, 'Ego vector shape should be Bx(K-1)x6'
    assert ego_motion_vectors.shape[2]==6, 'Ego vector shape should be Bx(K-1)x6'
    assert len(depth_maps.shape)==4, 'Depth map should BxKxHxW'
    # step 1) Use inverse cam_matrix and depth to convert
    # frame 1,2,3 into camera coordinates 
    import ipdb;ipdb.set_trace()

    # step 2) Generate transformation matrix from ego_motion_vectors

    # step 3) Transform Frame1 -> Frame2 using transformation matrix
    # then apply camera matrix to get back to camera coords, this
    # would require sampling

    # step 4) repeat step 3) for Frame2 -> Frame3 
    # step 5) apply reconstruction loss

    # TODO placeholder function to get the NN to run
    return torch.sum(ego_motion_vectors)