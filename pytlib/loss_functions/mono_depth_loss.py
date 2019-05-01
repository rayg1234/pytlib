import torch
import torch.nn.functional as F

def mono_depth_loss(ego_motion_vectors,depth_maps,calib_frames):
    batch_size = calib_frames.shape[0]
    assert calib_frames.shape[1]==3, 'Currently only support 3-sequence frames!'
    # step 1) Use inverse cam_matrix and depth to convert
    # frame 1,2,3 into camera coordinates
    # calib_frames -> Bx3x3x3
    #
    # first apply inverse instrincs to get cam coords per pixel
    # use meshgrid here
    # X_cam{x,y} = K^-1 * X_image{i,j}
    # then to set the correct depth per pixel: X_cam{x,y}*=d/X_cam_z{x,y} 
    # this should get us BxKxNx3 where N is the number of pixels
    # and the last dimension represents 3D points (x,y,z)
    import ipdb;ipdb.set_trace()

    # step 2) Generate transformation matrix from ego_motion_vectors

    # step 3) Transform Frame1 -> Frame2 using transformation matrix
    # then apply camera matrix to get back to camera coords, this
    # would require sampling

    # step 4) repeat step 3) for Frame2 -> Frame3 
    # step 5) apply reconstruction loss

    # TODO placeholder function to get the NN to run
    return torch.sum(ego_motion_vectors)