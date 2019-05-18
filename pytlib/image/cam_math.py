import torch
import torch.nn.functional as F 
import numpy as np

def image_to_cam(image, depth, instrincs):
    """
    Args:
        image: CxHxW image
        depth: HxW depth map
        instrincs: 3x3 instrinc matrix
    Returns:
        a Nx3 tensor that represents the pixels in cam coordinates
        where N is the number of total pixels
        a Nx3 tensor that represents the RGB values of the previous pixels
        and second dimension represents 3D points (x,y,z,r,g,b)
    """
    # use meshgrid here
    # X_cam{x,y} = d * K^-1 * X_image{i,j}
    assert len(image.shape)==3, 'Image should CxHxW'
    assert len(depth.shape)==2, 'depth should HxW'
    assert image.shape[-1] == depth.shape[-1], 'Image W should match depth_map W'
    assert image.shape[-2] == depth.shape[-2], 'Image H should match depth_map H'
    ww,hh = torch.meshgrid(torch.arange(0,depth.shape[0],dtype=image.dtype,device=image.device),
                           torch.arange(0,depth.shape[1],dtype=image.dtype,device=image.device))
    flat_ww, flat_hh = torch.flatten(ww), torch.flatten(hh)
    # 3x(H*W) vectors
    points = torch.stack([flat_hh,flat_ww,torch.ones_like(flat_hh)])
    cam_coords = torch.matmul(torch.inverse(instrincs),points.float())
    # scale by the depth
    depth_mul = torch.flatten(depth).expand(3,depth.numel())
    cam_coords = torch.mul(depth_mul,cam_coords)
    return cam_coords

def cam_to_image(proj_mat, cam_coords, original_image):
    """
    projects a set of 3D coords to reconstruct a 2D image
    Args:
        proj_mat: the 4x3 projection matrix from 3D to 2D
        cam_coords: the homogenous coords in 3D, 4xN points
        original_image: the original image to transform and sample from, in CHW
    Returns:
        the projected 2D image, 2D mask of valid pixels
    """
    # First project the 3D cam coords to 2D grid (Nx2xWxH)
    # next use the differentiable grid sampling function
    # to sample from the original image
    assert len(original_image.shape)==3, 'Image should CxHxW'
    assert len(cam_coords.shape)==2 and cam_coords.shape[0]==4, 'cam coords must be 3xN'
    num_cam_points = cam_coords.shape[1]
    num_pixels = original_image.shape[1]*original_image.shape[2]
    assert num_cam_points==num_pixels, \
        'number of cam points {} must match original image points {}'.format(num_cam_points,num_pixels)
    projected2D = torch.matmul(proj_mat, cam_coords)
    # next normalize the 2D points
    epsilon = torch.ones_like(projected2D[2])*1e-8
    # coords from 2x(HxW) -> 2xHxW -> 2xHxW
    coords2D = torch.div(projected2D[0:2],projected2D[2]+epsilon)
    # first we need to normalize these coords to between [-1,1] in
    # the input spatial dimensions, any coords outside [-1,1] are handled
    # by padding.
    scaling = torch.ones_like(coords2D)
    scaling[0,:] = original_image.shape[2]-1
    scaling[1,:] = original_image.shape[1]-1
    coords2D = (coords2D/scaling)*2 - 1
    grid2D = coords2D.reshape(2,original_image.shape[1],original_image.shape[2])
    grid2D = grid2D.transpose(0,1)
    grid2D = grid2D.transpose(1,2)
    # finally apply bilinear affine sampling
    mask = torch.ones_like(original_image)
    output_mask = F.grid_sample(mask.unsqueeze(0), grid2D.unsqueeze(0), mode='nearest', padding_mode='zeros')
    output = F.grid_sample(original_image.unsqueeze(0), grid2D.unsqueeze(0), mode='bilinear', padding_mode='zeros')
    return output.squeeze(0), output_mask.squeeze(0)[0,:,:]

def euler_to_mat(vec3):
    """Converts euler angles to rotation matrix.
    Args:
      vec3: vector of dim 3 representing the x,y,z rotations
    Returns:
      4x4 Rotation matrix
    """
    assert vec3.numel()==3, 'input must be single vector of size 3'
    x,y,z = torch.chunk(vec3,3)
    z = torch.clamp(z, -np.pi, np.pi)
    y = torch.clamp(y, -np.pi, np.pi)
    x = torch.clamp(x, -np.pi, np.pi)

    cosz = torch.cos(z)
    sinz = torch.sin(z)
    zero = torch.zeros_like(cosz)
    one = torch.ones_like(cosz)
    rotz_1 = torch.cat([cosz, -sinz, zero])
    rotz_2 = torch.cat([sinz, cosz, zero])
    rotz_3 = torch.cat([zero, zero, one])
    zmat = torch.stack([rotz_1, rotz_2, rotz_3])

    cosy = torch.cos(y)
    siny = torch.sin(y)
    roty_1 = torch.cat([cosy, zero, siny])
    roty_2 = torch.cat([zero, one, zero])
    roty_3 = torch.cat([-siny, zero, cosy])
    ymat = torch.stack([roty_1, roty_2, roty_3])

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    rotx_1 = torch.cat([one, zero, zero])
    rotx_2 = torch.cat([zero, cosx, -sinx])
    rotx_3 = torch.cat([zero, sinx, cosx])
    xmat = torch.stack([rotx_1, rotx_2, rotx_3])

    return torch.matmul(torch.matmul(xmat, ymat), zmat)

def six_dof_vec_to_matrix(vector):
    """
    convert 6dof vector of [dx,dy,dz,rx,ry,rz] to 
    4x4 homogenous transformation matrix
    rx,ry,rz are rotations in euler angles
    Args:
        vector: vector of size 6: [dx,dy,dz,rx,ry,rz]
    Returns:
        4x4 homogenous transformation matrix
    """
    assert len(vector.shape)==1, 'vector must be 1'
    assert vector.numel()==6, 'vector must be of size 6'
    trans,rot = torch.chunk(vector,2)
    rot_mat = euler_to_mat(rot)
    four_by_three = torch.cat([rot_mat,trans.reshape([3,1])],dim=1)
    last_row = torch.zeros_like(four_by_three[0,:])
    last_row[3]=1
    final_mat = torch.cat([four_by_three,last_row.unsqueeze(0)])
    return final_mat
