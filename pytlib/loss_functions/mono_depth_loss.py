import torch
import torch.nn.functional as F

def mono_depth_loss(ego_motion_vectors,depth_maps,target_frames):
	# step 1) Use calibration and depth to convert
	# frame 1,2,3 into world coordinates

	# step 2) Generate transformation matrix from ego_motion_vectors

	# step 3) Transform Frame1 -> Frame2 using transformation matrix
	# then apply inverse transformation to get back to camera coords
	# apply reconstruction loss

	# step 4) repeat step 3) for Frame2 -> Frame3 

	# TODO placeholder function to get the NN to run
    return torch.sum(ego_vectors)