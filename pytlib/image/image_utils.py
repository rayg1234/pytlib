import numpy as np

def generate_response_map_from_boxes(map_size,boxes=[]):
    assert len(map_size)==2, 'can only generate 2D response maps'
    binarized_target_map = np.zeros(map_size)
    for box in boxes:
        box2d = box.to_2x2_np_array();
        bounds_x = np.clip(np.round(box2d[0]).astype(int),0,map_size[0])
        bounds_y = np.clip(np.round(box2d[1]).astype(int),0,map_size[1])
        binarized_target_map[bounds_y[0]:bounds_y[1],bounds_x[0]:bounds_x[1]]=1
    return binarized_target_map
