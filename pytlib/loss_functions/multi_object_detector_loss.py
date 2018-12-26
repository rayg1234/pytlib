import torch

def multi_object_detector_loss(predictions,target):
    import ipdb;ipdb.set_trace()
    # 1) normalize targets and inputs coordinate system 0-1
    # 2) globally assign targets against predictions
    # 3) only targets that have been assigned gets a box loss
    # 4) all non-assigned ones should regressed to bg class
    return 0