import torch

def near_tensor_equality(m1,m2,tol=1e-4):
    return torch.all(torch.lt(torch.abs(torch.add(m1, -m2)), tol)) 
