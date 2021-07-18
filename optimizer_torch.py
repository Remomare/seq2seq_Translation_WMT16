import torch

def Adam(parameters, lr):
    return torch.optim.Adam(parameters, lr)

def SGD(parameters, lr ):
    return torch.optim.SGD(parameters, lr)