import torch

def init_parameter(layer, w, b):
    layer.weight.data.copy_(torch.from_numpy(w))
    layer.bias.data.copy_(torch.from_numpy(b))