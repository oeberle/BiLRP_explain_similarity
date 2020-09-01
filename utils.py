import os
import cv2
import torch
import numpy as np
import copy
import torch.nn as nn
from torch.nn import Module


def set_up_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def load_image(imgfile):
    img = np.array(cv2.imread(imgfile))[...,::-1]/255.0
    return img

def proc_image(imgfile, mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225] ):
    mean = torch.Tensor(mean).reshape(1,-1,1,1)
    std  = torch.Tensor(std).reshape(1,-1,1,1)
    img = load_image(imgfile)
    X = (torch.FloatTensor(img[np.newaxis].transpose([0,3,1,2])*1) - mean) / std
    return X

def pool(X, stride):
    K = [torch.nn.functional.avg_pool2d(torch.from_numpy(o).unsqueeze(0).unsqueeze(1),kernel_size=stride, stride=stride, padding=0).squeeze().numpy() for o in X]
    return K


def newlayer(layer,g):
    layer = copy.deepcopy(layer)
    try: layer.weight = nn.Parameter(g(layer.weight))
    except AttributeError: pass

    try: layer.bias   = nn.Parameter(g(layer.bias))
    except AttributeError: pass

    return layer



class Flatten(Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.
    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).
    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).
    Examples::
        >>> m = nn.Sequential(
        >>>     nn.Conv2d(1, 32, 5, 1, 1),
        >>>     nn.Flatten()
        >>> )
    """
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim).unsqueeze(2).unsqueeze(3)

    
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x
   