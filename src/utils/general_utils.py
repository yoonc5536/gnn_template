import torch
import numpy as np


def dn(tensor: torch.Tensor) -> np.array:
    return tensor.detach().cpu().numpy()
