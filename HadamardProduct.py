import torch
import torch.nn as nn

class HadamardProduct(nn.Module):

    def __init__(self, shape):

        super(HadamardProduct, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(*shape))

    def forward(self, X):
        return X * self.weight