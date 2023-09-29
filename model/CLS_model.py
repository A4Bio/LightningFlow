import torch
import torch.nn as nn
import torch.nn.functional as F

class CLSModel(nn.Module):
    def __init__(self):
        super(CLSModel, self).__init__()
        self.layer = nn.Linear(784, 10)

    def forward(self, x):
        y = self.layer(x)
        output = F.log_softmax(y, dim=1)
        return {'output':output}