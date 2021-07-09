import torch
import torch.nn as nn


class DropoutHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, target_dim)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        y_hat = self.linear(x)
        return y_hat

class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        return y_hat
