import torch
import torch.nn as nn


class NNLoss(nn.Module):
    def __init__(self, dim, reduction='mean'):
        super(NNLoss, self).__init__()
        self.dim = dim
        self.reduction = reduction

    def forward(self, inputs, k):
        top_values, _ = torch.topk(inputs, k=k, dim=self.dim)
        loss = -torch.log(top_values).sum(dim=1) / k

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
