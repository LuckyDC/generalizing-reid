import torch
import torch.nn as nn


class NCALoss(nn.Module):
    def __init__(self, dim, reduction='mean'):
        super(NCALoss, self).__init__()

        self.dim = dim
        self.reduction = reduction

    def forward(self, inputs, k):
        values, indices = torch.topk(inputs, k=k, dim=self.dim)

        top_sum = torch.sum(values, dim=self.dim)
        loss = - torch.log(top_sum)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
