import torch


def one_hot(indices, depth, dtype=torch.float):
    y_onehot = torch.zeros(size=(indices.size(0), depth), dtype=dtype, device=indices.device)
    y_onehot.scatter_(1, indices.unsqueeze(1), 1)

    return y_onehot
