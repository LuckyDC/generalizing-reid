import torch.distributed as dist


def reduce_tensor(tensor):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()

    return tensor
