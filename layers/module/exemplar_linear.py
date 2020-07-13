import torch
import torch.nn as nn

from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn import init


class ExemplarLinearFunc(Function):

    @staticmethod
    def forward(ctx, input, memory, target, momentum=0.1):
        ctx.save_for_backward(memory, input, target)
        ctx.momentum = momentum

        return torch.mm(input, memory.t())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        memory, input, target = ctx.saved_tensors

        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(memory)

        momentum = ctx.momentum
        memory[target] *= momentum
        memory[target] += (1 - momentum) * input
        memory[target] /= torch.norm(memory[target], p=2, dim=1, keepdim=True)

        return grad_input, None, None, None


class ExemplarLinear(nn.Module):
    def __init__(self, num_instances, num_features, momentum=0.1):
        super(ExemplarLinear, self).__init__()

        self.num_instances = num_instances
        self.num_features = num_features
        self.momentum = momentum

        self.register_buffer('memory', torch.Tensor(num_instances, num_features))

        self.reset_buffers()

    def set_momentum(self, value):
        self.momentum = value

    def reset_buffers(self):
        init.normal_(self.memory, std=0.001)

    def forward(self, x, targets):
        return ExemplarLinearFunc.apply(x, self.memory, targets, self.momentum)
