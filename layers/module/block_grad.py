from torch import nn
from torch.autograd import Function


class BlockGradFunction(Function):
    @staticmethod
    def forward(ctx, data):
        return data

    @staticmethod
    def backward(ctx, grad_outputs):
        grad = None

        if ctx.needs_input_grad[0]:
            grad = grad_outputs.new_zeros(grad_outputs.size())

        return grad


class BlockGrad(nn.Module):
    def __init__(self):
        super(BlockGrad, self).__init__()

    def forward(self, data):
        return BlockGradFunction.apply(data)
