import copy

from torch import nn


def clone_without_grad(module):
    assert isinstance(module, nn.Module)

    mod = copy.deepcopy(module)

    for name, param in module.named_parameters():
        if '.' not in name:
            setattr(mod, name, nn.Parameter(param.data, requires_grad=False))
        else:
            splits = name.split('.')
            attr = mod

            for s in splits[:-1]:
                attr = getattr(attr, s)

            setattr(attr, splits[-1], nn.Parameter(param.data, requires_grad=False))

    return mod
