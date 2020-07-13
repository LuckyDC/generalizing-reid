from torch.nn.modules.batchnorm import _BatchNorm


def network_to_half(module):
    return norm_convert_float(module.half())


def norm_convert_float(module):
    '''
    BatchNorm layers need parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    '''
    if isinstance(module, _BatchNorm):
        module.float()
    for child in module.children():
        norm_convert_float(child)
    return module
