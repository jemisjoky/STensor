import torch

def bad_conversion(stens, tensor):
    """Check if conversion to tensor led to underflow/overflow problems"""
    underflow = torch.any(torch.logical_and(tensor==0, stens.data!=0))
    overflow = torch.any(torch.logical_and(torch.isinf(tensor), 
                                           torch.isfinite(stens.data)))
    return underflow or overflow

def tupleize(dim, ndim):
    """Convert one or more dims to a tuple of non-negative indices"""
    if not isinstance(dim, tuple):
        if hasattr(dim, '__iter__'):
            dim = tuple(dim)
        else:
            dim = (dim,)
    return tuple((i if i >=0 else ndim+i) for i in dim)

def squeeze_dims(tensor, dims):
    """Squeeze multiple singleton dimensions from a tensor input"""
    shape = tensor.shape
    assert all(shape[i] == 1 for i in dims)
    new_shape = tuple(d for i, d in enumerate(shape) if i not in dims)
    return tensor.view(new_shape)