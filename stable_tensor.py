import functools

import torch

# Function giving the target one-norm of a STensor based on its shape.
# TARGET_SCALE is a sort of module-wise hyperparameter whose choice
# influences the stability of operations on STensor instances
@functools.lru_cache()
def TARGET_SCALE(shape, nb):
    if isinstance(shape, torch.Tensor):
        shape = shape.shape
    assert len(shape) >= nb >= 0
    shape = shape[nb:]

    # We want to have one_norm(tensor) ~= num_el
    # return torch.log2(torch.prod(torch.tensor(shape)).float())
    return torch.log2(torch.sqrt(torch.prod(torch.tensor(shape)).float()))


### STensor core tools ###

def stensor(data, num_batch, dtype=None, device=None, requires_grad=False, pin_memory=False):
    """
    Constructs a STensor from input data and a partition index placement

    Args:
        data: Initial data for the stensor. Can be a list, tuple,
            NumPy ``ndarray``, scalar, and other types.
        num_batch: Location of partition of data axes separating 
            batch axes and data axes, with negative values counted 
            from end of data tensor
        dtype (optional): the desired data type of returned tensor.
            Default: if ``None``, infers data type from :attr:`data`.
        device (optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
    """
    if isinstance(data, STensor):
        return data.rescale()

    # Convert data to Pytorch tensor if it's not already
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)

    # Initialize with trivial scale tensor
    d_shape = data.shape
    b_shape = d_shape[:num_batch]
    scale = torch.zeros(b_shape, requires_grad=data.requires_grad, 
                        dtype=data.dtype, layout=data.layout, device=data.device)

    # Build and rescale STensor
    stensor = STensor(data, scale)
    stensor.rescale_()
    return stensor

class STensor:
    def __init__(self, data, scale):
        # Check that the shapes of data and scale tensors are compatible
        assert data.shape[:len(scale.shape)] == scale.shape

        self.data = data
        self.scale = scale

    def __str__(self):
        # Slap a disclaimer on any questionable printed values
        disclaimer = ("\ninf and/or zero entries may be artifact of conversion"
                      "\nto non-stable tensor, use repr to view underlying data")
        t = self.to_tensor()
        # Don't slap a disclaimer on normal printed values
        if (torch.any(t != 0) or torch.all(self.data == 0)) \
                            and torch.all(torch.isfinite(t)):
            disclaimer = ""
        # return f"STensor(\n{t}){disclaimer}"
        return f"stable\n{t}{disclaimer}"

    def __repr__(self):
        return f"STensor(data=\n{self.data}, scale=\n{self.scale})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def batch_shape(self):
        return self.scale.shape

    @property
    def data_shape(self):
        return self.data.shape[self.num_batch:]

    @property
    def num_batch(self):
        return len(self.scale.shape)

    @property
    def num_data(self):
        return len(self.data.shape) - self.num_batch

    def rescale_(self):
        """In-place rescaling method"""
        # Get the L1 norm of data and scale correction for each fiber
        nb, nt = self.num_batch, len(self.shape)
        tens_scale = torch.sum(self.data.abs(), dim=list(range(nb, nt)), 
                                keepdim=True)
        log_shift = torch.floor(TARGET_SCALE(self.shape, nb) - 
                                torch.log2(tens_scale))

        # Keep the scale for zero fibers unchanged
        if torch.any(torch.isinf(log_shift)):
            log_shift = torch.where(torch.isfinite(log_shift), log_shift,
                                    torch.zeros_like(log_shift))

        self.data *= 2**log_shift
        self.scale -= log_shift.view_as(self.scale)

    def rescale(self):
        """Return STensor with rescaled data"""
        # Get the L1 norm of data and scale correction for each fiber
        nb, nt = self.num_batch, len(self.shape)
        tens_scale = torch.sum(self.data.abs(), dim=list(range(nb, nt)), 
                                keepdim=True)
        log_shift = torch.floor(TARGET_SCALE(self.shape, nb) - 
                                torch.log2(tens_scale))

        # Keep the scale for zero fibers unchanged
        if torch.any(torch.isinf(log_shift)):
            log_shift = torch.where(torch.isfinite(log_shift), log_shift,
                                    torch.zeros_like(log_shift))
        return STensor(self.data*(2**log_shift), 
                       self.scale-log_shift.view_as(self.scale))

    def to_tensor(self):
        """Return destabilized Tensor version of STensor"""
        long_shape = self.batch_shape + (1,)*self.num_data
        return self.data * 2**self.scale.view(long_shape)

    def __torch_function__(self, fun, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        type_cond = all(issubclass(t, (torch.Tensor,STensor)) for t in types)
        if fun in STABLE_FUNCTIONS and type_cond:
            return STABLE_FUNCTIONS[fun](*args, **kwargs)
        else:
            return NotImplemented


### Wrappers for converting Pytorch functions to ones on STensors ###

# Dictionary to store reimplemented Pytorch functions for use on stensors
STABLE_FUNCTIONS = {}

def register_from_name(fun_name, wrapper):
    """Use torch function name and wrapper to register stabilized function"""
    global STABLE_FUNCTIONS
    torch_fun = getattr(torch, fun_name)
    assert torch_fun not in STABLE_FUNCTIONS
    stable_fun = wrapper(torch_fun)
    STABLE_FUNCTIONS[torch_fun] = stable_fun

def hom_wrap(torch_fun, in_homs, out_homs):
    """
    Wrapper for reasonably simple homogeneous Pytorch functions

    Args:
        torch_fun: Homogeneous Pytorch function to be wrapped
        in_homs:   List of tuples, each of the form (hom_ind, hom_deg), 
                   where hom_ind gives the numerical position of a 
                   homogeneous input argument of torch_fun and hom_deg 
                   gives the degree of homoegeneity
        out_homs:  List of tuples of the same format as in_homs giving
                   homogeneity info for the outputs of torch_fun
    """
    # TODO: Simplify implementation to not need out_homs (only used for var_mean)
    #       
    # Handle case of homogeneous input _region_
    if in_homs[-1][0] < 0:
        # Negative indices -n in last entry of in_homs means that 
        # (n-1)th and future entries to torch_fun are homogeneous
        in_regs = True
        reg_start, reg_deg = in_homs.pop()
        reg_start = -reg_start - 1
    else:
        in_regs = False

    # Reinterpret in_homs, out_homs as dictionaries
    in_homs = {idx: deg for idx, deg in in_homs}
    out_homs = {idx: deg for idx, deg in out_homs}

    # Function for checking if an argument index is homogeneous
    hom_ind = lambda i: i in in_homs or (in_regs and i >= reg_start)

    @functools.wraps(torch_fun)
    def stable_fun(*args, **kwargs):
        # TODO: Do shape inference to be able to handle non-stensor inputs
        # for homogeneous args, warn when there's ambiguity

        # Separate out homogeneous args and put everything in all_args
        all_args, in_scales = [], []
        for i, t in enumerate(args):
            if hom_ind(i):
                # Homogeneous input args
                if not isinstance(t, STensor):
                    breakpoint()
                    raise ValueError(f"Input argument {i} to stable version"
                                    f" {torch_fun.__name__} must be an STensor")
                all_args.append(t.data)
                deg = reg_deg if in_regs and i>=reg_start else in_homs[i]
                in_scales.append(deg * t.scale)
            else:
                # Nonhomogeneous input args
                all_args.append(t)

        # Compute overall rescaling associated with input tensors
        if len(in_scales) > 1:
            out_scale = sum(torch.broadcast_tensors(*in_scales))
        else:
            out_scale = in_scales.pop()

        # Call wrapped Pytorch function, get output as list
        output = torch_fun(*all_args, **kwargs)
        output = list(output) if isinstance(output, tuple) else [output]

        # Convert entries of outputs to STensors as needed
        for i, t in enumerate(output):
            if i in out_homs:
                stens = STensor(t, out_homs[i]*out_scale)
                stens.rescale_()
                output[i] = stens

        return tuple(output) if len(output) > 1 else output[0]

    return stable_fun

### Re-registration of the Pytorch library as stable functions ###

HOMOG = {'abs': ([(0, 1)], [(0, 1)]),
         'bmm': ([(0, 1), (1, 1)], [(0, 1)]),
         'conj': ([(0, 1)], [(0, 1)]),
         'cosine_similarity': ([(0, 0), (1, 0)], [(0, 0)]),
         'cross': ([(0, 1), (1, 1)], [(0, 1)]),
         'div': ([(0, 1), (1, -1)], [(0, 1)]),
         'dot': ([(0, 1), (1, 1)], [(0, 1)]),
         'ger': ([(0, 1), (1, 1)], [(0, 1)]),
         'imag': ([(0, 1)], [(0, 1)]),
         'real': ([(0, 1)], [(0, 1)]),
         'inverse': ([(0, -1)], [(0, 1)]),
         'matmul': ([(0, 1), (1, 1)], [(0, 1)]),
         'mm': ([(0, 1), (1, 1)], [(0, 1)]),
         'mode': ([(0, 1)], [(0, 1)]),
         'mul': ([(0, 1), (1, 1)], [(0, 1)]),
         'mv': ([(0, 1), (1, 1)], [(0, 1)]),
         'pinverse': ([(0, -1)], [(0, 1)]),
         'reciprocal': ([(0, -1)], [(0, 1)]),
         'relu': ([(0, 1)], [(0, 1)]),
         'square': ([(0, 2)], [(0, 1)]),
         'std': ([(0, 1)], [(0, 1)]),
         'sum': ([(0, 1)], [(0, 1)]),
         't': ([(0, 1)], [(0, 1)]),
         'tensordot': ([(0, 1), (1, 1)], [(0, 1)]),
         'trace': ([(0, 1)], [(0, 1)]),
         'transpose': ([(0, 1)], [(0, 1)]),
         'tril': ([(0, 1)], [(0, 1)]),
         'triu': ([(0, 1)], [(0, 1)]),
         'true_divide': ([(0, -1)], [(0, 1)]),
         'var': ([(0, 2)], [(0, 1)]),
         }

# Register all homogeneous functions as possible functions on stensors
for fun_name, value in HOMOG.items():
    in_homs, out_homs = value
    wrapper = functools.partial(hom_wrap, in_homs=in_homs, out_homs=out_homs)
    register_from_name(fun_name, wrapper)

# Doesn't make sense with stensors, and/or method doesn't have PyTorch
# documentation. Not implementing
DOUBTFUL = ['affine_grid_generator', 'alpha_dropout', 'adaptive_avg_pool1d', 
'adaptive_max_pool1d', 'avg_pool1d', 'batch_norm', 
'batch_norm_backward_elemt', 'batch_norm_backward_reduce', 'batch_norm_elemt', 
'batch_norm_gather_stats', 'batch_norm_gather_stats_with_counts', 'batch_norm_stats', 
'batch_norm_update_stats', 'bernoulli', 'bilinear', 'binary_cross_entropy_with_logits', 
'bincount', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'constant_pad_nd', 
'convolution', 'cosine_embedding_loss', 'ctc_loss', 'dequantize', 'dropout', 
'dsmm', 'embedding', 'embedding_bag', 'empty_like', 'fake_quantize_per_channel_affine', 
'fake_quantize_per_tensor_affine', 'fbgemm_linear_fp16_weight', 
'fbgemm_linear_fp16_weight_fp32_activation', 'fbgemm_linear_int8_weight', 
'fbgemm_linear_int8_weight_fp32_activation', 'fbgemm_linear_quantize_weight', 
'fbgemm_pack_gemm_matrix_fp16', 'fbgemm_pack_quantized_matrix', 
'feature_alpha_dropout', 'feature_dropout', 'frobenius_norm', 'geqrf', 'grid_sampler', 
'grid_sampler_2d', 'grid_sampler_3d', 'group_norm', 'gru', 'gru_cell', 
'hardshrink', 'hinge_embedding_loss', 'histc', 'hsmm', 'hspmm',
'index_add', 'index_copy', 'index_fill', 'index_put', 'index_select', 
'instance_norm', 'int_repr', 'is_distributed', 
'is_signed', 'isclose', 'kl_div', 'layer_norm', 
'lobpcg', 'log_softmax', 'lstm', 'lstm_cell', 
'margin_ranking_loss', 'masked_fill', 'masked_scatter', 
'masked_select', 'matrix_rank', 'max_pool1d', 'max_pool1d_with_indices', 
'max_pool2d', 'max_pool3d', 
'meshgrid', 'miopen_batch_norm', 'miopen_convolution', 'miopen_convolution_transpose', 
'miopen_depthwise_convolution', 'miopen_rnn', 
'multinomial', 'mvlgamma', 'native_batch_norm', 'native_layer_norm', 'native_norm', 
'neg', 'nonzero', 'norm', 'norm_except_dim', 'normal', 'nuclear_norm', 'ones_like', 
'pairwise_distance', 'pixel_shuffle', 'poisson', 'poisson_nll_loss', 'polygamma', 
'prelu', 'q_per_channel_axis', 
'q_per_channel_scales', 'q_per_channel_zero_points', 'q_scale', 'q_zero_point', 
'quantize_per_channel', 'quantize_per_tensor', 'quantized_batch_norm', 
'quantized_gru', 'quantized_gru_cell', 'quantized_lstm', 'quantized_lstm_cell', 
'quantized_max_pool2d', 'quantized_rnn_relu_cell', 'quantized_rnn_tanh_cell', 
'rand_like', 'randint_like', 'randn_like', 
'remainder', 'renorm', 'repeat_interleave', 'result_type', 
'rnn_relu', 'rnn_relu_cell', 'rnn_tanh', 'rnn_tanh_cell', 'rot90', 'rrelu', 
'rsub', 'saddmm', 'scalar_tensor', 'scatter', 'scatter_add', 'select', 
'selu', 'smm', 'softmax', 'split_with_sizes', 'spmm', 'sspaddmm', 'sub', 'threshold', 
'topk', 'tril_indices', 'triu_indices', 'triplet_margin_loss', 'unbind', 'zeros_like', ]

# Important and/or easy functions
TORCH = ['acos', 'angle', 'asin', 'atan', 'atan2', 'cartesian_prod', 'ceil', 
'celu', 'clamp', 'clamp_max', 'clamp_min', 'conv1d', 'conv2d', 'conv3d', 'conv_tbc', 
'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d', 'cos', 'cosh', 'digamma', 
'erf', 'erfc', 'erfinv', 'exp', 'expm1', 'fft', 'floor', 'frac', 'fmod', 'ifft', 
'irfft', 'is_complex', 'is_floating_point', 'is_nonzero', 'is_same_size', 
'isfinite', 'isinf', 'isnan', 'kthvalue', 'lerp', 'lgamma', 'logdet', 'logical_and', 
'logical_not', 'logical_or', 'logical_xor', 'numel', 'rfft', 
'round', 'sigmoid', 'sin', 'sinh', 'stft', 'tan', 'tanh', 'trunc', ]
DO_NOW = ['add', 'cumsum', 'dist', 'einsum', 'eq', 'ne', 'equal', 'ge', 'gt', 'le', 'lt', 
'log', 'log10', 'log1p', 'log2', 'max', 'min', 
'prod', 'sign', 'sqrt', 'rsqrt', ]

# Somewhat important and/or trickier functions
DO_SOON = ['allclose', 'all', 'any', 'argmax', 'argmin', 'argsort', 
'broadcast_tensors', '***broadcast_batch***', '***broadcast_data***', 
'cat', 'stack', 'chain_matmul', 'cumprod', 'det', 'detach', 'diag', 'diagonal', 
'flatten', 'flip', 'floor_divide', 'gather', 'logsumexp', 'matrix_power', 
'mean', 'median', 'pow', 'reshape', 'squeeze', 'unsqueeze', 'sort', 'split', 
# Homogeneous functions that for one reason or another can't be handled by hom_wrap
'pdist', 'trapz', 'take', 'unique_consecutive', 'var_mean', 'lu_solve', 'std_mean', 
# Matrix decompositions whose return types must be respected
'qr', 'eig', 'lstsq', 'svd', 'symeig', 'triangular_solve', 'solve']
# ***broadcast_{batch,data}***: Functions which only broadcast a subset of the indices,
#                               I believe with no restriction on the other subset

# Not important, could be tough
LATER = ['addbmm', 'addcdiv', 'addcmul', 'addmm', 'addmv', 'addr', 
'baddbmm', 'cdist', 'cholesky', 'cholesky_inverse', 'cholesky_solve', 
'chunk', 'clone', 'combinations', 'cummax', 'cummin', 'diag_embed', 'diagflat', 
'full_like', 'narrow', 'orgqr', 'ormqr', 'roll', 'slogdet', 'where', ]

if __name__ == '__main__':
    # m = stensor(torch.ones((5,2,2)), 1)
    # v = stensor(torch.ones((5,2,3)), 1)
    # mv = torch.bmm(m, v)
    mat = stensor(1024*torch.eye(5), 0)
    print(repr(torch.pinverse(mat)))


    # Make sure there aren't functions which can be overwridden but don't appear above
    func_dict = torch._overrides.get_overridable_functions()
    fun_names = [f.__name__ for f in func_dict[torch]]
    for name in fun_names:
        if all(name not in big_set for big_set in [HOMOG.keys(), DOUBTFUL, TORCH, DO_NOW, DO_SOON, LATER]):
            assert False, name

    # import inspect
    # override_dict = torch._overrides.get_testing_overrides()
    # for fun_name in HOMOG:
    #     if HOMOG[fun_name][0][0] != () and HOMOG[fun_name][1][0] != ():
    #         continue
    #     dummy_fun = override_dict[getattr(torch, fun_name)]
    #     print(f"{fun_name}: {inspect.signature(dummy_fun)}")