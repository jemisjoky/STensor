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
    return torch.log2(torch.prod(torch.tensor(shape)).float())


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


### Tools to convert/register Pytorch functions as ones on STensors ###

# Dictionary to store reimplemented Pytorch functions for use on stensors
STABLE_FUNCTIONS = {}

def existing_method_from_name(fun_name):
    """Add method to STensor for existing stable function"""
    global STensor
    assert hasattr(torch.Tensor, fun_name)
    if getattr(torch, fun_name) in STABLE_FUNCTIONS:
        stable_fun = STABLE_FUNCTIONS[getattr(torch, fun_name)]
        setattr(STensor, fun_name, stable_fun)
    else:
        print(f"STILL NEED TO IMPLEMENT {fun_name}")

def hom_wrap(fun_name, in_homs, in_place=False):
    """
    Wrapper for reasonably simple homogeneous Pytorch functions

    Args:
        fun_name:  Name of homogeneous Pytorch function to be wrapped
        in_homs:   List of tuples, each of the form (hom_ind, hom_deg), 
                   where hom_ind gives the numerical position of a 
                   homogeneous input argument of torch_fun and hom_deg 
                   gives the degree of homoegeneity
        in_place:  Boolean specifying if we should implement the operation 
                   as an in-place method
    """
    # TODO: Deal with non-stensors as inputs
    num_homs = len(in_homs)
    if in_place:
        torch_fun = getattr(torch.Tensor, fun_name)
    else:
        torch_fun = getattr(torch, fun_name)

    @functools.wraps(torch_fun)
    def stable_fun(*args, **kwargs):
        # TODO: Do shape inference to be able to handle non-stensor inputs
        # for homogeneous args, warn when there's ambiguity

        # Separate out homogeneous args and put everything in all_args
        all_args, in_scales = [], []
        for i, t in enumerate(args):
            if i < num_homs:
                # Homogeneous input args
                if not isinstance(t, STensor):
                    raise ValueError(f"Input argument {i} to stable version"
                                    f" {torch_fun.__name__} must be an STensor")
                all_args.append(t.data)
                in_scales.append(in_homs[i] * t.scale)
            else:
                # Nonhomogeneous input args
                all_args.append(t)

        # Compute overall rescaling associated with input tensors
        if len(in_scales) > 1:
            out_scale = sum(torch.broadcast_tensors(*in_scales))
        else:
            out_scale = in_scales.pop()

        # Call wrapped Pytorch function, get output as list, and return
        # Different behavior for in-place vs regular cases
        if in_place:
            # Call in-place method of data tensor, then readjust scale
            self = args[0]  # <- Object whose method is being called
            getattr(self.data, fun_name)(*all_args[1:], **kwargs)
            self.scale = out_scale
            self.rescale_()
        else:
            # Call Torch function with data, then convert to stensor
            output = torch_fun(*all_args, **kwargs)
            assert isinstance(output, torch.Tensor)
            stens = STensor(output, out_scale)
            stens.rescale_()
            return stens

    return stable_fun

def inplace_hom_method_from_name(fun_name):
    """Add in-place versions of homogeneous function to STensor"""
    assert hasattr(torch.Tensor, fun_name)
    assert fun_name[-1] == '_'
    in_homs = HOMOG[fun_name[:-1]]
    stable_method = hom_wrap(fun_name, in_homs, in_place=True)
    setattr(STensor, fun_name, stable_method)

### Re-registration of the Pytorch library as stable functions ###

HOMOG = {'abs': (1,),
         'bmm': (1, 1),
         'conj': (1,),
         'cosine_similarity': (0, 0),
         'cross': (1, 1),
         'div': (1, -1),
         'dot': (1, 1),
         'ger': (1, 1),
         'imag': (1,),
         'real': (1,),
         'inverse': (-1,),
         'matmul': (1, 1),
         'mm': (1, 1),
         'mode': (1,),
         'mul': (1, 1),
         'mv': (1, 1),
         'pinverse': (-1,),
         'reciprocal': (-1,),
         'relu': (1,),
         'square': (2,),
         'std': (1,),
         'sum': (1,),
         't': (1,),
         'tensordot': (1, 1),
         'trace': (1,),
         'true_divide': (-1,),
         'var': (2,),
         }

# Register all homogeneous functions as possible functions on stensors
for fun_name, in_homs in HOMOG.items():
    torch_fun = getattr(torch, fun_name)
    assert torch_fun not in STABLE_FUNCTIONS
    stable_fun = hom_wrap(fun_name, in_homs, in_place=False)
    STABLE_FUNCTIONS[torch_fun] = stable_fun

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
'irfft', 'is_complex', 'is_floating_point', 'is_nonzero', 'is_same_size', 'kthvalue', 
'lerp', 'lgamma', 'logdet', 'numel', 'rfft', 
'round', 'sigmoid', 'sin', 'sinh', 'stft', 'tan', 'tanh', 'trunc', ]
DO_NOW = ['add', 'cumsum', 'dist', 'einsum', 'eq', 'ne', 'equal', 'ge', 'gt', 'le', 'lt', 
'log', 'log10', 'log1p', 'log2', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 
'max', 'min', 'prod', 'sign', 'sqrt', 'rsqrt', 'isfinite', 'isinf', 'isnan', ]

# Somewhat important and/or trickier functions
DO_SOON = ['allclose', 'all', 'any', 'argmax', 'argmin', 'argsort', 
'broadcast_tensors', '***broadcast_batch***', '***broadcast_data***', 
'cat', 'stack', 'chain_matmul', 'cumprod', 'det', 'detach', 'diag', 'diagonal', 
'flatten', 'flip', 'floor_divide', 'gather', 'logsumexp', 'matrix_power', 
'mean', 'median', 'pow', 'reshape', 'squeeze', 'unsqueeze', 'sort', 'split', 
# Homogeneous functions that for one reason or another can't be handled by hom_wrap
'pdist', 'trapz', 'take', 'unique_consecutive', 'var_mean', 'lu_solve', 'std_mean', 
'tril', 'triu', 'transpose', 
# Matrix decompositions whose return types must be respected
'qr', 'eig', 'lstsq', 'svd', 'symeig', 'triangular_solve', 'solve']
# ***broadcast_{batch,data}***: Functions which only broadcast a subset of the indices,
#                               I believe with no restriction on the other subset

# Not important, could be tough
LATER = ['addbmm', 'addcdiv', 'addcmul', 'addmm', 'addmv', 'addr', 
'baddbmm', 'cdist', 'cholesky', 'cholesky_inverse', 'cholesky_solve', 
'chunk', 'clone', 'combinations', 'cummax', 'cummin', 'diag_embed', 'diagflat', 
'full_like', 'narrow', 'orgqr', 'ormqr', 'roll', 'slogdet', 'where', ]

ALL_FUN = list(HOMOG.keys()) + DOUBTFUL + TORCH + DO_NOW + DO_SOON + LATER

### Register stabilized Pytorch functions as methods for stensors ###

EXISTING_METHOD = ['abs', 'bmm', 'conj', 'cross', 'div', 'dot', 'ger', 'inverse', 
'matmul', 'mm', 'mode', 'mul', 'mv', 'pinverse', 'reciprocal', 'relu', 'square', 
'std', 'sum', 't', 'trace', 'transpose', 'tril', 'triu', 'true_divide', 'var',
'acos', 'angle', 'asin', 'atan', 'atan2', 'ceil', 'clamp', 
'clamp_max', 'clamp_min', 'cos', 'cosh', 'digamma', 'erf', 'erfc', 'erfinv', 
'exp', 'expm1', 'fft', 'floor', 'fmod', 'frac', 'ifft', 'irfft', 'is_complex', 
'is_floating_point', 'is_nonzero', 'is_same_size', 'kthvalue', 'lerp', 'lgamma', 
'logdet', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'numel', 
'rfft', 'round', 'sigmoid', 'sin', 'sinh', 'stft', 'tan', 'tanh', 'trunc']

HOM_INPLACE = ['abs_', 'div_', 'mul_', 'reciprocal_', 'relu_', 'square_', 't_', 
'true_divide_']

# Add all methods which I've already implement as stable functions
for name in EXISTING_METHOD:
    existing_method_from_name(name)

# Implement in-place versions of homogeneous functions
for name in HOM_INPLACE:
    inplace_hom_method_from_name(name)

TORCH_INPLACE = ['acos_', 'asin_', 'atan2_', 'atan_', 'ceil_', 'clamp_', 'clamp_max_', 
'clamp_min_', 'cos_', 'cosh_', 'digamma_', 'erf_', 'erfc_', 'erfinv_', 'exp_', 'expm1_', 
'floor_', 'fmod_', 'frac_', 'lerp_', 'lgamma_', 'round_', 'sigmoid_', 'sin_', 
'sinh_', 'tan_', 'tanh_', 'trunc_']

ATTRIBUTES = ['T', '__abs__', '__add__', '__and__' ,'__array__', '__array_priority__', 
'__array_wrap__', '__bool__', '__contains__', '__deepcopy__', '__delitem__', 
'__div__', '__float__', '__floordiv__', '__getitem__', '__iadd__', '__iand__', 
'__idiv__', '__ifloordiv__', '__ilshift__', '__imul__', '__index__', '__int__', 
'__invert__', '__ior__', '__ipow__', '__irshift__', '__isub__', '__iter__', 
'__itruediv__', '__ixor__', '__len__', '__long__', '__lshift__', '__matmul__', '__mod__', 
'__mul__', '__neg__', '__nonzero__', '__or__', '__pow__', '__radd__', '__rdiv__', 
'__reversed__', '__rfloordiv__', '__rmul__', '__rpow__', '__rshift__', '__rsub__', 
'__rtruediv__', '__setitem__', '__setstate__', '__sub__', '__truediv__', '__xor__', 

'_backward_hooks', '_base', '_cdata', '_coalesced_', '_dimI', '_dimV', '_grad', 
'_grad_fn', '_indices', '_is_view', '_make_subclass', '_nnz', '_update_names', '_values', 
'_version', 'abs', 'abs_', 'acos', 'acos_', 'add', 'add_', 'addbmm', 'addbmm_', 'addcdiv', 
'addcdiv_', 'addcmul', 'addcmul_', 'addmm', 'addmm_', 'addmv', 'addmv_', 'addr', 'addr_', 
'align_as', 'align_to', 'all', 'allclose', 'angle', 'any', 'apply_', 'argmax', 'argmin', 
'argsort', 'as_strided', 'as_strided_', 'asin', 'asin_', 'atan', 'atan2', 'atan2_', 
'atan_', 'backward', 'baddbmm', 'baddbmm_', 'bernoulli', 'bernoulli_', 'bfloat16', 
'bincount', 'bitwise_and', 'bitwise_and_', 'bitwise_not', 'bitwise_not_', 'bitwise_or', 
'bitwise_or_', 'bitwise_xor', 'bitwise_xor_', 'bmm', 'bool', 'byte', 'cauchy_', 'ceil', 
'ceil_', 'char', 'cholesky', 'cholesky_inverse', 'cholesky_solve', 'chunk', 'clamp', 
'clamp_', 'clamp_max', 'clamp_max_', 'clamp_min', 'clamp_min_', 'clone', 'coalesce', 
'conj', 'contiguous', 'copy_', 'cos', 'cos_', 'cosh', 'cosh_', 'cpu', 'cross', 'cuda', 
'cummax', 'cummin', 'cumprod', 'cumsum', 'data', 'data_ptr', 'dense_dim', 'dequantize', 
'det', 'detach', 'detach_', 'device', 'diag', 'diag_embed', 'diagflat', 'diagonal', 
'digamma', 'digamma_', 'dim', 'dist', 'div', 'div_', 'dot', 'double', 'dtype', 'eig', 
'element_size', 'eq', 'eq_', 'equal', 'erf', 'erf_', 'erfc', 'erfc_', 'erfinv', 'erfinv_', 
'exp', 'exp_', 'expand', 'expand_as', 'expm1', 'expm1_', 'exponential_', 'fft', 'fill_', 
'fill_diagonal_', 'flatten', 'flip', 'float', 'floor', 'floor_', 'floor_divide', 
'floor_divide_', 'fmod', 'fmod_', 'frac', 'frac_', 'gather', 'ge', 'ge_', 'geometric_', 
'geqrf', 'ger', 'get_device', 'grad', 'grad_fn', 'gt', 'gt_', 'half', 'hardshrink', 
'has_names', 'histc', 'ifft', 'index_add', 'index_add_', 'index_copy', 'index_copy_', 
'index_fill', 'index_fill_', 'index_put', 'index_put_', 'index_select', 'indices', 'int', 
'int_repr', 'inverse', 'irfft', 'is_coalesced', 'is_complex', 'is_contiguous', 'is_cuda', 
'is_distributed', 'is_floating_point', 'is_leaf', 'is_mkldnn', 'is_nonzero', 'is_pinned', 
'is_quantized', 'is_same_size', 'is_set_to', 'is_shared', 'is_signed', 'is_sparse', 
'isclose', 'item', 'kthvalue', 'layout', 'le', 'le_', 'lerp', 'lerp_', 'lgamma', 
'lgamma_', 'log', 'log10', 'log10_', 'log1p', 'log1p_', 'log2', 'log2_', 'log_', 
'log_normal_', 'log_softmax', 'logdet', 'logical_and', 'logical_and_', 'logical_not', 
'logical_not_', 'logical_or', 'logical_or_', 'logical_xor', 'logical_xor_', 'logsumexp', 
'long', 'lstsq', 'lt', 'lt_', 'lu', 'lu_solve', 'map2_', 'map_', 'masked_fill', 
'masked_fill_', 'masked_scatter', 'masked_scatter_', 'masked_select', 'matmul', 
'matrix_power', 'max', 'mean', 'median', 'min', 'mm', 'mode', 'mul', 'mul_', 
'multinomial', 'mv', 'mvlgamma', 'mvlgamma_', 'name', 'names', 'narrow', 'narrow_copy', 
'ndim', 'ndimension', 'ne', 'ne_', 'neg', 'neg_', 'nelement', 'new', 'new_empty', 
'new_full', 'new_ones', 'new_tensor', 'new_zeros', 'nonzero', 'norm', 'normal_', 'numel', 
'numpy', 'orgqr', 'ormqr', 'output_nr', 'permute', 'pin_memory', 'pinverse', 'polygamma', 
'polygamma_', 'pow', 'pow_', 'prelu', 'prod', 'put_', 'q_per_channel_axis', 
'q_per_channel_scales', 'q_per_channel_zero_points', 'q_scale', 'q_zero_point', 'qr', 
'qscheme', 'random_', 'reciprocal', 'reciprocal_', 'record_stream', 'refine_names', 
'register_hook', 'reinforce', 'relu', 'relu_', 'remainder', 'remainder_', 'rename', 
'rename_', 'renorm', 'renorm_', 'repeat', 'repeat_interleave', 'requires_grad', 
'requires_grad_', 'reshape', 'reshape_as', 'resize', 'resize_', 'resize_as', 'resize_as_', 
'retain_grad', 'rfft', 'roll', 'rot90', 'round', 'round_', 'rsqrt', 'rsqrt_', 'scatter', 
'scatter_', 'scatter_add', 'scatter_add_', 'select', 'set_', 'shape', 'share_memory_', 
'short', 'sigmoid', 'sigmoid_', 'sign', 'sign_', 'sin', 'sin_', 'sinh', 'sinh_', 'size', 
'slogdet', 'smm', 'softmax', 'solve', 'sort', 'sparse_dim', 'sparse_mask', 
'sparse_resize_', 'sparse_resize_and_clear_', 'split', 'split_with_sizes', 'sqrt', 
'sqrt_', 'square', 'square_', 'squeeze', 'squeeze_', 'sspaddmm', 'std', 'stft', 
'storage', 'storage_offset', 'storage_type', 'stride', 'sub', 'sub_', 'sum', 
'sum_to_size', 'svd', 'symeig', 't', 't_', 'take', 'tan', 'tan_', 'tanh', 'tanh_', 'to', 
'to_dense', 'to_mkldnn', 'to_sparse', 'tolist', 'topk', 'trace', 'transpose', 
'transpose_', 'triangular_solve', 'tril', 'tril_', 'triu', 'triu_', 'true_divide', 
'true_divide_', 'trunc', 'trunc_', 'type', 'type_as', 'unbind', 'unflatten', 'unfold', 
'uniform_', 'unique', 'unique_consecutive', 'unsqueeze', 'unsqueeze_', 'values', 'var', 
'view', 'view_as', 'where', 'zero_']


### TODOS ###
# (1)  Redo triu, tril to work with batch indices
# (2)  Redo transpose to ensure we aren't flipping batch and data indices


if __name__ == '__main__':
    # mat = stensor(torch.randn(200, 200), 0)
    # print(dir(mat))
    # mat.scale *= 150
    # print(mat.scale)
    # print(torch.norm(mat.data))
    # mat = mat.matmul(mat)
    # print(mat.scale)
    # print(torch.norm(mat.data))

    # Get the nontrivial attributes of a Pytorch tensor
    class Objer:
        def __init__(self):
            pass
    obj = Objer()
    tens_atts = [f for f in dir(torch.ones(2)) if f not in dir(obj)]
    hom_atts = [f for f in tens_atts if f in HOMOG]
    torch_atts = [f for f in tens_atts if f in TORCH]
    other_atts = [f for f in tens_atts if f not in EXISTING_METHOD+HOM_INPLACE+TORCH_INPLACE]
    # print(other_atts)


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