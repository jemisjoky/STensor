from functools import lru_cache

import torch

# from stable_funs import STABLE_FUNCTIONS

# Function giving the target one-norm of a STensor based on its shape.
# TARGET_SCALE is a sort of module-wise hyperparameter whose choice
# influences the stability of operations on STensor instances
@lru_cache()
def TARGET_SCALE(shape, nb):
    if isinstance(shape, torch.Tensor):
        shape = shape.shape
    assert len(shape) >= nb >= 0
    shape = shape[nb:]

    # We want to have one_norm(tensor) ~= num_el
    # return torch.log2(torch.prod(torch.tensor(shape)).float())
    return torch.log2(torch.sqrt(torch.prod(torch.tensor(shape)).float()))


### STensor core tools ###

class STensor:
    def __init__(self, data, scale):
        # Check that the shapes of data and scale tensors are compatible
        batch_shape = scale.shape
        assert data.shape[:len(batch_shape)] == batch_shape

        self.data = data
        self.scale = scale


    # Handles indexing of STensor
    # def __getitem__(self, key):
    #     num_inds = len(key) if hasattr(key, '__len__') else 1
    #     nb = len(self.scale.shape)

    #     # Only indexing of batch indices is supported
    #     if num_inds > nb:
    #         raise ValueError("Attempted to index beyond batch indices of "
    #                         f"HomArray ({num_inds} indices requested, but "
    #                         f"only {max_inds} in HomArray). Maybe try "
    #                         "using to_array first, then indexing?")
    #     else:
    #         return STensor(self.data.__getitem__(key),
    #                         scale=self.scale.__getitem__(key))

    def __repr__(self):
        return f"STensor(tensor={self.data},\nscale={self.scale})"

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
        nb, nt = self.num_batch, len(self.shape)
        tens_scale = torch.sum(self.data.abs(), dim=list(range(nb, nt)), 
                                keepdim=True)
        log_shift = torch.floor(TARGET_SCALE(self.shape, nb) - 
                                torch.log2(tens_scale))
        self.data *= 2**log_shift
        self.scale -= log_shift.view_as(self.scale)

    def rescale(self):
        """Return STensor with rescaled data"""
        nb, nt = self.num_batch, len(self.shape)
        tens_scale = torch.sum(self.data.abs(), dim=list(range(nb, nt)), 
                                keepdim=True)
        log_shift = torch.floor(TARGET_SCALE(self.shape, nb) - 
                                torch.log2(tens_scale))
        return STensor(self.data*(2**log_shift), 
                       self.scale-log_shift.view_as(self.scale))

    def as_tensor(self):
        """Return destabilized Tensor version of STensor"""
        long_shape = self.batch_shape + (1,)*self.num_data
        return self.data * 2**self.scale.view(long_shape)

    def __torch_function__(self, fun, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        type_cond = all(issubclass(t, (torch.Tensor,STensor)) 
                                    for t in types)
        if fun in STABLE_FUNCTIONS and type_cond:
            return STABLE_FUNCTIONS[fun](*args, **kwargs)
        else:
            return NotImplemented


### Pytorch functions on STensors ###




# Doesn't make sense with STensors, and/or method doesn't have PyTorch
# documentation. Not implementing
DOUBTFUL = ['affine_grid_generator', 'align_tensors', 'alpha_dropout', 
'alpha_dropout_', 'adaptive_avg_pool1d', 'adaptive_max_pool1d', 'arange', 
'as_strided', 'as_strided_', 'avg_pool1d', 'bartlett_window', 'batch_norm', 
'batch_norm_backward_elemt', 'batch_norm_backward_reduce', 'batch_norm_elemt', 
'batch_norm_gather_stats', 'batch_norm_gather_stats_with_counts', 'batch_norm_stats', 
'batch_norm_update_stats', 'bernoulli', 'bilinear', 'binary_cross_entropy_with_logits', 
'bincount', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 
'blackman_window', 'can_cast', 'compiled_with_cxx11_abi', 'constant_pad_nd', 
'convolution', 'cosine_embedding_loss', 'ctc_loss', 'cudnn_affine_grid_generator', 
'cudnn_batch_norm', 'cudnn_convolution', 'cudnn_convolution_transpose', 
'cudnn_grid_sampler', 'cudnn_is_acceptable', 'dequantize', 'dropout', 'dropout_', 
'dsmm', 'embedding', 'embedding_bag', 'embedding_renorm_', 'empty', 'empty_like', 
'empty_strided', 'enable_grad', 'eye', 'fake_quantize_per_channel_affine', 
'fake_quantize_per_tensor_affine', 'fbgemm_linear_fp16_weight', 
'fbgemm_linear_fp16_weight_fp32_activation', 'fbgemm_linear_int8_weight', 
'fbgemm_linear_int8_weight_fp32_activation', 'fbgemm_linear_quantize_weight', 
'fbgemm_pack_gemm_matrix_fp16', 'fbgemm_pack_quantized_matrix', 
'feature_alpha_dropout', 'feature_alpha_dropout_', 'feature_dropout', 
'feature_dropout_', 'fill_', 'finfo', 'fork', 'frobenius_norm', 'from_file', 
'from_numpy', 'geqrf', 'get_default_dtype', 'get_device', 'get_file_path', 
'get_num_interop_threads', 'get_num_threads', 'get_rng_state', 'grid_sampler', 
'grid_sampler_2d', 'grid_sampler_3d', 'group_norm', 'gru', 'gru_cell', 
'hamming_window', 'hann_window', 'hardshrink', 'hinge_embedding_loss', 'histc', 
'hsmm', 'hspmm', 'iinfo', 'import_ir_module', 'import_ir_module_from_buffer', 
'index_add', 'index_copy', 'index_fill', 'index_put', 'index_put_', 'index_select', 
'initial_seed', 'instance_norm', 'int_repr', 'is_anomaly_enabled', 'is_distributed', 
'is_signed', 'is_storage', 'isclose', 'kl_div', 'layer_norm', 'layout', 'linspace', 
'load', 'lobpcg', 'log_softmax', 'logspace', 'lstm', 'lstm_cell', 'lu_unpack', 
'manual_seed', 'margin_ranking_loss', 'masked_fill', 'masked_scatter', 
'masked_select', 'matrix_rank', 'max_pool1d', 'max_pool1d_with_indices', 
'max_pool2d', 'max_pool3d', 'memory_format', 'merge_type_from_type_comment', 
'meshgrid', 'miopen_batch_norm', 'miopen_convolution', 'miopen_convolution_transpose', 
'miopen_depthwise_convolution', 'miopen_rnn', 'mkldnn_adaptive_avg_pool2d', 
'mkldnn_convolution', 'mkldnn_convolution_backward_weights', 'mkldnn_max_pool2d', 
'multinomial', 'mvlgamma', 'native_batch_norm', 'native_layer_norm', 'native_norm', 
'neg', 'neg_', 'no_grad', 'nonzero', 'norm', 'norm_except_dim', 'normal', 
'nuclear_norm', 'ones', 'ones_like', 'pairwise_distance', 'parse_ir', 'parse_schema', 
'parse_type_comment', 'pixel_shuffle', 'poisson', 'poisson_nll_loss', 'polygamma', 
'prelu', 'prepare_multiprocessing_environment', 'promote_types', 'q_per_channel_axis', 
'q_per_channel_scales', 'q_per_channel_zero_points', 'q_scale', 'q_zero_point', 
'qscheme', 'quantize_per_channel', 'quantize_per_tensor', 'quantized_batch_norm', 
'quantized_gru', 'quantized_gru_cell', 'quantized_lstm', 'quantized_lstm_cell', 
'quantized_max_pool2d', 'quantized_rnn_relu_cell', 'quantized_rnn_tanh_cell', 
'rand', 'rand_like', 'randint', 'randint_like', 'randn', 'randn_like', 'randperm', 
'range', 'relu', 'relu_', 'remainder', 'renorm', 'repeat_interleave', 'result_type', 
'rnn_relu', 'rnn_relu_cell', 'rnn_tanh', 'rnn_tanh_cell', 'rot90', 'rrelu', 'rrelu_', 
'rsub', 'saddmm', 'save', 'scalar_tensor', 'scatter', 'scatter_add', 'seed', 'select', 
'selu', 'selu_', 'set_anomaly_enabled', 'set_default_dtype', 'set_default_tensor_type', 
'set_flush_denormal', 'set_grad_enabled', 'set_num_interop_threads', 'set_num_threads', 
'set_printoptions', 'set_rng_state', 'smm', 'softmax', 'sparse_coo_tensor', 
'split_with_sizes', 'spmm', 'sspaddmm', 'sub', 'tensor', 'threshold', 'threshold_', 
'topk', 'tril_indices', 'triu_indices', 'triplet_margin_loss', 'typename', 'unbind', 
'wait', 'zero_', 'zeros', 'zeros_like', ]

# Important and/or easy functions
TORCH = ['acos', 'acos_', 'angle', 'asin', 'asin_', 'atan', 'atan2', 'atan_', 
'ceil', 'ceil_', 'celu', 'celu_', 'clamp', 'clamp_', 'clamp_max', 'clamp_max_', 
'clamp_min', 'clamp_min_', 'conv1d', 'conv2d', 'conv3d', 'conv_tbc', 'conv_transpose1d', 
'conv_transpose2d', 'conv_transpose3d', 'cos', 'cos_', 'cosh', 'cosh_', 'digamma', 
'erf', 'erf_', 'erfc', 'erfc_', 'erfinv', 'exp', 'exp_', 'expm1', 'expm1_', 'fft', 
'floor', 'floor_', 'frac', 'frac_', 'fmod', 'ifft', 'irfft', 'is_complex', 
'is_floating_point', 'is_grad_enabled', 'is_nonzero', 'is_same_size', 'is_tensor', 
'isfinite', 'isinf', 'isnan', 'kthvalue', 'lerp', 'lgamma', 'logdet', 'logical_and', 
'logical_not', 'logical_or', 'logical_xor', 'numel', 'pca_lowrank', 'rfft', 
'round', 'round_', 'sigmoid', 'sigmoid_', 'sin', 'sin_', 'sinh', 'sinh_', 'stft', 
'tan', 'tan_', 'tanh', 'tanh_', 'trunc', 'trunc_', ]
HOMOG = ['abs', 'abs_', 'bmm', 'cartesian_prod', 'conj', 'cosine_similarity', 'cross', 
'div', 'dot', 'eig', 'ger', 'imag', 'real', 'inverse', 'lstsq', 'lu', 'lu_solve', 'matmul', 
'mm', 'mode', 'mul', 'mv', 'pdist', 'pinverse', 'qr', 'reciprocal', 'reciprocal_', 
'square', 'square_', 'stack', 'std', 'std_mean', 'svd', 'sum', 'symeig', 't', 'take', 
'svd_lowrank', 'tensordot', 'trace', 'transpose', 'triangular_solve', 'solve', 
'tril', 'triu', 'trapz', 'true_divide', 'unique', 'unique_consecutive', 
'var_mean', 'var', ]
DO_NOW = ['add', 'cumsum', 'dist', 'einsum', 'eq', 'ne', 'equal', 'ge', 'gt', 'le', 'lt', 
'log', 'log10', 'log10_', 'log1p', 'log1p_', 'log2', 'log2_', 'log_', 'max', 'min', 
'prod', 'sign', 'sqrt', 'sqrt_', 'rsqrt', 'rsqrt_', ]

# Somewhat important and/or trickier functions
DO_SOON = ['allclose', 'all', 'any', 'argmax', 'argmin', 'argsort', 'as_tensor', 
'broadcast_tensors', 'cat', 'chain_matmul', 'cumprod', 'det', 'detach', 
'detach_', 'device', 'diag', 'diagonal', 'dtype', 'flatten', 'flip', 'floor_divide', 
'gather', 'logsumexp', 'matrix_power', 'mean', 'median', 'pow', 'reshape', 'resize_as_', 
'squeeze', 'unsqueeze', 'sort', 'split', ]

# Not important, could be tough
LATER = ['addbmm', 'addcdiv', 'addcmul', 'addmm', 'addmv', 'addmv_', 'addr', 
'baddbmm', 'cdist', 'cholesky', 'cholesky_inverse', 'cholesky_solve', 
'chunk', 'clone', 'combinations', 'cummax', 'cummin', 'diag_embed', 'diagflat', 
'full', 'full_like', 'narrow', 'orgqr', 'ormqr', 'roll', 'slogdet', 'where', ]

