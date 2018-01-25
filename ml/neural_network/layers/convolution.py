import numbers
from scipy.signal import correlate2d
from itertools import product
from collections import OrderedDict

from ..layer import Layer
from ..decorators import *
from ..initializers import *
from .padding import *
from ml.neural_network.cs231n.second.conv_layers_fast import *

class ConvolutionKernel(Layer):
    def __init__(self, kernel_size=None, stride=None, padding=None, name=None):
        super().__init__(name=name)
        self.set_kernel_size(kernel_size)
        self.set_stride(stride)
        self.set_padding(padding)

    def __repr__(self):
        if not hasattr(self, 'input_shape'):
            input_shape = (-1, -1, -1, -1)
        else:
            input_shape = self.input_shape
        if not hasattr(self, 'output_shape'):
            output_shape = (-1, -1, -1, -1)
        else:
            output_shape = self.output_shape
        return type(self).__name__ + '({}->{})'.format(input_shape[1:], output_shape[1:])

    def set_kernel_size(self, value):
        self._set_2d_size(value, 'kernel_size')
    def set_stride(self, value):
        self._set_2d_size(value, 'stride')
    def _set_2d_size(self, size, name):
        self._check_type(size, name, int, tuple, list)
        if isinstance(size, int):
            self._check_positive(size, name)
            size_h = size_w = size
        else:
            assert len(size) == 2, 'When tuple or list "{}" must contain two positive integers'.format(name)
            size_h, size_w = size
            self._check_int_positive(size_h, 'size_h')
            self._check_int_positive(size_w, 'size_w')
        self.__setattr__(name, (size_h, size_w))
        
    def set_padding(self, padding):
        if padding is None:
            self.padding = padding
        else:
            self.padding = self._parse_padding(padding)       
    def _parse_padding(self, padding):
        if isinstance(padding, int):
            self._check_positive(padding, 'padding')
            return [(padding, padding), (padding, padding)]
        if isinstance(padding, tuple):
            assert len(padding) == 2, 'If tuple, "padding" must contain two values.'
            pad_h, pad_w = padding
            self._check_int_nonnegative(pad_h, 'pad_h')
            self._check_int_nonnegative(pad_w, 'pad_w')
            return [padding, padding]
        if isinstance(padding, list):    
            assert len(padding) == 2, 'If list, "padding" must contain two values.'
            pad_h = self._parse_padding(padding[0])[0]
            pad_w = self._parse_padding(padding[1])[0]
            return [pad_h, pad_w]
        assert False, '"padding" must have type int, tuple or list'
        
    # INITIALIZATION
    def _initialize_input_shape(self, params):
        assert 'input_shape' in params
        input_shape = params['input_shape']
        assert len(input_shape) == 4
        _, n_channels, input_h, input_w = input_shape
        self.input_shape = (-1, n_channels, input_h, input_w)
        return params
    def _initialize_output_shape(self, params):
        n_samples, n_channels, input_h, input_w = self.input_shape
        stride_h, stride_w = self.stride
        kernel_h, kernel_w = self.kernel_size
        if self.padding is None:
            self.padding, _ = find_2d_padding((input_h, input_w), self.kernel_size, self.stride)
        (pad_low, pad_upp), (pad_left, pad_right) = self.padding
        self.use_padding = False
        if pad_low + pad_upp + pad_left + pad_right > 0:
            self.use_padding = True
        assert (input_h + pad_upp + pad_low - kernel_h) % stride_h == 0, 'Incorrect height padding.'
        assert (input_w + pad_left + pad_right - kernel_w) % stride_w == 0, 'Incorrect width padding.'
        output_h = (input_h + pad_low + pad_upp - kernel_h) // stride_h + 1
        output_w = (input_w + pad_left + pad_right - kernel_w) // stride_w + 1 
        self.output_shape = (n_samples, n_channels, output_h, output_w)
        params['input_shape'] = self.output_shape
        return params
        

class Convolution(ConvolutionKernel):
    def __init__(self, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=None, use_bias=True,  
                 W_init=None, b_init=None, W_reg=None, b_reg=None, name=None, method='fast'):
        super().__init__(kernel_size=kernel_size, stride=stride, padding=padding, name=name)
        self.set_n_filters(n_filters)
        self.set_method(method)
        self.use_bias = use_bias
        self.W_init = W_init
        self.b_init = b_init
        self.W_reg = W_reg
        self.b_reg = b_reg
    def set_n_filters(self, n_filters):
        self._check_int_positive(n_filters, 'n_filters')
        self.n_filters = n_filters
    def set_method(self, method):
        if method == 'corr':
            self.update_output = self._update_output
            self.update_grad_input = self._update_grad_input
        elif method == 'naive':
            self.update_output = self._update_output_naive
            self.update_grad_input = self._update_grad_input_naive
        elif method == 'fast':
            assert self.stride[0] == self.stride[1], 'Required for method = "{}"'.format(method)
            self.conv_params = {'stride': self.stride[0], 'pad': 0}
            self.update_output = self._update_output_fast
            self.update_grad_input = self._update_grad_input_fast
        else:
            assert False, 'Unknown method "{}"'.format(method)
        self.method = method
    
    # INITIALIZE
    def _initialize_params(self, params):
        self._initialize_W(params)
        self._initialize_b(params)
        return params
    def _initialize_W(self, params):
        n_samples, n_channels, input_h, input_w = self.input_shape
        filter_h, filter_w = self.kernel_size
        W_shape = (self.n_filters, n_channels, filter_h, filter_w)
        self.W_initializer = get_kernel_initializer(init=self.W_init, generator=self.generator, dtype=self.dtype)
        self.W = self.W_initializer(W_shape)
        self.reversed_W = self.W[:, :, ::-1, ::-1]
        self.grad_W = np.zeros_like(self.W, dtype=self.dtype)
        return params
    def _initialize_b(self, params):
        self.b_initializer = get_bias_initializer(init=self.b_init, dtype=self.dtype)
        self.b = self.b_initializer((self.n_filters,))
        self.grad_b = np.zeros_like(self.b, dtype=self.dtype)
        return params

    def _initialize_output_shape(self, params):
        super()._initialize_output_shape(params)
        n_samples, n_channels, output_h, output_w = self.output_shape
        self.output_shape = (n_samples, self.n_filters, output_h, output_w)
        params['input_shape'] = self.output_shape
        return params
    
    # FORWARD PROPAGATION
    def _update_output(self, input):
        assert input.ndim == 4
        assert input.shape[1:] == self.input_shape[1:]
        n_samples, n_channels, input_h, input_w  = input.shape
        _, n_filters, output_h, output_w = self.output_shape
        stride_h, stride_w = self.stride
        W = self.W
        output = np.zeros((n_samples, n_filters, output_h, output_w), dtype=self.dtype)
        for n_sample, n_filter, n_channel in product(range(n_samples), range(n_filters), range(n_channels)):
            layer = np.pad(input[n_sample, n_channel], self.padding, mode='constant')
            c = correlate2d(layer, W[n_filter, n_channel], mode='valid')
            output[n_sample, n_filter] += c[::stride_h, ::stride_w]
        output += self.b[None, :, None, None]
        self.output = output
    def _update_output_naive(self, input):
        assert input.ndim == 4
        assert input.shape[1:] == self.input_shape[1:]
        n_samples, n_channels, input_h, input_w  = input.shape
        _, n_filters, output_h, output_w = self.output_shape
        filter_h, filter_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        padded_image_h = input_h + pad_h[0] + pad_h[1]
        padded_image_w = input_w + pad_w[0] + pad_w[1]
        
        output = np.zeros((n_samples, n_filters, output_h, output_w), dtype=self.dtype)
        W = self.W.reshape((n_filters, -1))
        pad = [(0, 0), pad_h, pad_w]
        for n_sample in range(n_samples):
            image = np.pad(input[n_sample], pad, mode='constant')
            new_i = 0
            for i in range(0, padded_image_h - filter_h + 1, stride_h):
                new_j = 0
                for j in range(0, padded_image_w - filter_w + 1, stride_w):
                    col = image[:, i:i + filter_h, j:j + filter_w].flatten()
                    output[n_sample, :, new_i, new_j] = np.sum(np.multiply(col[None, :], W), axis=1)
                    new_j += 1
                new_i += 1
        output += self.b[None, :, None, None]
        self.output = output

    # BACKWARD PROPAGATION
    def _update_grad_input(self, input, grad_output):
        n_samples, n_channels, input_h, input_w  = input.shape
        n_samples, n_filters, output_h, output_w = grad_output.shape
        assert input.shape[1:] == self.input_shape[1:], 'input shapes: {} != {}'.format(
            input.shape[1:], self.input_shape[1:])
        assert grad_output.shape[1:] == self.output_shape[1:], 'ouput_shapes: {} != {}'.format(
            grad_output.shape[1:], self.output_shape[1:])
        stride_h, stride_w = self.stride
        filter_h, filter_w = self.kernel_size
        (pad_upp, pad_low), (pad_left, pad_right) = self.padding
        padded_input_h = input_h + pad_upp + pad_low
        padded_input_w = input_w + pad_left + pad_right
        full_output_h = padded_input_h - filter_h + 1
        full_output_w = padded_input_w - filter_w + 1
        
        self.grad_input = np.zeros((n_samples, n_channels, input_h, input_w), dtype=self.dtype)
        self.zero_grad_params()
        grad_X = self.grad_input
        grad_W = self.grad_W
        grad_b = self.grad_b
        grad_Y = np.zeros((n_samples, n_filters, full_output_h, full_output_w), dtype=self.dtype)
        grad_Y[:, :, ::stride_h, ::stride_w] = grad_output
        
        W = self.reversed_W
        for n_sample in range(n_samples):
            x = np.pad(input[n_sample], [(0, 0), (pad_upp, pad_low), (pad_left, pad_right)], mode='constant')
            for n_filter in range(n_filters):
                for n_channel in range(n_channels):
                    grad_x = correlate2d(grad_Y[n_sample, n_filter], W[n_filter, n_channel], mode='full')
                    assert grad_x.shape == (padded_input_h, padded_input_w)
                    grad_X[n_sample, n_channel] += grad_x[pad_upp:padded_input_h-pad_low, pad_left:padded_input_w-pad_right]
                    grad_W[n_filter, n_channel] += correlate2d(x[n_channel], grad_Y[n_sample, n_filter], mode='valid')
        for n_filter in range(n_filters):
            grad_b[n_filter] += np.sum(grad_Y[:, n_filter, :, :])
        if self.W_reg:
            grad_W += self.W_reg.grad(self.W)
        if self.b_reg:
            grad_b += self.b_reg.grad(self.b)

    
    def _update_grad_input_naive(self, input, grad_output):
        n_samples, n_channels, input_h, input_w = input.shape
        n_samples, n_filters, output_h, output_w = grad_output.shape
        assert input.shape[1:] == self.input_shape[1:], 'input_shapes: {} != {}'.format(
            input.shape[1:], self.input_shape[1:])
        assert grad_output.shape[1:] == self.output_shape[1:], 'output_shapes: {} != {}'.format(
            grad_output.shape[1:], self.output_shape[1:])

        stride_h, stride_w = self.stride
        filter_h, filter_w = self.kernel_size
        (pad_upp, pad_low), (pad_left, pad_right) = self.padding
        
        # Finding X_grad
        self.grad_input = np.zeros((n_samples, n_channels, input_h, input_w), dtype=self.dtype)
        self.zero_grad_params()
        grad_Y = grad_output
        grad_X = self.grad_input
        grad_W = self.grad_W
        grad_b = self.grad_b
        W = self.W
        for n_sample, n_channel in product(range(n_samples), range(n_channels)):
            for h, w in product(range(input_h), range(input_w)):
                # Finding derivative over X[n_sample, n_channel, h, w]
                h += pad_upp; w += pad_left # Coordinates in the padded X
                i_low = int(np.ceil((h - filter_h + 1) / stride_h)); i_max = h // stride_h
                j_low = int(np.ceil((w - filter_w + 1) / stride_w)); j_max = w // stride_w
                i_range = range(max(0, i_low), min(output_h - 1, i_max) + 1)
                j_range = range(max(0, j_low), min(output_w - 1, j_max) + 1)
                for i, j in product(i_range, j_range):
                    a = h - i * stride_h
                    b = w - j * stride_w
                    assert (a >= 0) & (a < filter_h)
                    assert (b >= 0) & (b < filter_w)
                    assert (i >= 0) & (i < output_h)
                    assert (j >= 0) & (j < output_w)
                    grad_X[n_sample, n_channel, h - pad_upp, w - pad_left] +=\
                      np.sum(W[np.arange(n_filters), n_channel, a, b] * grad_Y[n_sample, np.arange(n_filters), i, j])
        
        X = np.pad(input, [(0, 0), (0, 0), (pad_upp, pad_low), (pad_left, pad_right)], mode='constant')
        assert X.shape == (n_samples, n_channels, pad_upp + input_h + pad_low, pad_left + input_w + pad_right)
        for n_filter in range(n_filters): 
            for n_channel in range(n_channels):
                for h, w in product(range(filter_h), range(filter_w)):
                    # Finding derivative over W[n_filter, n_channel, h, w]
                    for i, j in product(range(output_h), range(output_w)):
                        grad_W[n_filter, n_channel, h, w] +=\
                            np.sum(X[:, n_channel, i * stride_h + h, j * stride_w + w] * grad_Y[:, n_filter, i, j])
                        # Finding derivative over b[n_filter]
            grad_b[n_filter] += np.sum(grad_Y[:, n_filter, :, :])
        if self.W_reg:
            grad_W += self.W_reg.grad(self.W)
        if self.b_reg:
            grad_b += self.b_reg.grad(self.b)

    def _update_output_fast(self, input):
        if self.use_padding:
            pad_h, pad_w = self.padding
            input = np.pad(input, [(0, 0), (0, 0), pad_h, pad_w], mode='constant')
        self.output, self.cache = conv_forward_fast(input, self.W, self.b, self.conv_params)
    def _update_grad_input_fast(self, input, grad_output):
        self.grad_input, self.grad_W, self.grad_b = conv_backward_fast(grad_output, self.cache)
        if self.use_padding:
            (pad_low, pad_upp), (pad_left, pad_right) = self.padding
            _, _, padded_input_h, padded_input_w = self.grad_input.shape
            self.grad_input = self.grad_input[:, :, pad_low:padded_input_h - pad_upp, pad_left:padded_input_w - pad_right]
        if self.W_reg:
            self.grad_W += self.W_reg.grad(self.W)
        if self.b_reg:
            self.grad_b += self.b_reg.grad(self.b)
    
    @check_initialized
    def get_params(self, copy=False):
        if copy:
            return OrderedDict([(self.name + ':W', self.W.copy()), (self.name + ':b', self.b.copy())])
        return OrderedDict([(self.name + ':W', self.W), (self.name + ':b', self.b)])
        
    @check_initialized
    def get_grad_params(self, copy=False):
        if copy:
            return OrderedDict([(self.name + ':W', self.grad_W.copy()), (self.name + ':b', self.grad_b.copy())])
        return OrderedDict([(self.name + ':W', self.grad_W), (self.name + ':b', self.grad_b)])
    
    @check_initialized
    def zero_grad_params(self):
        self.grad_W.fill(0)
        self.grad_b.fill(0)
        
    @check_initialized
    def get_regularization_loss(self):
        loss = 0.0
        if self.W_reg: loss += self.W_reg.loss(self.W)
        if self.use_bias:
            if self.b_reg: loss += self.b_reg.loss(self.b)  
        return loss
