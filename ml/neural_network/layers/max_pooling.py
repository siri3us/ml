#-*- coding: utf-8 -*-

from itertools import product
from .convolution import ConvolutionKernel
from ml.neural_network.cs231n.second.conv_layers_fast import *

class MaxPooling(ConvolutionKernel):
    def __init__(self, kernel_size=(2, 2), stride=(2, 2), padding=None, method='fast', name=None):
        super().__init__(kernel_size=kernel_size, stride=stride, padding=padding, name=name)
        self.set_method(method)
        
    def set_method(self, method):
        assert method in ['naive', 'fast']
        self.method = method
        if self.method == 'naive':
            self.update_output = self._update_output_naive
            self.update_grad_input = self._update_grad_input_naive
        else:
            self.update_output = self._update_output_fast
            self.update_grad_input = self._update_grad_input_fast
            assert self.stride[0] == self.stride[1]
            self.pool_params = {'pool_height': self.kernel_size[0],
                                'pool_width': self.kernel_size[1],
                                'stride': self.stride[0]}
    
    def _update_output_naive(self, input):
        assert input.shape[1:] == self.input_shape[1:]
        n_samples, n_channels, input_h, output_w = input.shape
        _, n_channels, output_h, output_w = self.output_shape
        pool_h, pool_w = self.kernel_size
        stride_h, stride_w = self.stride
        self.output = np.zeros((n_samples, n_channels, output_h, output_w), dtype=self.dtype)
        self.mask = np.zeros((n_samples, n_channels, output_h, output_w, 2), dtype=np.int32)
        for n_sample, n_channel in product(range(n_samples), range(n_channels)):
            input_slice = np.pad(input[n_sample, n_channel], self.padding, mode='constant')
            output_slice = self.output[n_sample, n_channel]
            mask_slice = self.mask[n_sample, n_channel]
            for i, j in product(range(output_h), range(output_w)):
                x = input_slice[i * stride_h : i * stride_h + pool_h, j * stride_w : j * stride_w + pool_w]
                h, w = np.unravel_index(np.argmax(x), x.shape)
                output_slice[i, j] = x[h, w]
                mask_slice[i, j] = np.array([h + i * stride_h, w + j * stride_w])
    def _update_grad_input_naive(self, input, grad_output):
        assert input.shape[1:] == self.input_shape[1:]
        assert grad_output.shape[1:] == self.output_shape[1:]
        n_samples, n_channels, image_h, image_w = input.shape
        _, n_channels, output_h, output_w = self.output_shape
        pool_h, pool_w = self.kernel_size
        stride_h, stride_w = self.stride
        (pad_low, pad_upp), (pad_left, pad_right) = self.padding
        self.grad_input = np.zeros(input.shape, dtype=self.dtype)
        for n_sample, n_channel in product(range(n_samples), range(n_channels)):
            grad_input_slice = self.grad_input[n_sample, n_channel]
            grad_output_slice = grad_output[n_sample, n_channel]
            mask_slice = self.mask[n_sample, n_channel]
            for i, j in product(range(output_h), range(output_w)):
                h, w = mask_slice[i, j]
                grad_input_slice[h - pad_low, w - pad_left] = grad_output_slice[i, j]
            
    def _update_output_fast(self, input):
        if self.use_padding:
            pad_h, pad_w = self.padding
            input = np.pad(input, [(0, 0), (0, 0), pad_h, pad_w], mode='constant')
        self.output, self.cache = max_pool_forward_fast(input, self.pool_params)
    def _update_grad_input_fast(self, input, grad_output):
        self.grad_input = max_pool_backward_fast(grad_output, self.cache)
        if self.use_padding:
            (pad_low, pad_upp), (pad_left, pad_right) = self.padding
            _, _, padded_input_h, padded_input_w = self.grad_input.shape
            self.grad_input =\
                self.grad_input[:, :, pad_low:padded_input_h - pad_upp, pad_left:padded_input_w - pad_right]
