# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from ..layer import Layer
from ..initializers import *
from ..regularizers import *
from ..decorators import *

class Dense(Layer):
    def __init__(self, units, use_bias=True, W_init=None, b_init=None, W_reg=None, b_reg=None, name=None):
        """
        Inputs:
        - units - Integer or Long, dimensionality of the output space.
        - W_initializer
        - b_initializer
        - seed - used for initializers!!!
        """
        super().__init__(name=name)
        self.set_units(units)
        self.set_use_bias(use_bias)
        self.W_init = W_init
        self.b_init = b_init
        self.W_reg  = W_reg
        self.b_reg  = b_reg
    def set_units(self, units):
        self._check_int_positive(units, 'units', '"units" must be a positive integer.')
        self.units = units
    def set_use_bias(self, use_bias):
        self._check_boolean(use_bias, 'use_bias', '"use_bias" must be a boolean')
        self.use_bias = use_bias
    def __repr__(self):
        input_size = -1
        if self.initialized:
            input_size = self.W.shape[0]
        return 'Dense({}->{})'.format(input_size, self.units)   
   
    ################################## 
    ###       Initialization       ###
    ##################################
    def _initialize_input_shape(self, params):
        super()._initialize_input_shape(params)
        assert len(self.input_shape) == 2, 'input to Dense layer must be a 2-dim tensor.'
        return params
    def _initialize_params(self, params):
        self._initialize_W(params)
        self._initialize_b(params)
        return params
    def _initialize_W(self, params):
        n_features = self.input_shape[1]
        W_shape = (n_features, self.units)
        self.W_initializer = get_kernel_initializer(init=self.W_init, generator=self.generator, dtype=self.dtype)
        self.W = self.W_initializer(W_shape)
        self.grad_W = np.zeros_like(self.W, dtype=self.dtype)
        if self.W_reg is None:
            self.W_reg = EmptyRegularizer()
        assert isinstance(self.W_reg, Regularizer), 'W_reg of layer "{}" must have type Regularizer.'.format(self.name)
        return params
    def _initialize_b(self, params):
        self.b_initializer = get_bias_initializer(init=self.b_init, dtype=self.dtype)
        self.b = self.b_initializer((self.units,))
        self.grad_b = np.zeros_like(self.b, dtype=self.dtype)
        if self.b_reg is None:
            self.b_reg = EmptyRegularizer()
        assert isinstance(self.b_reg, Regularizer), 'b_reg of layer "{}" must have type Regularizer.'.format(self.name)
        return params    
    def _initialize_output_shape(self, params):
        self.output_shape = (-1, self.units) # Input shape for the next layer
        params['input_shape'] = self.output_shape
        return params
    
    ################################## 
    ###     Forward propagation    ###
    ##################################
    # Проверки прямого распространения
    def _forward_check_shape(self, input, target=None):
        assert input.ndim == 2, 'Input to layer "{}" must be 2-dim numpy array'.format(self.name)
        assert input.shape[1] == self.W.shape[0], 'Expected input shape (-1, {}) but received (-1, {})'.format(self.W.shape[0], input.shape[1])
    # Прямое распространение
    def _forward(self, input, target=None):
        self.output = np.dot(input, self.W)  # [N x D] x [D x H] = [N x H]
        if self.use_bias:
            self.output += self.b[None, :]
    
    ################################## 
    ###    Backward propagation    ###
    ##################################
    # Проверка ошибок обратного распространения
    def _backward_check_shape(self, input, grad_output):
        assert input.ndim == 2
        assert grad_output.ndim == 2
        assert input.shape[1] == self.W.shape[0]
        assert input.shape[0] == grad_output.shape[0]
        assert grad_output.shape[1] == self.W.shape[1]
    # Обратное распространение
    def _backward(self, input, grad_output):
        self.grad_input = np.dot(grad_output, self.W.T)          # [N x H] x [H x D] = [N x D]
        self.grad_W = np.dot(input.T, grad_output)               # ([D x N] x [N x H]).T = [D, H]
        self.W_reg.update_grad(self.W, self.grad_W)
        if self.use_bias:
            self.grad_b = np.sum(grad_output, axis=0)
            self.b_reg.update_grad(self.b, self.grad_b)

    ################################## 
    ###         Parameters         ###
    ##################################
    @check_initialized
    def get_params(self, copy=False):  
        params = OrderedDict()
        params[self.name + ':W'] = self.W 
        params[self.name + ':b'] = self.b
        return self._make_dict_copy(params, copy=copy)
    @check_initialized
    def get_grad_params(self, copy=False):
        grad_params = OrderedDict()
        grad_params[self.name + ':W'] = self.grad_W
        grad_params[self.name + ':b'] = self.grad_b
        return self._make_dict_copy(grad_params, copy=copy)
        
    ################################## 
    ###       Regularization       ###
    ##################################
    @check_initialized
    def get_regularizers(self):
        regularizers = OrderedDict()
        regularizers[self.name + ':W'] = self.W_reg
        regularizers[self.name + ':b'] = self.b_reg
        return regularizers
