# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from ..layer import Layer
from ..initializers import *
from ..regularizers import *
from ..decorators import *


class Dense(Layer):
    def __init__(self, units, use_bias=True, W_init=None, b_init=None, W_reg=None, b_reg=None):
        """
        Inputs:
        - units - Integer or Long, dimensionality of the output space.
        - W_initializer
        - b_initializer
        - seed - used for initializers!!!
        """
        super().__init__()
        self.units = units
        self.use_bias = use_bias
        self.W_init = W_init
        self.b_init = b_init
        self.W_reg = W_reg
        self.b_reg = b_reg
        
    def __repr__(self):
        if hasattr(self, 'W'):
            input_size, output_size = self.W.shape
        else:
            input_size = '?'
            output_size = self.units
        return 'Dense({}->{})'.format(input_size, output_size)   
    
    # Initialization
    def _initialize(self, params):
        # Params check and name initialization
        params = super()._initialize(params)
        # Initializing params and grads
        params = self._initialize_W(params)
        params = self._initialize_b(params)
        # Regularization
        if self.W_reg is None: self.W_reg = EmptyRegularizer()
        if self.b_reg is None: self.b_reg = EmptyRegularizer()
        return params
    
    def _initialize_W(self, params):
        input_shape, seed, dtype = params['input_shape'], params['seed'], params['dtype']
        if self.W_init is None:
            self.W_init= NormalInitializer(seed=seed)
        elif isinstance(self.W_init, np.ndarray):
            assert self.W_init.shape == (input_shape[1], self.units)
            self.W_init = DeterministicInitializer(self.W_init)
        self.W = self.W_init(shape=(input_shape[1], self.units), dtype=dtype)
        self.grad_W = np.zeros_like(self.W, dtype=dtype)
        params['seed'] = seed + 1
        params['input_shape'] = (input_shape[0], self.units) # Input shape for the next layer
        self.output_shape = params['input_shape']
        return params
        
    def _initialize_b(self, params):
        dtype = params['dtype']
        if self.b_init is None:
            self.b_init = ZerosInitializer()
        elif isinstance(self.b_init, np.ndarray):
            assert self.b_init.shape == (self.units,)
            self.b_init = DeterministicInitializer(self.b_init)
        self.b = self.b_init(shape=(self.units,), dtype=dtype)
        self.grad_b = np.zeros_like(self.b, dtype=dtype)
        return params
    
    # Forward propagation
    def update_output(self, input):
        self.output = np.dot(input, self.W)  # [B x I] x [I x O] = [B x O]
        if self.use_bias:
            self.output += self.b[None, :]
        return self.output
    
    # Backward propagation
    def update_grad_input(self, input, grad_output):
        self.grad_input = np.dot(grad_output, self.W.T)         # [N x H] x [H x D] = [N x D]
        return self.grad_input
    def update_grad_param(self, input, grad_output):
        self.grad_W = np.dot(input.T, grad_output)               # ([D x N] x [N x H]).T = [D, H]
        if self.W_reg: self.grad_W += self.W_reg.grad(self.W)
        if self.use_bias:
            self.grad_b = np.sum(grad_output, axis=0)
            if self.b_reg: self.grad_b += self.b_reg.grad(self.b)
        
        
    @check_initialized
    def get_regularization_loss(self):
        loss = 0.0
        if self.W_reg: loss += self.W_reg.loss(self.W)
        if self.use_bias:
            if self.b_reg: loss += self.b_reg.loss(self.b)  
        return loss
    
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
