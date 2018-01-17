# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from .decorators import *

class Layer:
    # Checks
    def _assert_nans(self, arr):
        assert not np.any(np.isnan(arr)), 'NaNs etected: {}!'.format(self)
    def _assert_infs(self, arr):
        assert not np.any(np.isinf(arr)), 'Infs detected {}!'.format(self)
    def _check_arrays(self, *arrays):
        if not self.debug: return
        for arr in arrays:
            self._assert_nans(arr)
            self._assert_infs(arr)
    
    def __init__(self):
        self.output = None       # Output of the layer is always kept for backpropagatoin
        self.grad_input = None   # Input gradient is saved just in case
        self.training = True     # 
        
        self._forward_enter_callback  = lambda: None
        self._forward_exit_callback   = lambda: None
        self._backward_enter_callback = lambda: None
        self._backward_exit_callback  = lambda: None
        
        self.dtype = None         # Must be set during initialization
        self.debug = False        # Must be set during initialization
        self.initialized = False  # Must be set to True after initialization
        
    def __repr__(self):
        return type(self).__name__
    
    # Setting callbacks
    def set_forward_enter_call(self, callback=lambda: None):
        self._forward_enter_callback = callback
    def set_forward_exit_call(self, callback=lambda: None):
        self._forward_exit_callback = callback
    def set_backward_enter_call(self, callback=lambda: None):
        self._backward_enter_callback = callback
    def set_backward_exit_call(self, callback=lambda: None):
        self._backward_exit_callback = callback
    
    # Initialization
    def initialize(self, params):
        """This function is called during compilation process to initialize layer"""
        params = self._initialize(params)
        self.initialized = True
        return params
    def _initialize(self, params):
        """Must be called at each layer via super()._initialize(params) at the beginning of layer.initialize() call"""
        params = self._check_initialization_params(params)
        params = self._initialize_name(params)
        self.input_shape = params['input_shape']
        self.output_shape = self.input_shape # Default behavior
        self.seed        = params['seed']
        self.debug       = params['debug']
        self.dtype       = params['dtype']
        self.grad_clip   = params['grad_clip']
        return params
    def _check_initialization_params(self, params):
        assert 'input_shape' in params, 'Input shape must be provided.' # This is probably not critical
        params.setdefault('seed', 0)
        params.setdefault('names', {})
        params.setdefault('debug', False)
        params.setdefault('dtype', np.float64)
        params.setdefault('grad_clip', np.inf)
        return params
    def _initialize_name(self, params):
        names = params['names']
        layer_type_name = type(self).__name__
        n_layers = names.setdefault(layer_type_name, 0)
        self.name = layer_type_name + str(n_layers)
        names[layer_type_name] += 1
        return params

    # Forward propagation
    @check_initialized
    @dtype_conversion
    def forward(self, input):
        self._forward_enter_callback()
        self._check_arrays(input)        # Check
        self.update_output(input)        # Finding output tensor; self.output
        self._forward_exit_callback()    # Callback during forward propagation
        return self.output
    def update_output(self, input):
        self.output = input

    # Backward propagation
    @check_initialized
    @dtype_conversion
    def backward(self, input, grad_output):
        self._backward_enter_callback()
        self._check_arrays(input, grad_output)     # Checks and transformations
        self.update_grad_input(input, grad_output) # This updates self.grad_input
        self.update_grad_param(input, grad_output) # This updates all grad params
        self._clip_gradients()
        self._backward_exit_callback()
        return self.grad_input
    def update_grad_input(self, input, grad_output):
        assert input.shape == grad_output.shape
        self.grad_input = grad_output

    def update_grad_param(self, input, grad_output):
        pass
    def _clip_gradients(self):
        np.clip(self.grad_input, -self.grad_clip, self.grad_clip, self.grad_input)
        grad_params = self.get_grad_params()
        for param_name, param_value in grad_params.items():
            np.clip(param_value,-self.grad_clip, self.grad_clip, param_value)
    
    # Regulariation
    @check_initialized
    def get_regularization_loss(self):
        return 0.0
    
    # Getting params and gradients
    @check_initialized
    def get_params(self, copy=False):
        return OrderedDict()
    @check_initialized
    def get_grad_params(self, copy=False):
        return OrderedDict()
    @check_initialized
    def zero_grad_params(self):
        pass
    @check_initialized
    def set_params(self, new_params):
        params = self.get_params()
        for param_name in new_params:
            np.copyto(params[param_name], new_params[param_name]) 

    # Changing network mode
    @check_initialized
    def train(self):
        self.training = True
    @check_initialized
    def evaluate(self):
        self.training = False
