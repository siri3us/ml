# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from ..core import Checker
from .decorators import *

class Layer(Checker):
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
    
    def __init__(self, name=None):
        self.output = None       # Output of the layer is always kept for backpropagatoin
        self.grad_input = None   # Input gradient is saved just in case
        self.training = True     # 
        
        self._forward_enter_callback  = lambda: None
        self._forward_exit_callback   = lambda: None
        self._backward_enter_callback = lambda: None
        self._backward_exit_callback  = lambda: None
        
        self.name = name
        #self.dtype = None         # Must be set during initialization
        #self.debug = None         # Must be set during initialization
        #self.input_shape = None   # Must be set during initialization
        #self.output_shape = None  # Must be set during initialization
        #self.grad_clip = None     # Must be set during initialization
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
    def _check_initialized(self):
        assert hasattr(self, 'debug')
        assert hasattr(self, 'seed')
        assert hasattr(self, 'dtype')
        assert hasattr(self, 'input_shape')
        assert hasattr(self, 'output_shape')
    def initialize(self, params):
        """This function is called during compilation process to initialize layer"""
        self._initialize(params)
        self._check_initialized()
        self.initialized = True
        return params
    def _initialize(self, params):
        """Must be called at each layer via super()._initialize(params) at the beginning of layer.initialize() call"""
        self._initialize_debug(params)
        self._initialize_seed(params)
        self._initialize_dtype(params)
        self._initialize_grad_clip(params)
        self._initialize_name(params)
        self._initialize_input_shape(params)
        self._initialize_params(params)
        self._initialize_output_shape(params)
        return params
    def _initialize_name(self, params):
        names = params.setdefault('names', {})
        layer_type_name = type(self).__name__
        n_layers = names.setdefault(layer_type_name, 0)
        names[layer_type_name] += 1
        if self.name is None:
            self.name = layer_type_name + str(n_layers)
        return params
    def _initialize_seed(self, params):
        # Предполагается, что каждый уровень сети снабжен собственным случайным генератором (Just in case)
        self.seed = params.setdefault('seed', 0)
        self.generator = np.random.RandomState(self.seed)
        params['seed'] += 1
        return params
    def _initialize_dtype(self, params):
        self.dtype = params.setdefault('dtype', np.float64)
        return params
    def _initialize_debug(self, params):
        self.debug = params.setdefault('debug', False)
        return params
    def _initialize_params(self, params):
        # Empty layer does not have params
        return params
    def _initialize_input_shape(self, params):
        assert 'input_shape' in params, '"input_shape" is not provided though must be.'
        input_shape = params['input_shape']
        self.input_shape = tuple([-1] + list(input_shape[1:]))
        return params
    def _initialize_output_shape(self, params):
        # По умолчанию предполагается, что размер выхода сети совпадает с его входом.
        # Для уровней, изменющих размер входа, следует перегрузить данную функцию.
        self.output_shape = params['input_shape']
        return params
    def _initialize_grad_clip(self, params):
        self.grad_clip = params.setdefault('grad_clip', np.inf)
        assert self.grad_clip > 0, '"grad_clip" must be higher than 0.'
        return params
        
    # Forward propagation
    @check_initialized
    @dtype_conversion
    def forward(self, input):
        self._forward_enter_callback()
        input = self._forward_preprocess(input)
        self.update_output(input)        # Finding output tensor; self.output
        self._forward_postprocess()
        self._forward_exit_callback()   # Callback during forward propagation
        return self.output
    def update_output(self, input):
        self.output = input
   
    def _forward_preprocess(self, input):
        self._check_arrays(input)        # Check
        input = self._convert_to_dtype(input)
        return input
    def _forward_postprocess(self):
        return
        
        
    # Backward propagation
    @check_initialized
    def backward(self, input, grad_output):
        self._backward_enter_callback()
        input, grad_output = self._backward_preprocess(input, grad_output) 
        self.update_grad_input(input, grad_output)     # This updates self.grad_input
        self.update_grad_param(input, grad_output)     # This updates all grad params
        self._backward_postprocess()
        self._backward_exit_callback()
        return self.grad_input
    def update_grad_input(self, input, grad_output):
        assert input.shape == grad_output.shape
        self.grad_input = grad_output
    def update_grad_param(self, input, grad_output):
        pass
        
    def _backward_preprocess(self, input, grad_output):
        self._check_arrays(input, grad_output)         # Checks and transformations
        input, grad_output = self._convert_to_dtype(input, grad_output)
        return input, grad_output
    def _backward_postprocess(self):
        self._clip_gradients()                         # Gradients clipping
        
    def _convert_to_dtype(self, *args):
        args = tuple([arg.astype(self.dtype, copy=False) for arg in args])
        if len(args) == 1:
            return args[0]
        return args
    def _clip_gradients(self):
        np.clip(self.grad_input, -self.grad_clip, self.grad_clip, self.grad_input)
        grad_params = self.get_grad_params()
        for param_name, param_value in grad_params.items():
            np.clip(param_value, -self.grad_clip, self.grad_clip, param_value)
    
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
            assert param_name in params, 'Layer "{}" does not have a parameter with name "{}".'.format(self, param_name)
            np.copyto(params[param_name], new_params[param_name]) 
    @check_initialized
    def set_grad_params(self, new_grad_params):
        grad_params = self.get_grad_params() # Getting references to current gradients
        for param_name in new_grad_params:
            assert param_name in grad_params, 'Layer "{}" does not have a parameter with name "{}".'.format(self, param_name)
            np.copyto(grad_params[param_name], new_grad_params[param_name])
            
    # Changing network mode
    @check_initialized
    def train(self):
        self.training = True
    @check_initialized
    def evaluate(self):
        self.training = False
