# -*- coding: utf-8 -*-

import numpy as np
from .layer import Layer
from .decorators import *

class Criterion(Layer):
    def __init__ (self, name=None):
        super().__init__(name=name)
    def _initialize_output_shape(self, params):
        self.output_shape = (1, 1) # It is just a number
        return params
        
    @check_initialized
    def forward(self, input, target):
        self._forward_enter_callback()
        input = input.astype(self.dtype, copy=False) # Decorator is not used here!
        self.update_output(input, target)
        self._forward_exit_callback()
        return self.output

    @check_initialized
    def backward(self, input, target):
        self._backward_enter_callback()
        input = input.astype(self.dtype, copy=False) # Decorator is not used here!
        self.update_grad_input(input, target)
        self._clip_gradients()
        self._backward_exit_callback()
        return self.grad_input

        
class MSECriterion(Criterion):
    def __init__(self, name=None):
        super().__init__(name=name)
        
    def update_output(self, input, target):   
        self.output = np.mean((input - target) ** 2)
        return self.output
 
    def update_grad_input(self, input, target):
        self.grad_input = 2 * (input - target) / input.shape[0]
        return self.grad_input
        
      
class MulticlassLogLoss(Criterion):
    def __init__(self, proba_clip=1e-20, name=None):
        super().__init__(name=name)
        self.proba_clip = proba_clip
        assert proba_clip <= 1e-6
        
    def _initialize(self, params):
        super()._initialize(params)
        self.n_classes = self.input_shape[1]
        return params
    
    def _check_input_target(self, input, target):
        if not self.debug: return
        assert input.ndim == 2
        assert target.ndim in [1, 2]
        assert input.shape[0] == target.shape[0]
        if target.ndim == 2: labels = np.argmax(target, axis=1)
        else: labels = target
        assert np.max(labels) < self.n_classes
        assert np.min(labels) >= 0
        
    def update_output(self, input, target):
        # Checks and conversions
        self._check_input_target(input, target)
        if target.ndim == 2:
            target = np.argmax(target, axis=1)
        else:
            target = target.astype(np.int32, copy=False)
        input_clamp = np.clip(input, self.proba_clip, 1 - self.proba_clip) # Using this trick to avoid numerical errors
        self.output = -np.mean(np.log(input_clamp[np.arange(input.shape[0]), target]))
        return self.output
    
    def update_grad_input(self, input, target):
        self._check_input_target(input, target)
        if target.ndim == 1:
            target = target.astype(np.int32, copy=False)
            target = np.eye(self.n_classes)[target]
        self.grad_input = -np.array(target).astype(self.dtype, copy=False)
        self.grad_input /= input.shape[0] * np.maximum(input, self.proba_clip) # Using this trick to avoid numerical errors
        return self.grad_input
