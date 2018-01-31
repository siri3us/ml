# -*- coding: utf-8 -*-

import numpy as np
from .layer import Layer
from .decorators import *

class Criterion(Layer):
    def __init__ (self):
        super().__init__()
    def _initialize_output_shape(self, params):
        self.output_shape = (1, 1) # It is just a number
        return params
    def _forward_check_type(self, input, target):
        assert isinstance(input, np.ndarray)
        assert isinstance(target, np.ndarray)
    def _forward_check_shape(self, input, target):
        assert input.shape[0] == target.shape[0]

class MSECriterion(Criterion):
    def __init__(self, name=None):
        super().__init__(name=name)
    ################################## 
    ###     Forward propagation    ###
    ##################################
    def _forward(self, input, target):   
        self.output = np.mean((input - target) ** 2)
    ################################## 
    ###    Backward propagation    ###
    ##################################
    def _backward(self, input, target):
        self.grad_input = 2 * (input - target) / input.shape[0]
        
class MulticlassLogLoss(Criterion):
    def __init__(self, proba_clip=1e-20):
        super().__init__()
        self.proba_clip = proba_clip
        assert proba_clip <= 1e-6
        self._forward_preprocessors.append(self._forward_preprocess_input)
        self._forward_preprocessors.append(self._forward_preprocess_target)
        self._backward_preprocessors.append(self._backward_preprocess_input)
        self._backward_preprocessors.append(self._backward_preprocess_target)
    ################################## 
    ###       Initialization       ###
    ##################################
    def _initialize(self, params):
        super()._initialize(params)
        self.n_classes = self.input_shape[1]
        return params
        
    ################################## 
    ###     Forward propagation    ###
    ##################################
    # Проверки прямого распространения
    def _forward_check_shape(self, input, target):
        assert input.shape[0] == target.shape[0]
        assert input.ndim == 2
        assert target.ndim in [1, 2]
    def _forward_check_value(self, input, target):
        if target.ndim == 2:
            assert target.shape[1] == self.n_classes
        else:
            assert np.max(target) <= self.n_classes
            assert np.min(target) >= 0
    # Препроцессинг прямого распространения
    def _forward_preprocess_input(self, input, target):
        input = np.clip(input, self.proba_clip, 1 - self.proba_clip) # Using this trick to avoid numerical errors
        return input, target
    def _forward_preprocess_target(self, input, target):
        if target.ndim == 2:
            target = np.argmax(target, axis=1)
        else:
            target = target.astype(np.int32, copy=False)
        return input, target
    # Прямое распространение
    def _forward(self, input, target):
        self.output = -np.mean(np.log(input[np.arange(input.shape[0]), target]))
    
    ################################## 
    ###    Backward propagation    ###
    ##################################
    # Проверки обратного распространения
    def _backward_check_shape(self, input, target):
        self._forward_check_shape(input, target)
    def _backward_check_value(self, input, target):
        self._forward_check_value(input, target)
    # Препроцессинг обратного распространения
    def _backward_preprocess_input(self, input, target):
        return self._forward_preprocess_input(input, target)
    def _backward_preprocess_target(self, input, target):
        if target.ndim == 1:
            target = target.astype(np.int32, copy=False)
            target = np.eye(self.n_classes, dtype=self.dtype)[target]
        return input, target
    # Обратное распространение
    def _backward(self, input, target):
        self.grad_input = -np.array(target).astype(self.dtype, copy=False)
        self.grad_input /= input.shape[0] * np.maximum(input, self.proba_clip) # Using this trick to avoid numerical errors
