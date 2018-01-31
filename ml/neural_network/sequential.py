# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from .layer import Layer
from .decorators import *

class Sequential(Layer):
    def __init__(self, layers=[], name=None):
        super().__init__(name=name)
        self.layers = []
        for layer in layers:
            self.add(layer)
    def add(self, layer):
        assert isinstance(layer, Layer)
        self.layers.append(layer)
    def __repr__(self):
        return '->'.join([str(layer) for layer in self.layers])
    def __getitem__(self, n_layer):
        return self.layers[n_layer]
 
    ################################## 
    ###       Initialization       ###
    ##################################
    def _initialize(self, params):
        super()._initialize(params)
        for layer in self.layers:
            params = layer.initialize(params)
        self.output_shape = params['input_shape']
        return params

    ################################## 
    ###     Forward propagation    ###
    ##################################
    def _forward(self, input, target=None):
        """This function passes input through all layers and saves output"""
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        self.output = output
        
    ################################## 
    ###    Backward propagation    ###
    ##################################
    def _backward(self, input, grad_output):
        """This function backpropagates though all layers"""
        n_layers = len(self.layers)
        for n_layer in reversed(list(range(1, n_layers))):
            grad_output = self.layers[n_layer].backward(self.layers[n_layer - 1].output, grad_output)
        self.grad_input = self.layers[0].backward(input, grad_output)
        
    ################################## 
    ###         Parameters         ###
    ##################################
    # Получение параметров и градиентов
    @check_initialized
    def get_params(self, copy=False):
        params = OrderedDict()
        for layer in self.layers:
            for param_name, param_value in layer.get_params(copy=copy).items():
                assert param_name not in params, 'Parameters name clash!'
                params[param_name] = param_value
        return params
    @check_initialized
    def get_grad_params(self, copy=False):
        grad_params = OrderedDict()
        for layer in self.layers:
            for param_name, grad_param_value in layer.get_grad_params(copy=copy).items():
                assert param_name not in grad_params, 'Parameters name clash!'
                grad_params[param_name] = grad_param_value
        return grad_params

    ################################## 
    ###       Regularization       ###
    ##################################
    @check_initialized
    def get_regularizers(self):
        regularizers = OrderedDict()
        n_regularizers = 0
        for layer in self.layers:
            for param_name, reg in layer.get_regularizers().items():
                assert param_name not in regularizers, 'Parameters name clash!'
                regularizers[param_name] = reg
        return regularizers
        
    ################################## 
    ###   Changing operation mode  ###
    ##################################
    @check_initialized
    def train(self):
        """Sets all layers to training mode"""
        self.training = True
        for layer in self.layers:
            layer.train()
    @check_initialized
    def evaluate(self):
        """Sets all layers to evaluation mode"""
        self.training = False
        for layer in self.layers:
            layer.evaluate()
