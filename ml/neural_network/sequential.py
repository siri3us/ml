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
    def _forward(self, input):
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
            params.update(layer.get_params(copy=copy))
        return params
    @check_initialized
    def get_grad_params(self, copy=False):
        grad_params = OrderedDict()
        for layer in self.layers:
            grad_params.update(layer.get_grad_params(copy=copy))
        return grad_params
    # Выставление параметров и градиентов        
    @check_initialized
    def set_params(self, params):
        for layer in self.layers:
            layer.set_params(params)
    @check_initialized
    def set_grad_params(self, grad_params):
        for layer in self.layers:
            layer.set_grad_params(grad_params)
    @check_initialized
    def zero_grad_params(self):
        for layer in self.layers:
            layer.zero_grad_params()
            
    ################################## 
    ###       Regularization       ###
    ##################################
    @check_initialized
    def get_regularization_loss(self):
        loss = 0.0
        for layer in self.layers:
            loss += layer.get_regularization_loss()
        return loss
    @check_initialized
    def get_regularizers(self):
        regularizers = OrderedDict()
        for layer in self.layers:
            regularizers.update(layer.get_regularizers())
        return regularizers
        
    ################################## 
    ###   Changing operation mode  ###
    ##################################
    @check_initialized
    def train(self):
        """Sets all layers to training mode"""
        for layer in self.layers:
            layer.train()
    @check_initialized
    def evaluate(self):
        """Sets all layers to evaluation mode"""
        for layer in self.layers:
            layer.evaluate()
