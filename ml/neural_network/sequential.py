# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from .layer import Layer
from .decorators import *

class Sequential(Layer):
    def __init__(self, layers=[]):
        super().__init__()
        self.layers = []
        for layer in layers:
            self.add(layer)
    def add(self, layer):
        assert isinstance(layer, Layer)
        self.layers.append(layer)
    def __repr__(self):
        return '->'.join([str(layer) for layer in self.layers])
 
    # Initialization
    def _initialize(self, params):
        params = super()._initialize(params)
        for layer in self.layers:
            params = layer.initialize(params)
        self.output_shape = self.layers[-1].output_shape
        return params

    # Forward propagation
    def update_output(self, input):
        """This function passes input through all layers and saves output"""
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        self.output = output
        return self.output
        
    # Backward propagation
    def backward(self, input, grad_output):
        """This function backpropagates though all layers"""
        n_layers = len(self.layers)
        for n_layer in reversed(list(range(1, n_layers))):
            grad_output = self.layers[n_layer].backward(self.layers[n_layer - 1].output, grad_output)
        self.grad_input = self.layers[0].backward(input, grad_output)
        return self.grad_input
        
    # Get params and their gradients
    @check_initialized
    def get_params(self, copy=False):
        params = OrderedDict()
        for layer in self.layers:
            for param_name, param_value in layer.get_params(copy=copy).items():
                params[param_name] = param_value
        return params
    @check_initialized
    def get_grad_params(self, copy=False):
        grad_params = OrderedDict()
        for layer in self.layers:
            for grad_name, grad_value in layer.get_grad_params(copy=copy).items():
                grad_params[grad_name] = grad_value
        return grad_params
    
    @check_initialized
    def zero_grad_params(self):
        for layer in self.layers:
            layer.zero_grad_params()
            
    # Regularization
    @check_initialized
    def get_regularization_loss(self):
        loss = 0.0
        for layer in self.layers:
            loss += layer.get_regularization_loss()
        return loss

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
