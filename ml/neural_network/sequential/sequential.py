# -*- coding: utf-8 -*-

import numpy as np
from .layer import Layer

class Sequential(Layer):
    def __init__(self):
        super().__init__()
        self.layers = []
        
    def add(self, layer):
        assert isinstance(layer, Layer)
        self.layers.append(layer)
        
    def update_output(self, input):
        """This function passes input through all layers and saves output"""
        for n_layer, layer in enumerate(self.layers):
            output = layer.forward(input)
            input = output
        self.output = output
        return self.output
        
    def backward(self, input, grad_output):
        n_layers = len(self.layers)
        for n_layer in reversed(list(range(1, n_layers))):
            grad_output = self.layers[n_layer].backward(self.layers[n_layer - 1].output, grad_output)
        self.grad_input = self.layers[0].backward(input, grad_output)
        return self.grad_input
        
    def get_params(self):
        return [layer.get_params() for layer in self.layers]
        
    def get_grad_params(self):
        return [layer.get_grad_params() for layer in self.layers]
        
    def zero_grad_params(self):
        for layer in self.layers:
            layer.zero_grad_params()
            
    def __getitem__(self, n):
        return self.layers[n]
        
    def __repr__(self):
        return '->'.join([str(layer) for layer in self.layers])
        
    def train(self):
        """Sets all layers to training mode"""
        for layer in self.layers:
            layer.train()
            
    def evaluate(self):
        """Sets all layers to evaluation mode"""
        for layer in self.layers:
            layer.evaluate()
