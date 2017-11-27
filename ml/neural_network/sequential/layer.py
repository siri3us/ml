# -*- coding: utf-8 -*-

import numpy as np

class Layer:
    def assert_nans(self, arr):
        assert not np.any(np.isnan(arr))
    def assert_inf(self, arr):
        assert not np.max(self.grad_input) == np.inf
        
    def __init__(self):
        self.output = None
        self.grad_input = None
        self.training = True

    def forward(self, X_input):
        return self.update_output(X_input)
    def backward(self, X_input, grad_output):
        self.update_grad_input(X_input, grad_output)
        self.update_grad_param(X_input, grad_output)
        return self.grad_input
    
    def update_output(self, X_input):
        pass
    def update_grad_input(self, X_input, grad_input):
        pass
    def update_grad_param(self, X_input, grad_outut):
        pass
    def get_params(self):
        return []
    def get_grad_params(self):
        return []
    def zero_grad_params(self):
        pass
    def train(self):
        self.training = True
    def evaluate(self):
        self.training = False
