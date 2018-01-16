# -*- coding: utf-8 -*-

import numpy as np

class Layer:
    def assert_nans(self, arr):
        assert not np.any(np.isnan(arr))
    def assert_inf(self, arr):
        assert not np.any(np.isinf(self.grad_input))
        
    def __init__(self):
        self.output = None
        self.grad_input = None
        self.training = True

    def forward(self, input):
        return self.update_output(input)
        
    def backward(self, input, grad_output):
        self.update_grad_input(input, grad_output) # This updates self.grad_input
        self.update_grad_param(input, grad_output)
        return self.grad_input
    
    def update_output(self, input):
        pass
        
    def update_grad_input(self, input, grad_input):
        pass
        
    def update_grad_param(self, input, grad_outut):
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
