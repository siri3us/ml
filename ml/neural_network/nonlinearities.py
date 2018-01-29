# -*- coding: utf-8 -*-

import numpy as np
from .layer import Layer
from scipy.special import expit, logit

# TODO: возможно имеет смысл реализовать все activatoins через callable nonlinearities

class Nonlinearity:
    pass

class tanh(Nonlinearity):
    def forward(self, input):
        output = np.tanh(input)
        return output, output
    def backward(self, grad_output, cache):
        output = cache
        return np.multiply((1 - output ** 2), grad_output)

class relu(Nonlinearity):
    def forward(self, input):
        assert False
    def backward(self, grad_output, cache):
        assert False

class leaky_relu(Nonlinearity):
    def forward(self, input):
        assert False
    def backward(self, grad_output, cache):
        assert False
        
class elu(Nonlinearity):
    def forward(self, input):
        assert False
    def backward(self, grad_output, cache):
        assert False
        
class softplus(Nonlinearity):
    def forward(self, input):
        assert False
    def backward(self, grad_output, cache):
        assert False    


class Tanh(Layer):
    def __init__(self, name=None):
         super().__init__(name=name)
    
    def update_output(self, input):
        self.output = np.tanh(input)
        return self.output
    
    def update_grad_input(self, input, grad_output):
        self.grad_input = np.multiply((1 - self.output ** 2), grad_output)
        return self.grad_input
        

class ReLU(Layer):
    def __init__(self, name=None):
         super().__init__(name=name)
    
    def update_output(self, input):
        self.output = np.maximum(input, 0)
        return self.output
    
    def update_grad_input(self, input, grad_output):
        self.grad_input = np.multiply(grad_output, (input > 0).astype(self.dtype))
        return self.grad_input
        
        
class LeakyReLU(Layer):
    def __init__(self, slope=0.03, name=None):
        super().__init__(name=name)
        self.slope = slope
        
    def update_output(self, input):
        self.output = np.array(input)
        self.output[self.output < 0] *= self.slope
        return self.output
    
    def update_grad_input(self, input, grad_output):
        self.grad_input = np.array(grad_output)
        self.grad_input[input < 0] *= self.slope
        return self.grad_input

    
class ELU(Layer):
    def __init__(self, alpha=1.0, name=None):
        super().__init__(name=name)
        self.alpha = alpha
        
    def update_output(self, input):
        self.output = np.array(input)
        self.mask = self.output < 0
        self.output[self.mask] = self.alpha * (np.exp(self.output[self.mask]) - 1)
        return self.output
    
    def update_grad_input(self, input, grad_output):
        self.grad_input = np.array(grad_output)
        self.grad_input[self.mask] *= self.alpha * np.exp(input[self.mask])
        return self.grad_input
    

class SoftPlus(Layer):
    def __init__(self, limit=20, name=None):
        super().__init__(name=name)
        self.limit = limit
    
    def update_output(self, input):
        mask = input <= self.limit
        self.output = np.zeros_like(input, dtype=self.dtype)
        self.output[mask] = np.log(1 + np.exp(input[mask]))
        # These values are too high ot pass them to np.exp
        mask = np.logical_not(mask)
        self.output[mask] = input[mask]
        return self.output
    
    def update_grad_input(self, input, grad_output):
        self.grad_input = np.multiply(grad_output, expit(input))
        return self.grad_input
