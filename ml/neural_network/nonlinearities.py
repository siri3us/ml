# -*- coding: utf-8 -*-

import numpy as np
from .layer import Layer
from scipy.special import expit, logit

class Tanh(Layer):
    def __init__(self, name=None):
         super().__init__(name=name)
    def _forward(self, input, target=None):
        self.output = np.tanh(input)
    def _backward(self, input, grad_output):
        self.grad_input = np.multiply((1 - self.output ** 2), grad_output)


class ReLU(Layer):
    def __init__(self, name=None):
         super().__init__(name=name)
    def _forward(self, input, target=None):
        self.output = np.maximum(input, 0)
    def _backward(self, input, grad_output):
        self.grad_input = np.multiply(grad_output, (input > 0).astype(self.dtype))

        
class LeakyReLU(Layer):
    def __init__(self, slope=0.03, name=None):
        super().__init__(name=name)
        self.slope = slope
    def _forward(self, input, target=None):
        self.output = np.array(input)
        self.output[self.output < 0] *= self.slope
    def _backward(self, input, grad_output):
        self.grad_input = np.array(grad_output)
        self.grad_input[input < 0] *= self.slope

    
class ELU(Layer):
    def __init__(self, alpha=1.0, name=None):
        super().__init__(name=name)
        self.alpha = alpha
    def _forward(self, input, target=None):
        self.output = np.array(input)
        self.mask = self.output < 0
        self.output[self.mask] = self.alpha * (np.exp(self.output[self.mask]) - 1)
    def _backward(self, input, grad_output):
        self.grad_input = np.array(grad_output)
        self.grad_input[self.mask] *= self.alpha * np.exp(input[self.mask])
    

class SoftPlus(Layer):
    def __init__(self, limit=20, name=None):
        super().__init__(name=name)
        self.limit = limit
    def _forward(self, input, target=None):
        mask = input <= self.limit
        self.output = np.zeros_like(input, dtype=self.dtype)
        self.output[mask] = np.log(1 + np.exp(input[mask]))
        # These values are too high ot pass them to np.exp
        mask = np.logical_not(mask)
        self.output[mask] = input[mask]
    def _backward(self, input, grad_output):
        self.grad_input = np.multiply(grad_output, expit(input))
