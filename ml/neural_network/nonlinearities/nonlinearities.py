# -*- coding: utf-8 -*-
import random
import numpy as np
from ..sequential import Layer
from scipy.special import expit


class ReLU(Layer):
    def __init__(self):
         super().__init__()
    def update_output(self, X_input):
        self.output = np.maximum(X_input, 0)
        return self.output
    def update_grad_input(self, X_input, grad_output):
        self.grad_input = np.multiply(grad_output, X_input > 0)
        return self.grad_input
    def __repr__(self):
        return "ReLU"
    
    
class LeakyReLU(Layer):
    def __init__(self, slope=0.03):
        super().__init__()
        self.slope = slope
    def update_output(self, X_input):
        self.output = np.zeros_like(X_input.shape)
        self.mask = np.zeros_like(X_input)
        self.mask[X_input >= 0] = 1.0
        self.mask[X_input < 0] = self.slope
        self.output = np.multiply(self.mask, X_input)
        return self.output
    def update_grad_input(self, X_input, grad_output):
        assert X_input.shape[0] == grad_input.shape[0]
        self.grad_input = np.multiply(self.mask, grad_output)
        return self.grad_input
    def __repr__(self):
        return "LeakyReLU"
    
    
class ELU(Layer):
    def __init__(self, alpha=1.0):
        super().__init__()  
        self.alpha = alpha
    def update_output(self, X_input):
        self.output = np.zeros_like(X_input)
        mask = X_input >= 0
        self.output[mask] = X_input[mask]
        mask = np.invert(mask)
        self.output[mask] = self.alpha * (np.exp(X_input[mask]) - 1)
        return self.output
    def update_grad_input(self, X_input, grad_output):
        self.grad_input = np.zeros_like(grad_output)
        mask = X_input >= 0
        self.grad_input[mask] = grad_output[mask]
        mask = np.invert(mask)
        self.grad_input[mask] = self.alpha * np.exp(X_input[mask]) * grad_output[mask]
        return self.grad_input
    def __repr__(self):
        return "ELU"
    
    
class SoftPlus(Layer):
    def __init__(self):
        super().__init__()
    def update_output(self, X_input):
        self.output = X_input + np.log(1 + np.exp(-X_input))
        return self.output
    def update_grad_input(self, X_input, grad_output):
        self.grad_input = np.multiply(expit(X_input), grad_output)
        return self.grad_input
    def __repr__(self):
        return "SoftPlus"
