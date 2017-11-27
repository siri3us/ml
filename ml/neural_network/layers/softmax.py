# -*- coding: utf-8 -*-
import random
import numpy as np
from ..sequential import Layer


class SoftMax(Layer):
    def __init__(self):
        super().__init__()
    
    def update_output(self, X_input):
        self.assert_nans(X_input)
        self.output = np.subtract(X_input, X_input.max(axis=1, keepdims=True))
        np.exp(self.output, self.output)
        self.output /= np.sum(self.output, axis=1, keepdims=True)
        return self.output
    
    def update_grad_input(self, X_input, grad_output):
        self.assert_nans(grad_output)
        G = np.multiply(self.output, grad_output)
        self.grad_input = G - self.output * np.sum(G, axis=1, keepdims=True)
        assert self.grad_input.shape == grad_output.shape
        return self.grad_input
    
    def __repr__(self):
        return "SoftMax"
