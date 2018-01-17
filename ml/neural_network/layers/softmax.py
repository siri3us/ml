# -*- coding: utf-8 -*-

import numpy as np
from ..sequential import Layer

class SoftMax(Layer):
    def __init__(self):
        super().__init__()
    
    def update_output(self, input):
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        np.exp(self.output, self.output)
        self.output /= np.sum(self.output, axis=1, keepdims=True)
        return self.output
    
    def update_grad_input(self, input, grad_output):
        G = np.multiply(self.output, grad_output)
        self.grad_input = G - self.output * np.sum(G, axis=1, keepdims=True)
        return self.grad_input
