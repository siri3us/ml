# -*- coding: utf-8 -*-
import random
import numpy as np
from ..sequential import Layer

class Criterion(Layer):
    def __init__ (self):
        self.output = None
        self.grad_input = None
    def forward(self, X_input, target):
        return self.update_output(X_input, target)
    def backward(self, X_input, target):
        return self.update_grad_input(X_input, target)
    def update_output(self, X_input, target):
        assert False
    def update_grad_input(self, X_input, target):
        assert False  
    def __repr__(self):
        return "Criterion"

class MulticlassLogLoss(Criterion):
    def __init__(self, n_classes):
        super().__init__()
        self.eps = 1e-30
        self.n_classes = n_classes
    def update_output(self, X_input, target): 
        if len(target.shape) > 1:
            target = np.argmax(target, axis=1)
        assert np.max(target) < self.n_classes
        assert np.min(target) >= 0
        X_input_clamp = np.clip(X_input, self.eps, 1 - self.eps) # Using this trick to avoid numerical errors
        self.output = -np.sum(np.log(X_input_clamp[np.arange(X_input.shape[0]), target]))
        return self.output
    def update_grad_input(self, X_input, target):
        if len(target.shape) == 1:
            target = np.eye(self.n_classes)[target]
        self.grad_input = -np.array(target).astype(np.float64)
        self.grad_input /= np.maximum(X_input, self.eps) # Using this trick to avoid numerical errors
        self.assert_inf(self.grad_input)
        self.assert_nans(self.grad_input)
        return self.grad_input
    def __repr__(self):
        return "MulticlassLogLoss"
