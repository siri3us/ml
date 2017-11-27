# -*- coding: utf-8 -*-
import random
import numpy as np
from numpy.random import RandomState
from ..sequential import Layer

class Dropout(Layer):
    def __init__(self, p=0.5, random_state=1):
        super().__init__()
        self.p = p
        self.mask = None
        self.random_state = random_state
        self.gen = RandomState(random_state)
        
    def update_output(self, X_input):
        batch_size, input_size = X_input.shape
        if self.training:
            self.mask = self.gen.choice([0, 1], p=[self.p, 1 - self.p], size=(batch_size, input_size))
            self.output = np.multiply(self.mask, X_input)
        else:
            self.output = (1 - self.p) * X_input
        return self.output
    
    def update_grad_input(self, X_input, grad_output):
        if self.training:
            self.grad_input = np.multiply(self.mask, grad_output)
        else:
            self.grad_input = (1 - self.p) * grad_output
        return self.grad_input
    
    def train(self):
        self.training = True
    def evaluate(self):
        self.training = False
        self.mask = None
        
    def __repr__(self):
        return "Dropout"
