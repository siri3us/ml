# -*- coding: utf-8 -*-

import numpy as np
from ..sequential import Layer


class Dropout(Layer):
    def __init__(self, p=0.5, name=None):
        super().__init__(name=name)
        self.p = p
        self.mask = None
    def __repr__(self):
        return super().__repr__() + '({})'.format(self.p)
        
    # initialization
    def _initialize_seed(self, params):
        if self.debug: print('setting: ', params)
        self.seed = params.setdefault('seed', 0)
        self.generator = np.random.RandomState(self.seed)
        params['seed'] += 1
        return params
    
    # Setters
    def set_p(self, p):
        self.p = p
    
    # Forward propagation
    def update_output(self, input):
        if self.training:
            self.mask = self.generator.choice([0, 1], p=[self.p, 1 - self.p], size=input.shape)
            self.output = np.multiply(self.mask, input)
        else:
            self.output = (1 - self.p) * input
        return self.output
    
    # Backward propagation
    def update_grad_input(self, input, grad_output):
        if self.training:
            self.grad_input = np.multiply(self.mask, grad_output)
        else:
            self.grad_input = (1 - self.p) * grad_output
        return self.grad_input
