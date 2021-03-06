# -*- coding: utf-8 -*-

import numpy as np
from ..sequential import Layer


class Dropout(Layer):
    def __init__(self, p=0.5, name=None):
        super().__init__(name=name)
        self.set_p(p)
        self.mask = None
    def __repr__(self):
        return super().__repr__() + '({})'.format(self.p)
    def set_p(self, p):
        self._check_proba(p, 'p')
        self.p = p

    ################################## 
    ###     Forward propagation    ###
    ##################################
    def _forward(self, input, target=None):
        if self.training:
            self.mask = self.generator.choice([0, 1], p=[self.p, 1 - self.p], size=input.shape)
            self.output = np.multiply(self.mask, input)
        else:
            self.output = (1 - self.p) * input
    
    ################################## 
    ###    Backward propagation    ###
    ##################################
    def _backward(self, input, grad_output):
        if self.training:
            self.grad_input = np.multiply(self.mask, grad_output)
        else:
            self.grad_input = (1 - self.p) * grad_output
