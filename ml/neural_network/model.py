# -*- coding: utf-8 -*-

import numpy as np
from .layer import Layer
from .sequential import Sequential
from .criterions import Criterion
from .decorators import *

class Model(Layer):
    def __init__(self, sequential, criterion):
        super().__init__()
        assert isinstance(sequential, Sequential)
        assert isinstance(criterion, Criterion)
        self.sequential = sequential
        self.criterion = criterion
    def __repr__(self):
        return str(self.sequential) + '->[' + str(self.criterion) + ']'
    
    # Initialization
    def initialize(self):
        assert False, '"initialize" method is not defined for Model'
        
    def compile(self, **config):
        """
        Compilation stage for all layers in the network:
            sets random seeds
            sets dtypes
            sets names
            runs parameters initialization
        """
        self.input_shape = config['input_shape']
        self.dtype = config.setdefault('dtype', np.float64)
        self.seed  = config.setdefault('seed', 0)
        self.names = config.setdefault('names', {})
        self.debug = config.setdefault('debug', False)
        self.grad_clip = config.setdefault('grad_clip', np.inf)
        self.config = config
        config = self.sequential.initialize(config)
        config = self.criterion.initialize(config)
        from copy import deepcopy
        self.config = deepcopy(config)
        self.compiled = True
        return self
    
    @check_compiled
    def forward(self, X, y=None):
        self.sequential_output = self.sequential.forward(X) # X will be automatically converted to dtype during this call
        if y is None:
            return self.sequential_output
        self.main_loss = self.criterion.forward(self.sequential_output, y)
        self.reg_loss  = self.sequential.get_regularization_loss()
        self.output = self.main_loss + self.reg_loss
        return self.output

    @check_compiled
    def backward(self, X, y):
        grad_output = self.criterion.backward(self.sequential_output, y)
        np.clip(grad_output, -self.grad_clip, self.grad_clip, grad_output)
        self.grad_input = self.sequential.backward(X, grad_output)
        np.clip(self.grad_input, -self.grad_clip, self.grad_clip, self.grad_input)
        return self.grad_input
        
    @check_compiled
    def get_params(self, copy=False):
        return self.sequential.get_params(copy=copy)
    @check_compiled
    def get_grad_params(self, copy=False):
        return self.sequential.get_grad_params(copy=copy)
    @check_compiled
    def set_params(self, params):
        self.sequential.set_params(params)
        
    @check_compiled
    def get_regularization_loss(self):
        return self.sequential.get_regularization_loss()
    
    @check_compiled
    def train(self):
        self.training = True
        self.sequential.train()
    @check_compiled
    def evaluate(self):
        self.training = False
        self.sequential.evaluate()
