# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
from .layer import Layer
from .sequential import Sequential
from .criterions import Criterion
from .decorators import *


class Model(Layer):
    def __init__(self, sequential, criterion, name=None):
        super().__init__(name=name)
        if isinstance(sequential, list):
            sequential = Sequential(sequential)
        assert isinstance(sequential, Sequential)
        assert isinstance(criterion, Criterion)
        self.sequential = sequential
        self.criterion = criterion
    def __repr__(self):
        return str(self.sequential) + '->[' + str(self.criterion) + ']'
    def __getitem__(self, n_layer):
        return self.sequential[n_layer]
        
    ################################## 
    ###       Initialization       ###
    ##################################
    def initialize(self, config):
        """
        Initialization stage for all layers in the network:
            sets random seeds
            sets dtypes
            sets names
            runs parameters initialization
        """
        self._initialize(config)
        self.sequential.initialize(config)
        self.criterion.initialize(config)
        self.config = deepcopy(config)
        self.initialized = True
        return self
        
    ################################## 
    ###     Forward propagation    ###
    ##################################
    def _forward(self, input, target=None):
        self.sequential_output = self.sequential.forward(input)
        if target is None:
            self.output = self.sequential_output
        else:
            self.main_loss = self.criterion.forward(self.sequential_output, target)
            self.reg_loss  = self.get_regularization_loss()
            self.output    = self.main_loss + self.reg_loss

    ################################## 
    ###    Backward propagation    ###
    ##################################
    def _backward(self, input, target):
        grad_output = self.criterion.backward(self.sequential_output, target)
        self.grad_input = self.sequential.backward(input, grad_output)
        
    ################################## 
    ###         Parameters         ###
    ##################################
    # Получение параметров и градиентов
    @check_initialized
    def get_params(self, copy=False):
        return self.sequential.get_params(copy=copy)
    @check_initialized
    def get_grad_params(self, copy=False):
        return self.sequential.get_grad_params(copy=copy)
    # Выставление параметров и градиентов   
    @check_initialized
    def set_params(self, params):
        self.sequential.set_params(params)
    @check_initialized
    def set_grad_params(self, grad_params):
        self.sequential.set_grad_params(grad_params)
   
    ################################## 
    ###       Regularization       ###
    ##################################
    @check_initialized
    def get_regularization_loss(self):
        return self.sequential.get_regularization_loss()
        
    ################################## 
    ###   Changing operation mode  ###
    ##################################
    @check_initialized
    def train(self):
        self.training = True
        self.sequential.train()
    @check_initialized
    def evaluate(self):
        self.training = False
        self.sequential.evaluate()
