# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict
from ..sequential import Layer
from ..initializers import NormalInitializer, ZerosInitializer
from ..regularizers import EmptyRegularizer

class Dense(Layer):
    def __init__(self, units, use_bias=True, 
        W_initializer=None, b_initializer=None, W_regularizer=None):
        """
        Inputs:
        - units - Integer or Long, dimensionality of the output space.
        - W_initializer
        - b_initializer
        - seed - used for initializers!!!
        """
        super().__init__()
        self.units
        self.use_bias = use_bias
        self.seed = seed
        self.W_initializer = W_initializer
        self.b_initializer = b_initializer
        self.W_regularizer = W_regularizer
        self.b_regularizer = b_regularizer

    def _initialize(self, **kwargs):
        # Initializing weights
        assert 'input_shape' in kwargs, 'Input shape is not provided'
        assert 'seed' in kwargs, 'Seed must be provided even if not used'
        input_shape = kwargs['input_shape']
        seed = kwargs['seed']
        assert len(input_shape) == 2, 'Incorrect input shape'
        if self.W_initializer is None:
            self.W_initializer = NormalInitializer(seed)
            seed += 1
        if self.b_initializer is None:
            self.b_initializer = ZerosInitializer()
        self.W = self.W_initializer(shape=(input_shape[1], self.units))
        self.b = self.b_initializer(shape=(self.units,))
        kwargs['input_shape'] = (input_shape[0], self.units) # Input shape for the next layer
        kwargs['seed'] = seed # Seed for the next layer

        # Initializing zeros gradients
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        
        # Regularization
        if self.W_regularizer is None:
            self.W_regularizer = EmptyRegularizer()
        if self.b_regularizer is None:
            self.b_regularizer = EmptyRegularizer()
            
        # Initializing names
        names = kwargs['names']
        n_dense = names.setdefault('Dense', 0)
        self.name = 'Dense' + str(n_dense)
        names['Dense'] += 1
        
        return kwargs

    def update_output(self, input):
        self.assert_nans(input)
        self.output = np.dot(input, self.W)  # [B x I] x [I x O] = [B x O]
        if self.use_bias:
            self.output += self.b[None, :]
        return self.output
    
    def update_grad_input(self, input, grad_output):
        self.assert_nans(grad_output)
        self.grad_input = np.dot(grad_output, self.W.T)         # [B x O] x [O x I] = [B x I]
        return self.grad_input
    
    def update_grad_param(self, input, grad_output):
        self.assert_nans(grad_output)
        assert input.shape[0] == grad_output.shape[0]
        batch_size = input.shape[0]
        self.grad_W = np.dot(input.T, grad_output) / batch_size # ([I x B] x [B x O]).T = [I, O]
        if self.W_regularizer:
            self.grad_W += self.W_regularizer.grad(self.W)
        if self.use_bias:
            self.grad_b = np.mean(grad_output, axis=0)
            if self.b_regularizer:
                self.grad_b += self.b_regularizer.grad(self.b)
        
    def get_regularization_loss(self):
        loss = 0
        if self.W_regularizer:
            loss += self.W_regularizer.loss(self.W)
        if self.use_bias:
            if self.b_regularizer:
                loss += self.b_regularizer.loss(self.b)  
        return loss
 
    def get_params(self):
        return OrderedDict([(self.name + '/W', self.W), (self.name + '/b', self.b)])
    
    def get_grad_params(self):
        return OrderedDict([(self.name + '/W', self.W), (self.name + '/b', self.b)])

    def zero_grad_params(self):
        self.grad_W.fill(0)
        self.grad_b.fill(0)
    
    def __repr__(self):
        return 'Dense({}->{})'.format(self.input_size, self.output_size)
