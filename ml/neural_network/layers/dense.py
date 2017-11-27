# -*- coding: utf-8 -*-
import random
import numpy as np
from ..sequential import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size, l2_W_reg=0, bias=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.l2_W_reg = l2_W_reg
        self.bias = bias
        
        stdv = 1. / np.sqrt(input_size)
        self.W = np.random.uniform(-stdv, stdv, size=(input_size, output_size))
        self.grad_W = np.zeros_like(self.W)
        self.b = np.random.uniform(-stdv, stdv, size=output_size)
        self.grad_b = np.zeros_like(self.b)
            
    def update_output(self, X_input):
        self.assert_nans(X_input)
        self.output = np.dot(X_input, self.W)  # [B x I] x [I x O] = [B x O]
        if self.bias:
            self.output += self.b[None, :]
        return self.output
    
    def update_grad_input(self, X_input, grad_output):
        self.assert_nans(grad_output)
        self.grad_input = np.dot(grad_output, self.W.T)         # [B x O] x [O x I] = [B x I]
        return self.grad_input
    
    def update_grad_param(self, X_input, grad_output):
        self.assert_nans(grad_output)
        assert X_input.shape[0] == grad_output.shape[0]
        batch_size = X_input.shape[0]
        self.grad_W = np.dot(X_input.T, grad_output) / batch_size + self.l2_W_reg * self.W # ([I x B] x [B x O]).T = [I, O]
        if self.bias:
            self.grad_b = np.mean(grad_output, axis=0)
        
    def get_params(self):
        return [self.W, self.b]
    
    def get_grad_params(self):
        return [self.grad_W, self.grad_b]

    def zero_grad_params(self):
        self.grad_W.fill(0)
        self.grad_b.fill(0)
    
    def __repr__(self):
        return 'Dense({}->{})'.format(self.input_size, self.output_size)
