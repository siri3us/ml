# -*- coding: utf-8 -*-

import numpy as np
from ..layer import Layer

class Flatten(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
    def _initialize_input_shape(self, params):
        input_shape = params['input_shape']
        assert len(input_shape) == 4
        _, n_channels, input_h, input_w = input_shape
        self.input_shape = (-1, n_channels, input_h, input_w)
        return params
    def _initialize_output_shape(self, params):
        input_shape = params['input_shape']
        _, n_channels, input_h, input_w = input_shape
        self.output_shape = (-1, n_channels * input_h * input_w)
        params['input_shape'] = self.output_shape
        return params
    
    def update_output(self, input):
        self.output = input.reshape(self.output_shape)
    
    def update_grad_input(self, input, grad_output):
        self.grad_input = grad_output.reshape(self.input_shape)
