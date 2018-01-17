# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from ..sequential import Layer
from ..decorators import *

class BatchNormalization(Layer):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var  = momentum * running_var  + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    def __init__(self, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        
    def _initialize(self, params):
        # Check params and initialize name
        params = super()._initialize(params)
        input_shape = params['input_shape']
        dtype = params['dtype']
        n_features = input_shape[1]
        self.running_mean = np.zeros(n_features, dtype=dtype)
        self.running_var = np.zeros(n_features, dtype=dtype)
        self.gamma = np.ones(n_features, dtype=dtype)
        self.beta = np.zeros(n_features, dtype=dtype)
        return params

    # Forward propagation
    def update_output(self, input):
        if self.training:
            self.sample_mean = np.mean(input, axis=0)
            self.sample_var = np.var(input, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.sample_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.sample_var
            self.normed_input = (input - self.sample_mean[None, :]) / np.sqrt(self.sample_var + self.eps)[None, :]
            self.output = self.gamma[None, :] * self.normed_input + self.beta[None, :]
        else:
            normed_input = (input - self.running_mean[None, :]) / (np.sqrt(self.running_var[None, :] + self.eps))
            self.output = self.gamma[None, :] * normed_input + self.beta[None, :]
        return self.output

    # Backward propagation
    def update_grad_input(self, input, grad_output):
        if self.training:
            var = self.sample_var
        else:
            var = self.running_var
        self.grad_input = self.gamma / np.sqrt(var + self.eps)[None, :] *\
                ((grad_output - np.mean(grad_output, axis=0)[None, :]) -\
                 self.normed_input * np.mean(np.multiply(self.normed_input, grad_output), axis=0)[None, :]) 
    def update_grad_param(self, input, grad_output):
        self.grad_gamma = np.sum(np.multiply(self.normed_input, grad_output), axis=0)
        self.grad_beta = np.sum(grad_output, axis=0)
          
    # Get params and grad_params
    @check_initialized
    def get_params(self, copy=False):
        if copy:
            return OrderedDict([(self.name + '/gamma', self.gamma.copy()), (self.name + '/beta', self.beta.copy())])
        return OrderedDict([(self.name + '/gamma', self.gamma), (self.name + '/beta', self.beta)])
    @check_initialized
    def get_grad_params(self, copy=False):
        if copy:
            return OrderedDict([(self.name + '/gamma', self.grad_gamma.copy()), (self.name + '/beta', self.grad_beta.copy())])
        return OrderedDict([(self.name + '/gamma', self.grad_gamma), (self.name + '/beta', self.grad_beta)])
    @check_initialized
    def zero_grad_params(self):
        self.grad_gamma = np.zeros_like(self.gamma).astype(self.dtype, copy=False)
        self.grad_beta = np.zeros_like(self.beta).astype(self.dtype, copy=False)
    
