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
    def __init__(self, momentum=0.9, eps=1e-5, name=None):
        super().__init__(name=name)
        self.momentum = momentum
        self.eps = eps
        self._is_NCHW_input = False
        # Препроцессинг прямого распространения
        self._forward_preprocessors.append(self._forward_preprocess_shape)
        # Постпроцессинг прямого распространения
        self._forward_postprocessors.append(self._forward_postprocess_shape)
        # Препроцессинг обратного распространения
        self._backward_preprocessors.append(self._backward_preprocess_shape)
        # Постпроцессинг обратного распространения
        self._backward_postprocessors.append(self._backward_postprocess_shape)
        
    ################################## 
    ###       Initialization       ###
    ##################################
    def _initialize_params(self, params):
        self.n_features = self.input_shape[1]
        self.running_mean = np.zeros(self.n_features, dtype=self.dtype)
        self.running_var = np.zeros(self.n_features, dtype=self.dtype)
        self.gamma = np.ones(self.n_features, dtype=self.dtype)
        self.beta = np.zeros(self.n_features, dtype=self.dtype)
        self.grad_gamma = np.zeros_like(self.gamma, dtype=self.dtype)
        self.grad_beta = np.zeros_like(self.beta, dtype=self.dtype)
        return params
        
    def _NCHW_to_ND(self, X):
        X = np.transpose(X, [0, 2, 3, 1])
        self._NHWC_shape = X.shape
        return X.reshape((-1, X.shape[3]))
    def _ND_to_NCHW(self, X):
        return X.reshape(self._NHWC_shape).transpose(0, 3, 1, 2)
        
        
    ################################## 
    ###     Forward propagation    ###
    ##################################
    def _forward_preprocess_shape(self, input, target=None):
        self._is_NCHW_input = False
        if input.ndim == 4:
            self._is_NCHW_input = True
            input = self._NCHW_to_ND(input)
        return input, target
    # Прямое распространение
    def _forward(self, input, target=None):
        if self.training:
            self.sample_mean = np.mean(input, axis=0)
            self.sample_var  = np.var(input, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.sample_mean
            self.running_var  = self.momentum * self.running_var  + (1 - self.momentum) * self.sample_var
            self.normed_input = (input - self.sample_mean[None, :]) / np.sqrt(self.sample_var + self.eps)[None, :]
            self.output = self.gamma[None, :] * self.normed_input + self.beta[None, :]
        else:
            normed_input = (input - self.running_mean[None, :]) / (np.sqrt(self.running_var[None, :] + self.eps))
            self.output = self.gamma[None, :] * normed_input + self.beta[None, :]
    def _forward_postprocess_shape(self):
        if self._is_NCHW_input:
            self.output = self._ND_to_NCHW(self.output)
            
    ################################## 
    ###    Backward propagation    ###
    ##################################
    def _backward_preprocess_shape(self, input, grad_output):
        if self._is_NCHW_input:
            input = self._NCHW_to_ND(input)
            grad_output = self._NCHW_to_ND(grad_output)
        return input, grad_output
    # Прямое распространение
    def _backward(self, input, grad_output):
        if self.training:
            var = self.sample_var
        else:
            var = self.running_var
        self.grad_input = self.gamma / np.sqrt(var + self.eps)[None, :] *\
                ((grad_output - np.mean(grad_output, axis=0)[None, :]) -\
                 self.normed_input * np.mean(np.multiply(self.normed_input, grad_output), axis=0)[None, :]) 
        self.grad_gamma = np.sum(np.multiply(self.normed_input, grad_output), axis=0)
        self.grad_beta = np.sum(grad_output, axis=0)
    def _backward_postprocess_shape(self):
        if self._is_NCHW_input:
            self.grad_input = self._ND_to_NCHW(self.grad_input) 
            
    # Get params and grad_params
    @check_initialized
    def get_params(self, copy=False):
        params = OrderedDict()
        params[self.name + ':gamma'] = self.gamma
        params[self.name + ':beta'] = self.beta
        return self._make_dict_copy(params, copy=copy)
    @check_initialized
    def get_grad_params(self, copy=False):
        grad_params = OrderedDict()
        grad_params[self.name + ':gamma'] = self.grad_gamma
        grad_params[self.name + ':beta'] = self.grad_beta
        return self._make_dict_copy(grad_params, copy=copy) 
