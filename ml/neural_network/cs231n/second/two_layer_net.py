# -*- coding: utf-8 -*-

import numpy as np
import numbers
from .layers import *

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=None, reg=0.0, random_state=None):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random initialization of the weights. 
        If None, smart initialization is used.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.random_state = random_state
        if random_state is not None:
            self.gen = np.random.RandomState(random_state)
        else:
            self.gen = np.random
        if weight_scale is None:
            stds = {'W1': 1.0 / np.sqrt(input_dim), 'W2': 1.0 / np.sqrt(hidden_dim)}
        elif isinstance(weight_scale, numbers.Number):
            stds = {'W1': weight_scale, 'W2': weight_scale}
        else:
            raise ValueError('Unacceptable value of weight_scale.')
        self.params['W1'] = stds['W1'] * self.gen.normal(0, 1, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = stds['W2'] * self.gen.normal(0, 1, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        
        W1 = self.params['W1']; b1 = self.params['b1']
        W2 = self.params['W2']; b2 = self.params['b2']
        X1_in = X.reshape((X.shape[0], -1))
        X1_out, cache1 = affine_relu_forward(X1_in, W1, b1)
        X2_out, cache2 = affine_forward(X1_out, W2, b2)
        scores = X2_out
        if y is None: # If y is None then we are in test mode so just return scores
            return scores
            
        loss, dX2_out     = softmax_loss(X2_out, y)
        dX1_out, dW2, db2 = affine_backward(dX2_out, cache2)
        dX1_in,  dW1, db1 = affine_relu_backward(dX1_out, cache1)
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))
        
        grads = {}
        grads['W1'] = dW1 + self.reg * W1; grads['b1'] = db1
        grads['W2'] = dW2 + self.reg * W2; grads['b2'] = db2
        return loss, grads
