# -*- coding: utf-8 -*-


import numpy as np
from .layers import *

class FullyConnectedNet:
    """
    A fully-connected neural network with an arbitrary number of hidden layers, ReLU nonlinearities, 
    and a softmax loss function. This will also implement dropout and batch normalization as options.
    For a network with L layers, the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 use_relu=True, dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float64, seed=None, debug=False):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_relu = use_relu
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.dtype = dtype
        self.params = {}
        self.random = np.random.RandomState(seed)
   
        prev_output_dim = input_dim   # Размерность выхода предыдущего слоя
        hidden_dims += [num_classes]  # Добавляем последний affine-слой
        self.num_layers = len(hidden_dims) # L
        
        for n_layer in range(self.num_layers):
            input_dim = prev_output_dim
            output_dim = hidden_dims[n_layer]
            W = self.random.rand(input_dim, output_dim)
            b = np.zeros(output_dim)
            if weight_scale is None:
                W *= 1 / np.sqrt(input_dim)
            else:
                W *= weight_scale
            W_name = 'W' + str(n_layer); b_name = 'b' + str(n_layer)
            self.params[W_name] = W
            self.params[b_name] = b
            prev_output_dim = output_dim
            if debug:
                print('{}: {}'.format(W_name, W.shape))
                print('{}: {}'.format(b_name, b.shape))
            if use_batchnorm & (n_layer < self.num_layers - 1):
                self.params['gamma' + str(n_layer)] = np.ones(output_dim)
                self.params['beta' + str(n_layer)] = np.zeros(output_dim)
                
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout, 'gen': self.random}

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        self.caches = {}
        X_out = X
        for n_layer in range(self.num_layers - 1):
            caches = {}
            W = self.params['W' + str(n_layer)]
            b = self.params['b' + str(n_layer)]
            cache_name = 'aff'  + str(n_layer)
            X_out, caches[cache_name] = affine_forward(X_out, W, b)
            
            if self.use_batchnorm:
                gamma = self.params['gamma' + str(n_layer)]
                beta  = self.params['beta' + str(n_layer)]
                cache_name = 'bn' + str(n_layer)
                X_out, caches[cache_name] = batchnorm_forward(X_out, gamma, beta, self.bn_params[n_layer])
            
            if self.use_relu:
                cache_name = 'relu' + str(n_layer)
                X_out, caches[cache_name] = relu_forward(X_out)

            if self.use_dropout:
                cache_name = 'dropout' + str(n_layer)
                X_out, caches[cache_name] = dropout_forward(X_out, self.dropout_param)
                
            if mode == 'train':
                for k, v in caches.items():
                    self.caches[k] = v
                
        W, b = self.params['W' + str(self.num_layers - 1)], self.params['b' + str(self.num_layers - 1)]
        X_out, aff_cache = affine_forward(X_out, W, b)
        if mode == 'train':
            self.caches['aff'  + str(self.num_layers - 1)] = aff_cache   
        
        scores = X_out
        # If test mode return early
        if mode == 'test':
            return scores
        
        ############################################################################
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        grads = {}
        loss, output_grad = softmax_loss(scores, y)
        output_grad, dW, db = affine_backward(output_grad, self.caches['aff' + str(self.num_layers - 1)])
        W = self.params['W' + str(self.num_layers - 1)]
        grads['W' + str(self.num_layers - 1)] = dW + self.reg * W
        grads['b' + str(self.num_layers - 1)] = db
        loss += 0.5 * self.reg * np.sum(W ** 2)
        
        for n_layer in reversed(range(self.num_layers - 1)):
            if self.use_dropout:
                output_grad = dropout_backward(output_grad, self.caches['dropout' + str(n_layer)])
            if self.use_relu:
                output_grad = relu_backward(output_grad, self.caches['relu' + str(n_layer)])
            if self.use_batchnorm:
                output_grad, dgamma, dbeta = batchnorm_backward(output_grad, self.caches['bn' + str(n_layer)])
                grads['gamma' + str(n_layer)] = dgamma
                grads['beta' + str(n_layer)] = dbeta
            
            output_grad, dW, db = affine_backward(output_grad, self.caches['aff' + str(n_layer)])
            W = self.params['W' + str(n_layer)]
            grads['W' + str(n_layer)] = dW + self.reg * W
            grads['b' + str(n_layer)] = db
            loss += 0.5 * self.reg * np.sum(W ** 2)
        return loss, grads
