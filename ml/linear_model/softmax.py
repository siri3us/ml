# -*- coding: utf-8 -*-
"""
CS231n http://cs231n.github.io/
"""

import numpy as np

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    n_classes = W.shape[1]
    train_size = X.shape[0]

    for n_sample in range(train_size):
        scores = np.dot(X[n_sample], W)
        scores -= np.max(scores)
        scores = np.exp(scores)
        p = scores / np.sum(scores)
        loss += -np.log(p[y[n_sample]])
        p[y[n_sample]] -= 1.0
        for i in range(n_classes):
            dW[:, i] += p[i] * X[n_sample]
    loss /= train_size 
    dW /= train_size
    loss += reg * np.sum(W**2)
    dW   += 2 * reg * W
    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    n_classes = W.shape[1]
    train_size = X.shape[0]
    
    scores = np.dot(X, W)
    scores = scores - np.max(scores, axis=1)[:, None]
    np.exp(scores, scores)
    P = scores / np.sum(scores, axis=1)[:, None]
    loss = -np.sum(np.log(P[np.arange(train_size), y])) / train_size + reg * np.sum(W**2)
    dW = np.dot(X.T, P - np.eye(n_classes)[y]) / train_size + reg * 2 * W
    return loss, dW
