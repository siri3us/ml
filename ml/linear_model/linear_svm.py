# -*- coding: utf-8 -*-
"""
CS231n http://cs231n.github.io/
"""

import numpy as np

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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

    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    train_size  = X.shape[0]
    loss = 0.0
    for i in range(train_size):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, y[i]] -= X[i]
                dW[:, j]    += X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= train_size
    dW   /= train_size
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW   += 2 * reg * W
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    scores = np.dot(X, W)  # [N, C]
    indices = np.arange(X.shape[0])
    correct_class_scores = scores[indices, y]
    scores -= correct_class_scores[:, None] - 1.0
    scores[indices, y] = 0.0
    mask = (scores > 0)
    loss = np.sum(scores[mask]) / X.shape[0]
    loss += reg * np.sum(W**2)
    mask = mask.astype(np.int32)
    mask[indices, y] = -np.sum(mask, axis=1)
    dW = np.dot(X.T, mask) # [D, N] x [N, C] = [D, C]
    dW /= X.shape[0]
    dW += 2 * reg * W
    return loss, dW
