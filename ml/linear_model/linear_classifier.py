# -*- coding: utf-8 -*-
"""
CS231n http://cs231n.github.io/
"""

import numpy as np
from .softmax import softmax_loss_vectorized
from .linear_svm import svm_loss_vectorized

class LinearClassifier(object):
    def __init__(self, random_state=1):
        self.random_state = random_state
        self.W = None

    def train(self, X, y, learning_rate=1e-3, learning_rate_decay=1.0, reg=1e-5, max_iters=1000, tol=1e-5,
              batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - max_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        train_size, dim = X.shape
        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            np.random.seed(self.random_state)
            self.W = 0.001 * np.random.randn(dim, num_classes)
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.reg = reg
        
        # Run stochastic gradient descent to optimize W
        prev_loss = np.inf
        loss_history = []
        for n_iter in range(max_iters):
            indices = np.random.choice(train_size, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W -= self.learning_rate * grad
            if verbose and (n_iter % 100 == 0):
                print(type(self).__name__ + '.train: iteration %d / %d: loss %f' % (n_iter, max_iters, loss))
            if abs(prev_loss - loss) < tol:
                if verbose:
                    print('Stopping criterion is satisfied.')
                break
            prev_loss = loss 
            self.learning_rate *= self.learning_rate_decay
        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        return np.argmax(np.dot(X, self.W), axis=1)


    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.  Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        assert False, '"loss" not defined'


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
