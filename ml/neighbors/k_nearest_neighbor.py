# -*- coding: utf-8 -*-
"""
CS231n http://cs231n.github.io/
"""

import numpy as np
from itertools import product

"""The implementation of KNN given below uses the following numpy functions:
* np.bincount
* np.argmax
* np.argpartition https://stackoverflow.com/questions/34226400/find-the-k-smallest-values-of-a-numpy-array

Additionally, KNN uses division of X_train into batches during "no loop" computations 
in order not to exceed memory limit provided to constructor in parameter "memory_threshold"."""

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self, copy=False, memory_threshold=512):
        self.copy = copy
        self._distance_calculators = {0: self.compute_distances_no_loops,
                                      1: self.compute_distances_one_loop,
                                      2: self.compute_distances_two_loops}
        self.memory_threshold=memory_threshold # [Mb]
      
    def fit(self, X, y):
        self.train(X, y)
        
    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
        consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
        """
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(X.shape) == 2
        assert len(y.shape) == 1
        assert X.shape[0] == y.shape[0]
        
        if self.copy:
            self.X_train = np.array(X)
            self.y_train = np.array(y)
        else:
            self.X_train = X
            self.y_train = y
        self.train_size, self.n_dim = self.X_train.shape
       
    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
        between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
        test data, where y[i] is the predicted label for the test point X[i].  
        """
        assert k > 0
        assert num_loops in self._distance_calculators
        dists = self.compute_distances(X, num_loops=num_loops)
        return self.predict_labels(dists, k=k)

    def compute_distances(self, X, num_loops=0):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the 
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
        is the Euclidean distance between the ith test point and the jth training
        point.
        """
        assert isinstance(X, np.ndarray)
        assert len(X.shape) == 2
        assert X.shape[1] == self.n_dim
        self.dists = self._distance_calculators[num_loops](X)
        assert self.dists.shape == (X.shape[0], self.train_size)
        return self.dists
    
    def compute_distances_two_loops(self, X):
        test_size = X.shape[0]; train_size = self.train_size
        dists = np.zeros((test_size, train_size))
        for i, j in product(range(test_size), range(train_size)):
            dists[i, j] = np.sum((X[i] - self.X_train[j])**2)
        return dists

    def compute_distances_one_loop(self, X):
        test_size = X.shape[0]; train_size = self.train_size
        dists = np.zeros((test_size, train_size))
        for i in range(test_size):
            dists[i, :] = np.sum((self.X_train - X[i][None, :])**2, axis=1)
        return dists

    def compute_distances_no_loops(self, X):
        batch_size = self._get_batch_size(X)
        assert X.shape[1] == self.n_dim
        test_size = X.shape[0]
        dists = []
        X = X[:, None, :] # (test_size x 1 x D)
        X2 = (X**2).sum(axis=2, keepdims=True) # (test_size x 1 x 1)
        assert X2.shape == (X2.shape[0], 1, 1)
        for i in range(0, self.train_size, batch_size):
            j = min(i + batch_size, self.train_size)
            X_train_batch = self.X_train[i:j][None, :, :] # (1 x batch_size x D)
            assert X_train_batch.shape == (1, j - i, self.n_dim)
            X_train_batch2 = (X_train_batch**2).sum(axis=2, keepdims=True) # (1 x train_size x 1)
            assert X_train_batch2.shape == (1, j - i, 1)
            ds = X2 + X_train_batch2 - 2 * np.sum(X * X_train_batch, axis=2, 
                                                keepdims=True)
            assert ds.shape == (test_size, j - i, 1)
            dists.append(ds[:,:,0])
        dists = np.hstack(dists)
        return dists
    
    def _get_batch_size(self, X):
        test_size = X.shape[0]
        temp = 1.0 * self.n_dim * test_size * X.dtype.type(0).nbytes / 2 ** 20
        batch_size = int(self.memory_threshold / temp)
        batch_size = max(1, min(self.train_size, batch_size))
        self.batch_size = batch_size
        return batch_size
    
    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
        gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
        test data, where y[i] is the predicted label for the test point X[i].  
        """
        test_size = dists.shape[0]
        y_pred = np.zeros(test_size)
        A = np.argpartition(dists, kth=k - 1, axis=1); assert A.shape == dists.shape
        labels = self.y_train[A[:, :k]]
        def find_most_common(arr):
            return np.argmax(np.bincount(arr))
        y_pred = np.apply_along_axis(find_most_common, 1, labels)
        assert y_pred.shape == (test_size,)
        return y_pred
