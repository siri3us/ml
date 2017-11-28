# -*- coding: utf-8 -*-
import numpy as np
from sklearn.utils import resample

class TwoLayerNet:
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, use_relu=False, std=None, random_state=None):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_relu = use_relu
        if std is None:
            self.stds = {'W1': 1.0 / np.sqrt(input_size), 'W2': 1.0 / np.sqrt(hidden_size)}
        else:
            self.stds = {'W1': std, 'W2': std}
            
        self.random_state = random_state
        if random_state is None:
            self.gen = np.random
        else:
            self.gen = np.random.RandomState(random_state)
        
        self.params = {}
        self.params['W1'] = self.stds['W1'] * self.gen.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros((1, hidden_size))
        self.params['W2'] = self.stds['W2'] * self.gen.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros((1, output_size))
        
        self.grads = {}
        self.grads['W1']  = np.zeros((input_size, hidden_size))
        self.grads['b1']  = np.zeros((1, hidden_size))
        self.grads['W2']  = np.zeros((hidden_size, output_size))
        self.grads['b2']  = np.zeros((1, output_size))
        
    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        C = W2.shape[1]

        # Compute the forward pass
        X_input1 = X
        X_output1 = np.dot(X_input1, W1) + b1
        
        if self.use_relu:
            X_output_relu = X_output1.copy()
            mask = X_output1 < 0      # [B, H]
            X_output_relu[mask] = 0  # [B, H]
        else:
            X_output_relu = X_output1
            
        X_output2 = np.dot(X_output_relu, W2) + b2
        scores = X_output2
        assert scores.shape == (N, C)
        
        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        scores -= np.max(scores, axis=1)[:, None]
        Z = np.log(np.sum(np.exp(scores), axis=1))
        scores -= Z[:, None]
        loss = -np.mean(scores[np.arange(X.shape[0]), y])
        loss += 0.5 * reg * (np.sum(W1**2) + np.sum(W2**2))
        probas = np.exp(scores)
        
        # Backward pass: compute gradients
        grads = {}
        grad_output2 = (probas - np.eye(C)[y]) / N                         # [B, C]
        grads['W2'] = np.dot(X_output_relu.T, grad_output2) + reg * W2     # ([H, B] x [B, C]) = [H, C]
        grads['b2'] = np.sum(grad_output2, axis=0, keepdims=True)         # [B, C] -> [1, C]
        # ReLU
        grad_output1 = np.dot(grad_output2, W2.T)                          # [B, C] x [C, H] x  = [B, H]
        if self.use_relu:
            grad_output1[mask] = 0.0                                       # ReLU
        grads['W1'] = np.dot(X_input1.T, grad_output1) + reg * W1      # ([H, B] x [B, D]).T = [D, H]
        grads['b1'] = np.sum(grad_output1, axis=1)[None, :]               # [H, B] -> [H,] -> [1, H]
        return loss, grads

    def train(self, X, y, X_val, y_val,
              momentum=0.0, learning_rate=1e-3, learning_rate_decay=1.0, reg=0.0, 
              n_epochs=100, batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        train_size = X.shape[0]
        iterations_per_epoch = max(int(train_size / batch_size + 0.5), 1)
        if verbose:
            print('n_epochs = {}, iterations per epoch = {}'.format(n_epochs, iterations_per_epoch))
        
        self.lr       = learning_rate
        self.lr_decay = learning_rate_decay
        self.momentum = momentum
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_epochs = n_epochs
        
        # Use SGD to optimize the parameters in self.model
        self.loss_history      = []
        self.train_acc_history = []
        self.val_acc_history   = []

        for n_epoch in range(n_epochs):
            self.n_epoch = n_epoch
            full_loss = 0 
            #if verbose:
             #   print('\tEpoch {} started.'.format(n_epoch))
            for it in range(iterations_per_epoch):
                self.n_iter = it
                random_state = n_epoch * iterations_per_epoch + it
                X_batch, y_batch = resample(X, y, n_samples=batch_size, random_state=random_state)
                
                # Compute loss and gradients using the current minibatch
                loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
                full_loss += loss
                self.loss_history.append(loss)
                self.update_parameters(grads)
        
            # Every epoch, check train and val accuracy and decay learning rate.
            full_loss /= iterations_per_epoch
            # Check accuracy
            train_acc = (self.predict(X) == y).mean()
            val_acc   = (self.predict(X_val) == y_val).mean()
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            # Decay learning rate
            self.lr *= self.lr_decay
            if verbose:
                print('\tEpoch {} ended: lr = {:.5f}, loss = {:.5f}, train acc = {:.4f}, val acc = {:.4f}'.format(
                    n_epoch, self.lr, full_loss, train_acc, val_acc))

        return {
          'loss_history':      self.loss_history,
          'train_acc_history': self.ltrain_acc_history,
          'val_acc_history':   self.lval_acc_history,
        }

    def update_parameters(self, grads):
        if (self.momentum > 0) & (self.n_iter > 0):
            for param_name in self.grads.keys():
                self.grads[param_name]  *= self.momentum
                self.grads[param_name]  += (1 - self.momentum) * grads[param_name]
                self.params[param_name] -= self.lr * self.grads[param_name]
        else:
            for param_name in self.grads.keys():
                self.grads[param_name] = grads[param_name]
                self.params[param_name] -= self.lr * self.grads[param_name]
    
    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = []
        for i in range(0, X.shape[0], self.batch_size):
            scores = self.loss(X[i:i + self.batch_size])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        return y_pred
