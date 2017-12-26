# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict, Counter

class Solver:
    """
    A Solver encapsulates all the logic necessary for training classification models. The Solver performs 
    stochastic gradient descent using different update rules defined in optim.py.

    The solver accepts both training and validataion data and labels so it can periodically check
    classification accuracy on both training and validation data to watch out for overfitting.

    To train a model, you will first construct a Solver instance, passing the model, dataset, and various 
    options (learning rate, batch size, etc) to the constructor. You will then call the train() method to 
    run the optimization procedure and train the model.

    After the train() method returns, model.params will contain the parameters that performed best on the 
    validation set over the course of training. In addition, the instance variable solver.loss_history will contain 
    a list of all losses encountered during training and the instance variables solver.train_acc_history and 
    solver.val_acc_history will be lists of the accuracies of the model on the training and validation set at each epoch.

    Example usage might look something like this:

    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'y_val': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                      'learning_rate_decay': 0.95,
                    },
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()


    A Solver works on a model object that must conform to the following API:
    - model.params must be a dictionary mapping string parameter names to numpy
      arrays containing parameter values.
    - model.loss(X, y) must be a function that computes training-time loss and
      gradients, and test-time classification scores, with the following inputs
      and outputs:

      Inputs:
      - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
      - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
        label for X[i].

      Returns:
      If y is None, run a test-time forward pass and return:
      - scores: Array of shape (N, C) giving classification scores for X where
        scores[i, c] gives the score of class c for X[i].

      If y is not None, run a training time forward and backward pass and
      return a tuple of:
      - loss: Scalar giving the loss
      - grads: Dictionary with the same keys as self.params mapping parameter
        names to gradients of the loss with respect to those parameters.
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data containing:
          'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
          'X_val':   Array, shape (N_val, d_1, ..., d_k) of validation images
          'y_train': Array, shape (N_train,) of labels for training images
          'y_val':   Array, shape (N_val,) of labels for validation images

        Optional arguments:
        - update_rule:       A string giving the name of an update rule in optim.py. Default is 'sgd'.
        - optim_config:      A dictionary containing hyperparameters that will be passed to the chosen update rule.
            Each update rule requires different hyperparameters (see optim.py) but all update rules require a
           'learning_rate' parameter so that should always be present.
           'learning_rate_decay': A scalar for learning rate decay; after each epoch the learning rate is multiplied by this value.
        - batch_size:        Size of minibatches used to compute loss and gradient during training.
        - num_epochs:        The number of epochs to run for during training.
        - print_every_iter:  Integer; training losses will be printed every print_every_iter iterations; default is 1000000000.
        - print_every_epoch: Integer; training losses will be printed every print_every_epoch epochs; default is 1000000000.
        - verbose:           Boolean; if set to False then no output will be printed during training; default is False.
        - num_train_samples: Number of training samples used to check training accuracy; default is None, which uses the entire training set.
        - num_val_samples:   Number of validation samples to use to check val accuracy; default is None, which uses the entire validation set.
        - seed:              Used to initialize an internal random generator; default is 0.
        - checkpoint_name:   If not None, then save model checkpoints here every epoch.
        """
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val   = data['X_val']
        self.y_val   = data['y_val']

        # Unpack keyword arguments
        self.update_rule  = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.batch_size   = kwargs.pop('batch_size', 100)
        self.num_epochs   = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', None)
        self.num_val_samples   = kwargs.pop('num_val_samples', None)
        self.seed = kwargs.pop('seed', 0)
        self.gen  = np.random.RandomState(self.seed)
        
        self.checkpoint_name   = kwargs.pop('checkpoint_name', None)
        self.print_every_iter  = kwargs.pop('print_every_iter', 1000000000)
        self.print_every_epoch = kwargs.pop('print_every_epoch', 1000000000)
        self.verbose = kwargs.pop('verbose', False)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in sorted(list(kwargs.keys())))
            raise ValueError('Unrecognized arguments {}'.format(extra))

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        from . import optimizers
        if not hasattr(optimizers, self.update_rule):
            raise ValueError('Invalid update_rule "{}"'.format(self.update_rule))
        self.update_rule = getattr(optimizers, self.update_rule)
        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_loss_history = []
        self.train_acc_history  = []
        self.val_loss_history = []                  
        self.val_acc_history  = []
        self.history = {'loss_history': self.loss_history,
                        'train_loss_history': self.train_loss_history,
                        'train_acc_history': self.train_acc_history,
                        'val_loss_history': self.val_loss_history,
                        'val_acc_history': self.val_acc_history}
        
        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k : v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not be called manually.
        """
        # Make a minibatch of training data
        num_train  = self.X_train.shape[0]
        batch_mask = self.gen.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
          'model':              self.model,
          'update_rule':        self.update_rule,
          'optim_config':       self.optim_config,
          'batch_size':         self.batch_size,
          'num_train_samples':  self.num_train_samples,
          'num_val_samples':    self.num_val_samples,
          'n_epoch':            self.n_epoch,
          'loss_history':       self.loss_history,
          'train_loss_history': self.train_loss_history,
          'train_acc_history':  self.train_acc_history,
          'val_loss_history':   self.val_loss_history,
          'val_acc_history':    self.val_acc_history,
        }
        filename = '{}_epoch_{}.pkl'.format(self.checkpoint_name, int(self.epoch))
        if self.verbose:
            print('Saving checkpoint to "{}"'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

    def eval(self, X, y, num_samples=None, batch_size=100, eval_func=None):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # Maybe subsample the data
        N = X.shape[0]
        if (num_samples is not None) and (N > num_samples):
            mask = self.gen.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]
        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        scores = []
        for i in range(num_batches):
            start = i * batch_size
            end   = (i + 1) * batch_size
            batch_scores = self.model.loss(X[start:end])
            scores.append(batch_scores)
        scores = np.vstack(scores)
        return eval_func(scores, y)

    def _logloss(self, scores, y_true):
        n_samples = scores.shape[0]
        y_pred = scores - np.max(scores, axis=1, keepdims=True)
        y_pred = np.exp(y_pred)
        y_pred /= np.sum(y_pred, axis=1, keepdims=True)
        y_pred = np.clip(y_pred, 1e-18, 1 - 1e-18)
        return -np.mean(np.log(y_pred[np.arange(n_samples), y_true]))
        
    def _accuracy(self, scores, y_true):
        y_pred = np.argmax(scores, axis=1)
        #print(y_pred.shape, y_true.shape)
        return np.mean(y_pred == y_true)
    
    def _update_history(self):
        train_acc  = self.eval(self.X_train, self.y_train, num_samples=self.num_train_samples,
                              batch_size=self.batch_size,  eval_func=self._accuracy)
        train_loss = self.eval(self.X_train, self.y_train, num_samples = self.num_train_samples,
                              batch_size=self.batch_size,  eval_func=self._logloss)
        
        val_acc    = self.eval(self.X_val, self.y_val, num_samples=self.num_val_samples,
                              batch_size=self.batch_size,  eval_func=self._accuracy)
        val_loss   = self.eval(self.X_val, self.y_val, num_samples=self.num_val_samples,
                               batch_size=self.batch_size, eval_func=self._logloss)

        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
    
    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        num_iter_per_epoch = int(np.ceil(float(num_train) / self.batch_size))
        num_all_iterations = num_iter_per_epoch * self.num_epochs
        if self.verbose:
            print('num of epochs = {}\nnum of iterations = {}\niterations per epoch = {}'.format(
                self.num_epochs, num_all_iterations, num_iter_per_epoch))
        n_all_iter = 0
        
        self._update_history() # Initial model quality
        for n_epoch in range(self.num_epochs):
            self.n_epoch = n_epoch
            for n_iter in range(num_iter_per_epoch):
                self._step()
                # Maybe print training loss
                if self.verbose & ((n_all_iter + 1) % self.print_every_iter == 0):
                    msg = '(Iteration {}/{}) loss: {}'.format(n_all_iter + 1, num_all_iterations, 
                                                              self.loss_history[-1])
                    print(msg)
                n_all_iter += 1

            self._update_history()
            
            # Keep track of the best model
            val_acc = self.val_acc_history[-1]
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_params  = {}
                for k, v in self.model.params.items():
                    self.best_params[k] = v.copy()

            # Maybe print training loss
            if self.verbose & ((n_epoch + 1) % self.print_every_epoch == 0):
                msg = '(Epoch {}/{}) train acc: {:.2}; val acc: {:.2}, train loss: {:.5}; val loss: {:.5}'.format(
                    self.n_epoch + 1, self.num_epochs, 
                    self.train_acc_history[-1],  self.val_acc_history[-1],
                    self.train_loss_history[-1], self.val_loss_history[-1])
                print(msg)
            
            # Save the model at the end of every epoch
            self._save_checkpoint()
            
            # At the end of every epoch, increment the epoch counter and decay the learning rate.
            for k in self.optim_configs:
                optim_config = self.optim_configs[k]
                lr_decay = optim_config.get('learning_rate_decay', 1.0)
                optim_config['learning_rate'] *= lr_decay

        # At the end of training swap the best params into the model
        self.model.params = self.best_params
        return self.model
