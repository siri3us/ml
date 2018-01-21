# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter, OrderedDict
from .layer import Layer
from .sequential import Sequential
from .model import Model
from .optimizers import *

class Solver:
    def __init__(self, model, data, model_config=None, optim_config=None, batch_size=100, n_epochs=10, 
                 verbose=False, print_every_iter=None, print_every_epoch=None, use_tqdm=False, update_history=True,
                 checkpoint_name=None, seed=0):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: An instance of class Model
        - data: A dictionary of training and validation data containing:
          'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
          'X_val':   Array, shape (N_val, d_1, ..., d_k) of validation images
          'y_train': Array, shape (N_train,) of labels for training images
          'y_val':   Array, shape (N_val,) of labels for validation images
        - optim_config:      A dictionary containing hyperparameters describing optimization process
           - 'update_rule'         : default 'sgd'
           - 'learning_rate'       : 
           - 'learning_rate_decay' : default 1.0
        - model_config:
            - seed           Used to initialize an internal random generator; default is 0.
            - dtype          np.float64
            - grad_clip      np.inf
        - batch_size:        Size of minibatches used to compute loss and gradient during training.
        - n_epochs:          The number of epochs to run for during training.
        - verbose:           Boolean; if set to False then no output will be printed during training; default is False.
        - print_every_iter:  Integer; training losses will be printed every print_every_iter iterations; default is 1000000000.
        - print_every_epoch: Integer; training losses will be printed every print_every_epoch epochs; default is 1000000000.
        - checkpoint_name:   If not None, then save model checkpoints here every epoch.
        - seed
        """
        
        # Unpack keyword arguments
        self.batch_size = batch_size
        self.num_epochs = n_epochs

        self.verbose = verbose
        self.use_tqdm = use_tqdm
        self.update_history = update_history
        if print_every_iter is None: 
            self.print_every_iter = 1000000000
        else: 
            self.print_every_iter = print_every_iter
        if print_every_epoch is None:
            self.print_every_epoch = 1000000000
        else:
            self.print_every_epoch = print_every_epoch
        self.checkpoint_name = checkpoint_name
        
        # Unpacking data
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val   = data['X_val']
        self.y_val   = data['y_val']
        self.gen = np.random.RandomState(seed)

        # Compiling model
        assert isinstance(model, Model)
        if model_config is None: 
            model_config = {}
        model_config.setdefault('dtype', np.float64)
        model_config.setdefault('seed', seed + 1)
        model_config.setdefault('input_shape', self.X_train.shape)
        self.model = model.compile(**model_config)
        if self.verbose: print(self.model)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if optim_config is None: optim_config = {}
        self.optim_config = optim_config
        self.update_rule  = self.optim_config.setdefault('update_rule', 'sgd')
        if self.update_rule   == 'sgd':          self.update_rule = sgd
        elif self.update_rule == 'sgd_momentum': self.update_rule = sgd_momentum
        elif self.update_rule == 'rmsprop':      self.update_rule = rmsprop
        elif self.update_rule == 'adam':         self.update_rule = adam
        else: assert False, 'Unknown update rule "{}"'.format(self.update_rule)
        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this manually.
        """
        # Set up some variables for book-keeping
        self.n_epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_loss_history = []
        self.train_acc_history  = []
        self.val_loss_history = []                  
        self.val_acc_history  = []
        self.history = {'loss_history':       self.loss_history,
                        'train_loss_history': self.train_loss_history,
                        'train_acc_history':  self.train_acc_history,
                        'val_loss_history':   self.val_loss_history,
                        'val_acc_history':    self.val_acc_history}
        
        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for param_name in self.model.get_params():
            self.optim_configs[param_name] = {k : v for k, v in self.optim_config.items()}

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
        loss = self.model.forward(X_batch, y_batch)
        input_grad = self.model.backward(X_batch, y_batch)
        self.loss_history.append(loss)
        
        # Perform a parameter update
        params = self.model.get_params()
        grad_params = self.model.get_grad_params()
        new_params = {}
        for param_name, param_value in params.items():
            grad_param_value = grad_params[param_name]
            config = self.optim_configs[param_name]
            new_params[param_name], self.optim_configs[param_name] =\
                self.update_rule(param_value, grad_param_value, config)
        self.model.set_params(new_params)
        
    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
          'model':              self.model,
          'optim_config':       self.optim_config,
          'batch_size':         self.batch_size,
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

    def eval(self, X, y, eval_func=None):
        """
        Evaluate the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.

        Returns:
        - metric_value: Scalar giving the the value of the required metric.
        """
        # Compute predictions in batches
        N = X.shape[0]
        num_batches = N // self.batch_size
        if N % self.batch_size != 0:
            num_batches += 1
        batch_outputs = []
        for i in range(num_batches):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            batch_output = self.model.forward(X[start:end])
            batch_outputs.append(batch_output)
        output = np.vstack(batch_outputs)
        assert output.ndim == 2
        return eval_func(output, y)

    def _logloss(self, probas, y_true):
        y_true = y_true.astype(np.int32, copy=False)
        n_samples = probas.shape[0]
        probas = np.clip(probas, 1e-18, 1 - 1e-18)
        return -np.mean(np.log(probas[np.arange(n_samples), y_true]))
        
    def _accuracy(self, scores, y_true):
        y_pred = np.argmax(scores, axis=1)
        return np.mean(y_pred == y_true)
    
    def _update_history(self):
        if not self.update_history:
            return
        train_acc  = self.eval(self.X_train, self.y_train, eval_func=self._accuracy)
        train_loss = self.eval(self.X_train, self.y_train, eval_func=self._logloss)
        val_acc    = self.eval(self.X_val, self.y_val, eval_func=self._accuracy)
        val_loss   = self.eval(self.X_val, self.y_val, eval_func=self._logloss)
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
            iter_range = range(num_iter_per_epoch)
            if self.use_tqdm:
                iter_range = tqdm(iter_range)
            for n_iter in iter_range:
                self._step()
                # Maybe print training loss
               
                if (n_all_iter + 1) % self.print_every_iter == 0:
                    msg = '(Iteration {}/{}) loss: {}'.format(n_all_iter + 1, num_all_iterations, 
                                                              self.loss_history[-1])
                    print(msg)
                n_all_iter += 1
            self._update_history()
            
            # Keep track of the best model
            if self.update_history:
                val_acc = self.val_acc_history[-1]
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params  = OrderedDict()
                    for param_name, param_value in self.model.get_params().items():
                        self.best_params[param_name] = param_value.copy()

            # Maybe print training loss
            if ((n_epoch + 1) % self.print_every_epoch == 0) & self.update_history:
                msg = '(Epoch {}/{}) train acc: {:.2}; val acc: {:.2}, train loss: {:.4}; val loss: {:.4}'.format(
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
        if self.update_history:
            self.model.set_params(self.best_params)
        return self.model
