# -*- coding: utf-8 -*-
import numpy as np
from ..criterions import Criterion
from ...generators import BatchGenerator


def sgd_momentum(x, dx, config, state):
    """
    config:
        - momentum
        - learning_rate
    state:
        - old_grad
    """
    
    # x and dx have complex structure, old dx will be stored in a simpler one
    state.setdefault('old_grad', {})
    
    i = 0
    momentum = config['momentum']
    lr = config['learning_rate']
    for n_layer, (cur_layer_x, cur_layer_dx) in enumerate(zip(x, dx)):
        for n_param, (cur_x, cur_dx) in enumerate(zip(cur_layer_x, cur_layer_dx)):
            #print('n_layer={}, n_param={}, param.shape={}, param_grad.shape={}'.format(
            #        n_layer, n_param, cur_x.shape, cur_dx.shape))
            if i not in state['old_grad']:
                cur_old_grad = cur_dx
                state['old_grad'][i] = cur_old_grad
                #cur_old_grad = state['old_grad'].setdefault(i, np.zeros_like(cur_dx))
            else:
                cur_old_grad = state['old_grad'][i]     
                np.add(momentum * cur_old_grad, (1 - momentum) * cur_dx, out=cur_old_grad)
            
            cur_x -= lr * cur_old_grad
            i += 1


def _accuracy(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)
    assert len(y_pred.shape) == 1
    assert len(y_true.shape) == 1
    return np.mean(y_pred == y_true)
    
def _mlogloss(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    assert len(y_pred.shape) == 2
    if len(y_true.shape) == 1:
        n_classes = y_pred.shape[1]
        y_true = np.eye(n_classes)[y_true]
    y_pred = np.clip(y_pred, 1e-30, 1-1e-30)
    return -np.sum(np.multiply(np.log(y_pred), y_true)) / y_pred.shape[0]
     
_EVAL_FUNCS = {'mlogloss': _mlogloss, 
               'accuracy': _accuracy}
            
class NetworkTrainer:
    def __init__(self, net, criterion, learning_rate=1e-4, learning_rate_decay=1.0, momentum=0.9):
        assert isinstance(criterion, Criterion)
        self.net = net
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.loss_history = []
        self.evals_results = []
        self.optimizer_config = {'momentum': momentum}
        self.optimizer_state = {}
        
    def __call__(self, X_train, y_train, n_epochs, batch_size=256, 
                 eval_sets=(), eval_funcs=(), random_state=0, verbose=False):
        train_batch_gen = BatchGenerator(X_train, y_train, batch_size=batch_size, random_state=random_state)
        learning_rate = self.learning_rate
        for n_epoch in range(n_epochs):
            #self.net.zero_grad_params()
            n_samples_per_batch = 0
            loss = 0
            for X_batch, y_batch in train_batch_gen:
                n_samples_per_batch += X_batch.shape[0]
                predictions = self.net.forward(X_batch)
                loss += self.criterion.forward(predictions, y_batch)
                #print(X_batch.shape, y_batch.shape, loss)
                # Backward
                dp = self.criterion.backward(predictions, y_batch)
                self.net.backward(X_batch, dp)
                # Update weights
                self.optimizer_config['learning_rate'] = learning_rate
                sgd_momentum(self.net.get_params(),  self.net.get_grad_params(), 
                             self.optimizer_config,  self.optimizer_state)
            learning_rate *= self.learning_rate_decay
            loss /= n_samples_per_batch
            self.loss_history.append(loss)
            
            s = ['[train|loss={:.5f}]'.format(loss)]
            evals_results = [('train', 'loss', loss)]
            for eval_set_name, eval_set in eval_sets:
                X_test, y_test = eval_set
                y_pred = self.predict_probas(X_test)
                for eval_func_name in eval_funcs:
                    eval_func = _EVAL_FUNCS[eval_func_name]
                    eval_value = eval_func(y_pred, y_test)
                    result = (eval_set_name, eval_func_name, eval_value)
                    if verbose: 
                        s.append('[' + eval_set_name + '|' + eval_func_name + '={:.5f}]'.format(eval_value))
                    evals_results.append(result)
            if len(evals_results) > 0:
                self.evals_results.append(evals_results)
            if verbose & (len(s) > 0):
                print('n_epoch={}: {}'.format(n_epoch, ' '.join(s)))
        return self
    
    def predict(self, X, batch_size=256):
        gen = BatchGenerator(X, shuffle=False, batch_size=batch_size)
        y_preds = []
        self.net.evaluate()
        for X_batch in gen:
            y_preds.append(np.argmax(self.net.forward(X_batch), axis=1))
        self.net.train()
        return np.concatenate(y_preds)
    
    def predict_probas(self, X, batch_size=256):
        gen = BatchGenerator(X, shuffle=False, batch_size=batch_size)
        y_preds = []
        self.net.evaluate()
        for X_batch in gen:
            y_preds.append(self.net.forward(X_batch))
        self.net.train()
        return np.vstack(y_preds)
