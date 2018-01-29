# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from .layer import Layer
from .model import Model
from .criterions import *


def rel_error(x, y, eps=1e-8):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(eps, np.abs(x) + np.abs(y))))



def eval_numerical_gradient(f, x, h=1e-5):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h) # the slope
        it.iternext() # step to next dimension
    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-12, np.abs(x) + np.abs(y))))

class GradientsChecker:
    """
    Это специальный класс предназначенный для вычисления градиентов по параметрам и входам нейронной сети
    """
    def __init__(self, step=1e-5, batch_size=10, verbose=False, seed=0):
        self.step = step
        self.batch_size = batch_size
        self.verbose = verbose
        self.gen = np.random.RandomState(seed)
    def _print(self, msg):
        if self.verbose:
            print(msg)
        
    def _get_X(self, layer, X):
        if X is None:
            input_shape = layer.input_shape
            assert len(input_shape) >= 2
            input_shape = tuple([self.batch_size] + list(input_shape[1:]))
            X = self.gen.normal(size=input_shape)
            self._print('Generated X with shape = {}'.format(X.shape))
        assert isinstance(X, np.ndarray)
        return X
    def _get_y(self, model, X, y):
        batch_size = X.shape[0]
        if y is None:
            if isinstance(model.criterion, MSECriterion):
                y = self.gen.normal(size=batch_size)
            elif isinstance(model.criterion, MulticlassLogLoss):
                n_classes = model.criterion.n_classes
                y = self.gen.randint(0, n_classes, size=batch_size, dtype=np.int32)
            else:
                assert False, 'Unknown criterion: cannot generate (X, y) pair'
            if self.verbose:
                self._print('Generated y with shape = {}'.format(y.shape))
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[1:] == model.input_shape[1:]
        assert X.shape[0] == y.shape[0]
        return y
    def _get_grad_Y(self, layer, X, grad_Y):
        assert isinstance(layer, Layer)
        assert isinstance(X, np.ndarray)
        output_shape = tuple([X.shape[0]] + list(layer.output_shape[1:]))
        if grad_Y is None:
            grad_Y = self.gen.normal(size=output_shape)
            self._print('Generated grad_Y with shape = {}'.format(grad_Y.shape))
        assert isinstance(grad_Y, np.ndarray)
        assert grad_Y.shape == output_shape
        return grad_Y
        
    def _save_state(self, layer):
        self.saved_params = layer.get_params(copy=True)           # Получить копии параметров сети
        self.saved_grad_params = layer.get_grad_params(copy=True) # Получить копии градиентов параметров сети
    
    def _restore_state(self, layer):
        layer.set_params(self.saved_params)              # Восстановление параметров слоя
        layer.set_grad_params(self.saved_grad_params)    # Восстановление градиентов слоя
        
    def eval_gradients(self, layer, input=None, output=None, params=None, restore_state=False):
        method_name = 'eval_gradients'
        assert isinstance(layer, Layer)
        # Сохранение состояния модели в случае использования внешних параметров
        if restore_state:
            self._save_state(layer)
        # Выставление внешних параметров
        if params is not None: # При оценке градиентов будут использованы указанные в params параметры модели
            layer.set_params(params) # Выставление требуемых параметров
            layer.zero_grad_params() # Обнуление градиентов
            if self.verbose:
                print('{}.{}(...): setting outer parameters'.format(type(self).__name__, method_name))    
        if isinstance(layer, Model):
            self._eval_model_gradients(layer, input, output)
        else:
            self._eval_layer_gradients(layer, input, output)
        if input is None:
            output = None
        # Восстановление состояния модели
        if restore_state:
            self._restore_state(layer)
        
    def _eval_model_gradients(self, model, X=None, y=None):
        """
        Оценивает градиент модели по параметрам.
        
        Внимание: после оценки градиентов модель в общем случае уже не будет находиться в изначальном состоянии,
        хотя и значения параметров и их производных будут возвращены в исходное значение. Как минимум это связано
        с потенциальным вызовом генераторов. Кроме того, в случае BatchNormalization слоя происходит обновление 
        скользящих среднего и дисперсии.
        
        Аргументы:
        - model
        - X
        - y
        Возвращаемое значение:
        -
        """   
        X = self._get_X(model, X)
        y = self._get_y(model, X, y)
        # Функция потерь
        def function(self, *args, **kwargs):
            return model.forward(X, y)
        # Численная оценка градиентов
        self.num_grad_params = OrderedDict()
        for p_name, p_value in model.get_params().items():
            self.num_grad_params[p_name] = eval_numerical_gradient(function, p_value, h=self.step)
        # Аналитическая оценка градиентов
        loss = model.forward(X, y)
        _    = model.backward(X, y)
        self.grad_params = model.get_grad_params(copy=True)
        # Печать относительной ошибки
        for p_name in self.grad_params:
            print('grad_{} error = {}'.format(
                p_name, rel_error(self.num_grad_params[p_name], self.grad_params[p_name])))    

    def _eval_layer_gradients(self, layer, X=None, grad_Y=None):
        X = self._get_X(layer, X)
        grad_Y = self._get_grad_Y(layer, X, grad_Y)
        # Функция нахождения выхода сети
        def function(self, *args, **kwargs):
            return layer.forward(X)
        # Численная оценка градиента по входу
        self.num_grad_input = eval_numerical_gradient_array(function, X, grad_Y, h=self.step)
        # Численная оценка градиентов по параметрам
        self.num_grad_params = OrderedDict()
        for p_name, p_value in layer.get_params().items():
            self.num_grad_params[p_name] = eval_numerical_gradient_array(function, p_value, grad_Y, self.step)
        # Аналитическая оценка градиентов по входу и параметрам
        self.grad_input  = layer.backward(X, grad_Y)
        self.grad_params = layer.get_grad_params(copy=True)
        # Печать относительной ошибки
        for p_name in self.grad_params:
            print('grad_{} error = {}'.format(
                p_name, rel_error(self.num_grad_params[p_name], self.grad_params[p_name])))
        print('grad_X error = {}'.format(rel_error(self.num_grad_input, self.grad_input)))
