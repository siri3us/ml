# -*- coding: utf-8 -*-

import numpy as np
from ..core import Checker

class Regularizer(Checker):
    def get_loss(self, param):
        assert False, 'Not defined'
    def get_grad(self, param):
        assert False, 'Not defined'
    def update_grad(self, param, grad_param):
        """
        Производит inplace обновление градиентов для экономии памяти.
        """
        assert False, 'Not defined'

class EmptyRegularizer(Regularizer):
    def get_loss(self, param):
        return 0.0
    def get_grad(self, param):
        return np.zeros_like(param, dtype=param.dtype)
    def update_grad(self, param, grad_param):
        pass
    
class L2Regularizer(Regularizer):
    def __init__(self, l2=0.0):
        self._check_nonegative(l2, 'l2')
        self.l2 = l2
    def get_loss(self, param):
        return 0.5 * self.l2 * np.sum(param ** 2)
    def get_grad(self, param):
        return self.l2 * param
    def update_grad(self, grad_param):
        grad_param += self.get_grad(grad_param)
