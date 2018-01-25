# -*- coding: utf-8 -*-

import numpy as np

class Regularizer:
    pass


class EmptyRegularizer(Regularizer):
    def __bool__(self):
        return False


class L2Regularizer(Regularizer):
    def __init__(self, l2=0.0):
        self.l2 = l2
    def __bool__(self):
        return True
    def loss(self, arr):
        return 0.5 * self.l2 * np.sum(arr ** 2)
    def grad(self, arr):
        return self.l2 * arr
