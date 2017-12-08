# -*- coding: utf-8 -*-

import numpy as np
import collections

class FeatureShapeError(Exception):
    def __init__(self, fname, shape, exp_shape, calee=None):
        super().__init__()
        self.fname = fname; self.shape = shape; self.exp_shape = exp_shape; self.calee = calee
        self.str = 'Feature "{}" has shape {} while expected {}.'.format(fname, shape, exp_shape)
        if calee is not None:
           self.str = type(calee).__name__ + ': ' + self.str
    def __str__(self):
        return self.str
    def message(self):
        return self.str


class FeatureLengthError(Exception):
    def __init__(self, fname, length, exp_length, calee=None):
        super().__init__()
        self.fname = fname; self.length = length; self.exp_length = exp_length; self.calee = calee
        self.str = 'Feature "{}" has length {} while expected {}.'.format(fname, length, exp_length)
        if calee is not None:
           self.str = type(calee).__name__ + ': ' + self.str
    def __str__(self):
        return self.str
    def message(self):
        return self.str


class UnknownFeatureError(Exception):
    def __init__(self, fname, calee=None):
        super().__init__()
        self.str = 'Unknown feature "{}".'.format(cname)
        if calee is not None:
            self.str = type(calee).__name__ + ': ' + self.str 
    def __str__(self):
        return self.str
    def message(self):
        return self.str
    
class ConstantFeatureError(Exception):
    def __init__(self, fname, calee=None):
        super().__init__()
        self.str = 'Feature "{}" is constant'.format(fname)
        if calee is not None:
            self.str += type(calee).__name__ + ': ' + self.str
    def __str__(self):
        return self.str
    def message(self):
        return self.str
        
class NumericalFeatureError(Exception):
    def __init__(self, fname, cname=None):
        super().__init__()
    def __str__(self):
        return self.str
    def message(self):
        return self.str
        
        
