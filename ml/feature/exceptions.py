# -*- coding: utf-8 -*-

import numpy as np
import collections
from .feature_base import FeatureBase


class UnknownFeatureError(Exception):
    def __init__(self, fname, cname):
        super().__init__()
        assert isinstance(feature, FeatureBase)
        self.str = 'Class "{}" does not contain feature "{}".'.format(cname, fname)
    def __str__(self):
        return self.str
    def message(self):
        return self.str
    
class ConstantFeatureError(Exception):
    def __init__(self, fname, cname=None):
        super().__init__()
        assert isinstance(feature, FeatureBase)
        self.str = 'Feature "{}" is constant'.format(fname)
        if isinstance(cname, str):
            self.str += ': not allowed in an instance of class "{}".'.format(cname)
        else:
            self.str += '.'
    def __str__(self):
        return self.str
    def message(self):
        return self.str
