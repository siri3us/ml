# -*- coding: utf-8 -*-

import numpy as np
import unittest
import copy
import numbers
from scipy.sparse import csc_matrix, csr_matrix
from .feature_base import FeatureBase

class NumericalFeature(FeatureBase):
    def __init__(self, values, name, verbose=0):
        super().__init__(values, name, verbose)
        self._name = self._get_numerical_name(name)
        self._kernel._check_numeric(self._values, self._name, True)

    def get_categorical_feature(self, bins, right=True, include_lowest=False):
        """
        Создает категориальный признак из числового.
        TODO include_lowest
        Аргументы:
            :param bins - 
            :param right -  
            :param include_lowest - на данный момент не используется
        """
        if self._sparse:
            values = self._values.toarray().flatten()
        else:
            values = self._values
        cat_values = np.array(pd.cut(values, bins, right=right))
        cat_name = self._get_categorical_name(self._name)
        from .categorical_feature import CategoricalFeature
        return CategoricalFeature(cat_values, cat_name)

