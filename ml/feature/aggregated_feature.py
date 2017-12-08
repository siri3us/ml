# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy
from collections import defaultdict, Counter, OrderedDict

from .feature_base import FeatureBase
from .exceptions import *
from ..core.helpers import Checker, Printer

class AggregatedFeature(Checker):
    DELETE_FEATURE = 4
    """
    Позволяет хранить значения нескольких признаков, например, OHE представления категориальных признаков.
    Внутренне хранит признаки в OrderedDict с доступом по имени признака
    Методы:
        is_constant
        get_values
        get_features
    """
    def __init__(self, features, name, copy=True, treat_const='none', verbose=0):
        """
        Аргументы:
            :param features - список объектов FeatureBase
            :param name     - имя агрегированного признака
            :param exclude_const - исключить константные признаки из множества?
            :param copy     - если True, то каждый каждый признак будет скопирован
            :param verbose  - уровень печати (nonnegative int)
        """
        self._set_treat_const(treat_const)
        self._set_features(features, name, copy)
        self._verbose = verbose
        
        
    def _set_treat_const(self, treat_const):
        treat_vals = ['none', 'delete', 'error']
        if treat_const not in treat_vals:
            raise ValueError('treat_const must be one of the following: {}.'.format(treat_vals))
        self._treat_const = treat_const
        
    def _set_features(self, features, name, copy=True):
        """
            :param features - список объектов FeatureBase
            :param name - имя агрегированного признака
            :param copy - если True, то сохраняются копии признаков
        """
        self._check_features(features, name)
        self._name = name
        self._feature_names = [feature.get_name() for feature in features]
        _features = OrderedDict()
        for feature in features:
            if copy: _features[feature.get_name()] = feature.deepcopy()
            else:    _features[feature.get_name()] = feature

        if self._treat_const == 'none':
            self._features = _features
        elif self._treat_const == 'delete':
            self._features = OrderedDict()
            for fname, feature in _features.items():
                if not feature.is_constant():
                    self._features[feature.get_name()] = feature
        else:
            for fname, feature in _features.items():
                if feature.is_constant():
                    raise ConstantFeatureError(feature.get_name(), type(self).__name__)
            self._features = _features
        
    def _check_features(self, features, name):
        """
        Проверяет, что признаки хранятся в list или np.ndarray, все признаки имеют тип FeatureBase,
        размеры признаков равны.
        """
        if not isinstance(features, (np.ndarray, list)):
            raise TypeError('Wrong format of "features" with name "{}" for "{}".'.format(name, type(self).__name__))
        if not all([isinstance(feature, FeatureBase) for feature in features]):
            raise TypeError('One of subfeatures of feature "{}" is not an object of FeatureBase.'.format(name))
        lengths = [len(feature) for feature in features]
        if min(lengths) != max(lengths):
            raise ValueError('Provided features with name "{}" have different lengths'.format(name))
        if min(lengths) == 0:
            raise ValueError('Features with name "{}" have zero length. Must have positive length.'.format(name))

    def _check_feature_names(self, feature_names=None):
        if feature_names is None:
            feature_names = []
        for feature_name in feature_names:
            if not feature_name in self._features:
                raise UnknownFeatureError(feature_name, type(self).__name__)
            
    def exclude_constant(self, verbose=False):
        """
        Исключает константные подпризнаки из рассмотрения.
        """
        to_delete = []
        for feature_name in self._features:
            if self._features[feature_name].is_constant():
                to_delete.append(feature_name)
        for feature_name in to_delete:
            if verbose: print('Deleting constant feature "{}"'.format(feature_name))
            del self._features[feature_name]
        self._feature_names = [feature_name for feature_name in self._feature_names 
                               if feature_name in self._features]

    def is_constant(self):
        if len(self._features) == 0: # In case if all feature are excluded due to constant values
            return True
        return all([feature.is_constant() for feature in self._features.values()])
            
    def get_values(self, feature_names=None, sparse=False, as_dataframe=False, **kwargs):
        """
        Возвращает np.array или pd.DataFrame размера N x D, где N - число объектов, а D - число признаков.
        Аргументы:
            :param feature_names - список имен признаков, для агреггирования
            :param sparse        - если True, 
            :param as_dataframe  - если True, 
        Возвращаемое значение:
            np.array или pd.DataFrame
        """
        if feature_names is None:
            feature_names = [feature_name for feature_name in self._features]
        self._check_feature_names(feature_names)
        features = [self._features[feature_name] for feature_name in feature_names]
        
        X = []
        for feature in features:
            X.append(feature.get_values(sparse=sparse))
        if len(X) == 0: # Если вдруг все пусто
            raise ValueError("All values are constant. Senseless feature!")
        if sparse:
            X = scipy.sparse.hstack(X)
        else:
            X = np.concatenate(X, axis=1)
            if as_dataframe:
                X = pd.DataFrame(X, columns=feature_names)
        return X
    
    def get_features(self, copy=True):
        features = []
    
    def __repr__(self):
        s = '{}['.format(self._name)
        for feature_name in [feature_name for feature_name in self._feature_names 
                             if feature_name in self._features]:
            feature = self._features[feature_name]
            s += str(feature)
        s += ']'
        return s
