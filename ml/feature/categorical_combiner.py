# -*- coding: utf-8 -*-

import numpy as np
from itertools import product, chain, combinations
from collections import OrderedDict, Counter, defaultdict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from ..core.helpers import Checker, Printer
from .categorical_feature import CategoricalFeature

class CategoricalCombiner(Checker):
    """
    get_all_combinations
    get_combined_feature
    """
    METHOD = 4
    def __init__(self, treat_const='none', verbose=0):
        super().__init__()
        self._verbose = verbose
        self._set_treat_const(treat_const)
        
    def _set_treat_const(self, treat_const):
        treat_values = ['none', 'delete', 'error']
        if treat_const not in treat_values:
            raise ValueError('treat_const must be one of the following: {}'.format(treat_values))
        self._treat_const = treat_const
        
    def _preprocess_features(self, features):
        _features = OrderedDict()
        self._check_type(features, 'features', (dict, tuple, list, np.ndarray))
        if isinstance(features, dict):
            for fname, feature in features.items():
                _features[fname] = feature
        elif isinstance(features, (list, tuple, np.ndarray)):
            for feature in features:
                _features[feature.get_name()] = feature
        else:
            assert False, 'This place must not be reached. Something is wrong.'
        
        if self._treat_const == 'none':
            return _features
        
        features = _features
        if self._treat_const == 'delete':
            _features = OrderedDict()
            for fname, feature in features.items():
                if not feature.is_constant():
                    _features[fname] = feature
            return _features
        
        assert self._treat_const == 'error'
        for fname, feature in feature.items():
            if feature.is_constant():
                raise ConstantFeatureError(fname, type(self).__name__)
            _features[fname] = feature
        return _features
    
    def get_all_combinations(self, features, degree=None, hash=hash):
        """
        Возвращает всевозможные комбинации степени degree из признаков 
        Константные признаки согласно политике treat_const
        Аргументы:
            :param features - признаки для комбинирования; все должны быть CategoricalFeature
            :param degree   - степень комбинаций; каждый новый признак - это комбинация degree признаков
            :param hash     - функция превращения комбинации признаков в значение нового признака
        """

        _features = self._preprocess_features(features)
        assert isinstance(_features, OrderedDict), 'By this time "features" must be stored in OrderedDict.'
        _feature_names = list(_features.keys())
        _features = list(_features.values())
        assert isinstance(_features, list), 'By this time "features" must be stored in list.'
        
        method_msg = self._method_msg('get_all_combinations')
        methdo_msg = method_msg + '({}, degree={})'.format(_feature_names, degree)
        self._printers[self.METHOD](method_msg)
        
        combined_features = OrderedDict()
        if degree is None:
            degree_range = range(1, len(_feature_names) + 1)
        else:
            degree_range = [degree]
        for degree in degree_range:
            for some_features in combinations(_features, degree):
                new_feature = self.get_combined_feature(some_features, hash) 
                combined_features[new_feature.get_name()] = new_feature
        combined_fetures = self._preprocess_features(combined_features)
        assert isinstance(combined_features, OrderedDict), 'By this time "combined_features" must be stored in OrderedDict.'
        return combined_features

    def get_combined_feature(self, features, hash=hash):
        self.check_sizes_(features)
        if len(features) < 1:
            raise ValueError('At least one feature name must be given')
        if len(features) == 1:
            return features[0].deepcopy()
                             
        feature_values = []
        feature_names = []
        for feature in features:
            values = feature.get_values(False).flatten()
            feature_values.append(values)
            feature_names.append(feature.get_name())
            
        new_values = []
        for hyper_value in zip(*feature_values):
            new_values.append(hash(hyper_value))
        new_values = LabelEncoder().fit_transform((new_values))
        new_name = '+'.join(feature_names)
        return CategoricalFeature(new_values, new_name)

    def check_sizes_(self, features):
        if len(Counter([len(feature) for feature in features])) != 1:
            raise ValueError('Features must have equal sizes!')
