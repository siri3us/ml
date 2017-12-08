# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy
import copy
from collections import OrderedDict, defaultdict, Counter
from ..core.helpers import Printer, Checker
from .categorical_feature import CategoricalFeature
from .categorical_combiner import CategoricalCombiner

class CategoricalFeaturesManager(Checker):
    """
    Данный класс отвечает за хранение категориальных признаков. Предоставляет пользователю 
    следующие возможности:
        is_present
        add_feature
        set_feature
        del_feature
        get_feature
        get_list_of_features
    """
    METHOD = 4
    def __init__(self, features=None, treat_const='none', verbose=0):
        """
            :param verbose - уровень печати сообщений
        """
        super().__init__()
        self._features = OrderedDict()
        self._n_samples = None
        self._verbose = verbose
        self._categorical_combiner = CategoricalCombiner(treat_const, verbose)
        if features is not None:
            self.set_features(features)
        
    def __contains__(self, feature_name):
        return feature_name in self._features
 
    ###################################################################
    def is_present(self, name):
        """
        Возвращает True, если признак с таким именем содержится в хранилище. Иначе - False.
        """
        return (name in self._features)
    def _check_if_present(self, *args):
        for name in args:
            if not self.is_present(name):
                raise ValueError(self._error_msg("unknown feature \"{}\"".format(name)))  
    def _is_binary(self, values):
        if len(Counter(values)) == 2:
            return True
        return False
    def _check_feature(self, feature):
        self._check_type(feature, str(feature), CategoricalFeature)
        if (self._n_samples is not None) & (len(feature) != self._n_samples):
            raise ValueError("Given feature vector has size {} while must have size {}.".format(
                        len(feature), self._n_samples))

    ###################################################################
    def add_feature(self, feature, copy=True, replace=False):
        self.set_feature(feature, copy, replace)
    def set_feature(self, feature, copy=True, replace=False):
        """
        Помещает признак в хранилище. По умолчанию всегда вызывает исключение, если признак с таким
        именем уже есть в хранилище. По умолчанию всегда сохраняет в хранилище копию признака.
        Аргументы:
            :param feature - словарь из {имя_признака: признак}. (dict)
            :param copy    - если True, то в хранилище будет помещена копия признака. (bool)
            :parma replace - если True, то признак с таким же именем будет заменен;
                             если False, то наличине признака с таким же именем вызывает исключение. (bool)
        """
        
        self._check_feature(feature) # новый признак имеет правильный размер и категориальный тип
        if not replace:              # если замена признака запрещена ...
            if self.is_present(feature.get_name()): # и уже есть признак с таким именем, то ...
                error_msg = self._method_msg('set_feature: ') +\
                    'feature "{}" cannot be replaced. Check "replace" parameter'.format(feature.get_name())
                raise ValueError(error_msg)

        self._n_samples = len(feature)
        self.del_feature(feature, throw=False)
        if copy:
            self._features[feature.get_name()] = feature.deepcopy()
        else:
            self._features[feature.get_name()] = feature
    def set_features(self, features, copy=True, replace=False):
        for feature in features:
            self.set_feature(feature, copy=copy, replace=replace)
    def del_feature(self, feature_name, throw=True):
        """
        Аргументы:
            :param feature_name
            :param throw
        """
        if not self.is_present(feature_name):
            if throw:
                error_msg = self._method_msg('del_feature') +\
                    'feature "{}" is not present in storage. Cannot be deleted.'.format(feature_name)
                raise KeyError(error_msg)
            return False
        else:
            del self._features[feature_name]
            if len(self._features) == 0:
                self._n_samples = None
            return True
    
    def get_feature(self, feature_name, copy=True):
        self._check_if_present(feature_name)
        if copy:
            return self._features[feature_name].deepcopy()
        return self._features[feature_name]
    
    def get_list_of_features(self):
        return list(self._features.keys())


    ###################################################################
    #    Функции комбинирования категориальных признаков              #
    ###################################################################
    def add_all_combinations(self, feature_names, degree, hash=hash):
        self.get_all_combinations(feature_names, degree, hash=hash, store=True, copy=False)
        
    def get_all_combinations(self, feature_names, degree, hash=hash, store=True, copy=True):
        method_msg = self._method_msg('get_all_combinations')
        self._printers[self.METHOD](method_msg + '(names={}, degree={}, store={})'.format(feature_names, degree, store))
        self._check_if_present(feature_names)
        
        features = {name: self._features[name] for name in feature_names}
        combined_features = self._categorical_combiner.get_all_combinartions(features, degree=degree, hash=hash)
        if store:
            if degree > 1:
                for name, combined_feature in combined_features:
                    self.set_feature(combined_feature, copy=copy, replace=False)
        return combined_features

    def add_combined_feature(self, feature_names, hash=hash):
        self.get_combined_feature(feature_names, hash=hash, store=True, copy=False)
        
    def get_combined_feature(self, feature_names, hash=hash, store=True, copy=True):
        method_msg = self._method_msg('get_combined_feature')
        self._printers[self.METHOD](method_msg + '(names={}, store={})'.format(feature_names, store))
        self._check_if_present(*feature_names)
        
        features = [self._features[name] for name in feature_names]
        combined_feature = CategoricalCombiner().get_combined_feature(features, hash=hash)
        if store:
            if len(feature_names) > 1:
                self.set_feature(combined_feature, copy=copy, replace=False)
        return combined_feature


    ############################################################
    ##       Сборка итогового признакового представления      ##
    ############################################################
    def assemble_data_frame(self, feature_names=None):
        """
        Возвращает dense матрицу
        """
        if feature_names is None:
            feature_names = list(self._features.keys())
        self._check_if_present(*feature_names)
        feature_values = []
        for feature_name in feature_names:
            feature_values.append(self._features[feature_name].get_values(sparse=False))
        return pd.DataFrame(np.hstack(feature_values), columns=feature_names)
    
    def assemble(self, feature_names, sparse=False):
        """
        Аргументы:
            :param feature_names
            :param sparse
        """
        self._check_if_present(feature_names)
        X = []
        feature_map = copy.deepcopy(feature_map)
        for feature_name in feature_names:
            feature = self._features[feature_name]
            X.append(feature.get_values(sparse=sparse))
        if sparse:
            return scipy.sparse.hstack(X)
        return np.hstack(X)

    def filter_features(self, names=None, threshold=1):
        if names is None:
            names = list(self._features.keys())
        for name in names:
            self._features[name]._filter_feature(threshold)
        self._update_dict()
        
    def _update_dict(self):
        new_features = OrderedDict()
        for name in self._features:
            feature = self._features[name]
            new_name = feature.get_name()
            new_features[new_name] = feature
        self._features = new_features
        
    def add_filtered(self, name, threshold):
        self.get_filtered(name, threshold, store=True, copy=False)
    def get_filtered(self, name, threshold, store=True, copy=True):
        self._check_if_present(name)
        new_feature = self._features[name].get_filtered_feature(threshold)
        if store:
            self.set_feature(new_feature, copy=copy, replace=False)
        return new_feature

    def add_counter(self, name):
        self.get_filtered(name, store=True, copy=False)
    def get_counter(self, name, store=True, copy=True):
        self._check_if_present(name)
        new_feature = self._features[name].get_counter_feature(threshold)
        if store:
            self.set_feature(new_feature, copy=copy, replace=False)
        return new_feature
    def get_loo(self, name, y_train, cv, seed=1234):
        self._check_if_present(name)
        new_feature = self._features[name].get_loo_feature(y_train, cv=cv, seed=seed)
        return new_feature
        
