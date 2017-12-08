# -*- coding: utf-8 -*-

import numpy as np
import copy
import numbers
from collections import defaultdict, OrderedDict, Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from itertools import product, chain

from .feature_base import *



class CategoricalFeature(FeatureBase):
    """
    Класс для хранения категориальных признаков. На данный момент доступна только реализация с 
    dense хранением данных. 
    
    Список методов:
        deepcopy
        set_label2cat
        set_cat2label
        get_cat_values
        get_filter_feature
        get_counter_feature
        get_loo_feature
        get_le_feature
        get_ohe_feature
    """
    ################################################################################### 

    CAT_FEATURE_INIT = 8
    OHE = 9
    def _is_label_encoded(self, values, name=None):
        """
        Возвращает True, если значения признака label encoded. Иначе возвращает False или вызывает исключение
        (в зависимости от параметра throw).
        Аргументы:
            :param values - категориальные значения (np.ndarray)
            :param name - имя категориального признака (str)
        """
        if not self.is_numeric():
            return False
        labels = sorted(list(set(values)))
        prev_value = labels[0]
        if prev_value != 0:
            return False
        for value in labels[1:]:
            if value != prev_value + 1:
                return False
            prev_value = value
        return True
    
    def is_label_encoded(self):
        return self._is_label_encoded(self._values, self._name)

    def _check_label_encoded(self, values, name, throw=True):
        """
        Возвращает True, если значения признака label encoded. Иначе возвращает False или вызывает исключение
        (в зависимости от параметра throw).
        Аргументы:
            :param values - значения признака (np.ndarray)
            :param name   - имя категориального признака (str)
            :param throw  - вызывать исключение? (bool)
        """
        if not self._is_label_encoded(values):
            if throw: 
                raise ValueError(self._error_msg('Feature "{}" is not label-encoded'.format(name)))
            return False
        return True
            
    def _check_cat2label(self, values, name, cat2label, throw=True):
        """
        Проверяет, что преобразование категорий в метки корректно. Возвращает True в случае корректности.
        Иначе возвращает False или вызывает исключение (в зависимости от параметра throw).
        Аргументы:
            :param values    - значения признака (np.ndarray)
            :param name      - имя признака (cat)
            :param cat2label - преобразование в метки (dict)
            :param throw     - вызывать исключение? (bool)
        """
        if not set(values) == set(cat2label.values()):
            if throw: 
                print(set(values), set(cat2label.values()))
                raise ValueError(self._error_msg('Num of values != number of labels for feature "{}"'.format(name)))
            else: 
                return False
        if not len(set(cat2label.keys())) == len(set(cat2label.values())):
            if throw: 
                raise ValueError(self._error_msg('There is no one-to-one correspondance in cat2label for feature "{}"'.format(name)))
            else: 
                return False
        return True
    
    ###################################################################################
    def deepcopy(self):
        new_feature = CategoricalFeature(copy.deepcopy(self._values), self._name, verbose=self._verbose)
        new_feature.set_cat2label(self._cat2label)
        return new_feature
        
    def __repr__(self):
        return type(self).__name__ + '({}; {})'.format(self.name, self.values.shape)
    
    def __init__(self, values, name, cat2label=None, verbose=0):
        """
        По завершении работы конструктора признаки оказываются закодированы метками от 1 до N, где
        N - число различных значений признака.
        Аргументы:
            :param values - значения категориальной переменной (np.ndarray, list)
            :param name   - имя категориальной переменной (str)
            :param cat2label - mapping для преобразования категорий в метки (dict)
        """
        assert isinstance(values, (np.ndarray, list))
        super().__init__(values, name, verbose)
        self._name = self._get_categorical_name(name)
        msg_init = self._info_msg('__init__({})'.format(name))
        
        self._cat2label = None
        self._label2cat = None
        if cat2label is not None:
            self._printers[self.CAT_FEATURE_INIT](msg_init + ': applying mapping "cat2label" to values')
            self._values = np.array(list(map(lambda cat: cat2label[cat], self._values)))
            self._check_cat2label(self._values, self._name, cat2label)
            self._cat2label = copy.deepcopy(cat2label)
            self._label2cat = {label:cat for cat, label in cat2label.items()}
            
        self._properties = {}
        self._properties['is_numeric'] = self.is_numeric()
        self._properties['is_label_encoded'] = self.is_label_encoded()
        self._properties['is_constant'] = self.is_constant()
        if self._properties['is_label_encoded']:
            self._printers[self.CAT_FEATURE_INIT](msg_init + ': feature "{}" is already label encoded'.format(name))
        else:
            self._printers[self.CAT_FEATURE_INIT](msg_init + ': label encoding feature "{}"'.format(name))
        # These values are used for filtering rare values
        self._threshold = None
        self._unique_label = None
        self._label_encode()

        assert self._properties['is_label_encoded'], 'By the end of constructor feature "{}" is not label encoded. Something is wrong.'.format(self.name)
        assert self._properties['is_numeric'], 'By the end of the constructor feature "{}" is not numeric. Something is wrong'.format(self.name)
        
    ##################################################################################
    def set_label2cat(self, label2cat=None):
        if label2cat is None:
            self._label2cat = None
            self._cat2label = None
        else:
            cat2label = {cat:label for label, cat in label2cat.items()}
            self._check_cat2label(self._values, self._name, cat2label, True)
            self._label2cat = copy.deepcopy(label2cat)
            self._cat2label = cat2label
            
    def set_cat2label(self, cat2label=None):
        """
        Подразумевает, что сейчас в self._values хранятся метки
        """
        if cat2label is None:
            self._cat2label = None
            self._label2cat = None
        else:
            self._check_cat2label(self._values, self._name, cat2label, True)
            self._cat2label = copy.deepcopy(cat2label)
            self._label2cat = {label:cat for cat, label in cat2label.items()}
        
    def get_cat_values(self):
        """
        Возвращает признаки в виде изначальных категорий, а не в LE-закодированном виде, в котором 
        они хранятся внутри класса CategoricalFeature.
        """
        if self._label2cat is None:
            # Такое возможно только если признак изначально был передан в закодированном виде
            assert self._properties['is_label_encoded'], 'Expected encoded feature.'
            return np.array(self._values)
        return np.array(list(map(lambda label: self._label2cat[label], self._values)))
        
    ##################################################################################
        
    def _filter_feature(self, threshold):
        """
        Отфильтровывает те категории, которые встречаются не более threshold раз. Заменяет их на новую 
        категорию. Данная категория будет иметь максимальное значение метки. Применение данной функции 
        ведет к преобразованию имени признака: добавляется приставка FIL_
        
        Аргументы:
            :param threshold - если число появлений категории не превосходит threshold, 
                                то она отсеивается (int, float)
        """
        
        # Checking if the feature is label encoded
        if not self._properties['is_label_encoded']:
            raise ValueError('Cannot filter feature "{}" as it is not label encoded.'.format(self.name))
        # Even if the filtration does not change feature values, we change its name and threshold parameters
        self._name = self._get_filtered_name(self._name, threshold)
        self._threshold = threshold
        
        # Checking if there are rare values present in the feature
        counts = Counter(self._values)
        for label, n_occurences in counts.items():
            if n_occurences <= threshold:
                self._unique_label = self._values.max() + 1
                self._properties['is_label_encoded'] = False
                break
        if self._unique_label is None: 
            # There are no labels which occur less or equal to threshold times
            return
        
        # Some features occur less or equal threshold times. Let us find them
        rare_labels = set()
        rare_categories = set()
        # Changing rare labels to the chosen unique_label
        for n, label in enumerate(self._values):
            if counts[label] <= threshold:
                if self._cat2label is not None:
                    rare_labels.add(label)
                    rare_categories.add(self._label2cat[label])
                self._values[n] = self._unique_label # setting rare label to new value

        # Forming new categories names
        if self._cat2label is not None:    
            if len(rare_categories) > 1:
                new_cat = '(' + '|'.join(sorted(list(rare_categories))) + ')'
            elif len(rare_categories) == 1:
                new_cat = list(rare_categories)[0]
            else:
                assert False, '"rare_categories" must not be empty at this point. Something is wrong.'

            for label in rare_labels:
                del self._label2cat[label]
            self._label2cat[self._unique_label] = new_cat
            for cat in rare_categories:
                del self._cat2label[cat]
            self._cat2label[new_cat] = self._unique_label
            
        self._properties['is_constant'] = self.is_constant()
        self._label_encode()
        assert self._properties['is_label_encoded']
        assert self._properties['is_numeric']
        
    def get_filtered_feature(self, threshold):
        """ 
        Возвращает признак, полученный из данного фильтрацией категорий по порогу threshold: 
        все категории, встречающиеся не чаще чем threshold, отфильтровываются функцией _filter_feature.
        Все отфильтрованные категории становятся новой категорией.
       
        Аргументы:
            :param - если число появлений категории не превосходит threshold, то она отсеивается (int, float)
        """
        new_feature = self.deepcopy()
        new_feature._filter_feature(threshold)
        return new_feature
    
    def get_counter_feature(self):
        """
        Возвращает признак NumericalFeature, равный числу появления каждой из категорий.
        """
        counts = Counter(self._values)
        new_values = np.zeros_like(self._values)
        for n, value in enumerate(self._values):
            new_values[n] = counts[value]
        new_name = self._get_counter_name(self._name)
        from .numerical_feature import NumericalFeature
        return NumericalFeature(new_values, new_name)

    def get_loo_feature(self, Y_train, cv, alpha=0.01, seed=1234, scale=0.01):
        """
        Предполагает, что первые len(Y_train) примеров принадлежат обучающей выборке
        """
        assert isinstance(Y_train, (np.ndarray, list))
        assert len(Y_train) <= len(self._values)
        train_size = len(Y_train)
        test_size = len(self._values) - train_size
        
        np.random.seed(seed)
        X_train = self._values[:train_size]
        mean_y = np.mean(Y_train)
        all_labels = set(self._values)
        X_new_train = np.zeros(len(X_train))

        for n_split, (train_indices, test_indices) in enumerate(cv.split(X_train, Y_train)):
            x_train, y_train = X_train[train_indices], Y_train[train_indices]
            x_test, y_test = X_train[test_indices], Y_train[test_indices]
            for label in all_labels:
                N_all = x_train.shape[0]
                train_mask = x_train == label
                N_label = np.sum(train_mask)
                print('n_split = {}, label = {}, den = {}'.format(n_split, label, N_label + alpha * N_all))
                X_new_train[test_indices[x_test == label]] = \
                    (np.sum(y_train[train_mask]) + alpha * mean_y * N_all) / (max(N_label, 1) + alpha * N_all)
        if scale > 0:
            multipliers = np.random.normal(loc=1.0, scale=scale, size=len(self._values))
        else:
            multipliers = np.ones(len(self._values))
        if test_size > 0:
            X_test = self._values[train_size:]
            X_new_test = np.zeros(test_size)
            for label in all_labels:
                train_mask = X_train == label
                N_all = train_size
                N_label = np.sum(train_mask)
                X_new_test[X_test == label] = (np.sum(Y_train[train_mask]) +
                                               alpha * mean_y * train_size) / (max(N_label, 1) + alpha * N_all)

            X_new = np.concatenate([X_new_train, X_new_test]) * multipliers
        else:
            X_new = X_new_train * multipliers
        new_name = self._get_loo_name(self._name)
        from .numerical_feature import NumericalFeature
        return NumericalFeature(X_new, new_name)
        
    ############################################################
    ##                       Кодировщики                      ##
    ############################################################
    def _label_encode(self):
        """
        Выполняет label-кодирование признака.
        """
        if self._properties['is_label_encoded']:
            if len(FEATURE_PREFIXES['LE']) > 0:
                if not self._name.startswith(FEATURE_PREFIXES['LE']):
                    self.name = self._get_label_encoded_name(self._name)
            return
        
        label_encoder = LabelEncoder()
        self._values = label_encoder.fit_transform(self._values)
        classes = label_encoder.classes_
        old_label2new_label = {old_label:new_label for new_label, old_label in enumerate(classes)}
        new_label2old_label = {new_label:old_label for new_label, old_label in enumerate(classes)}

        self._name = self._get_label_encoded_name(self.name)
        self._properties['is_label_encoded'] = self.is_label_encoded()
        self._properties['is_numeric'] = self.is_numeric()
        self._properties['is_constant'] = self.is_constant()
        
        if self._unique_label is not None:
            # This placed can be reached when _label_encode() is invoked from _filter_feature()
            self._unique_label = old_label2new_label[self._unique_label]
            assert self._unique_label == len(old_label2new_label) - 1
            assert (FEATURE_PREFIXES['FIL'] + '{}_'.format(self._threshold)) in self._name
            
        if self._label2cat is None:
            self._cat2label = old_label2new_label
            self._label2cat = new_label2old_label
        else:
            new_label2cat = {}
            for old_label in self._label2cat:
                new_label = old_label2new_label[old_label]
                new_label2cat[new_label] = self._label2cat[old_label]
            cat2new_label = {cat:new_label for new_label, cat in new_label2cat.items()}
            self._cat2label = cat2new_label
            self._label2cat = new_label2cat
            
        assert self._properties['is_label_encoded']
        assert self._properties['is_numeric']

    def get_le_feature(self):
        """
        Возвращает LE-закодированный признак, полученный на основе данного. В данной реализации CategoricalFeature
        поддерживается инваринат: внутреннее состояние признака всегда LE-закодированное. Поэтому вызов
        _label_encode() в реализации функции по сути бесполезен. Возможно что-то измениться в будущих версиях.
        """
        new_feature = self.deepcopy()
        new_feature._label_encode()
        return new_feature
    
    def get_ohe_feature(self, sparse=True, omit_uniques=False):
        """
        Аргументы:
            :param sparse       - вернуть sparse или dense представление? (bool)
            :param omit_uniques - если True, то отфильтрованная категория не войдет в состав OHE-признака (bool)
        """
        assert self._properties['is_label_encoded']
        assert self._properties['is_numeric']
        msg_base = self._method_msg('get_ohe_feature(): ')
        
        if (not omit_uniques) or (self._unique_label is None):
            unique_label = -1
        else:
            unique_label = self._unique_label
        
        ohe_name = self._get_ohe_name(self._name)
        counter = Counter(self._values)
        
        from .numerical_feature import NumericalFeature
        if self._properties['is_constant']:
            # No sense of OHE for constant feature
            assert np.sum(self._values) == 0
            assert len(self._cat2label) == 1
            assert list(self._cat2label.values())[0] == 0
            self._printers[self.OHE](msg_base + 'OHE of constant feature "{}".'.format(self._name))
            return NumericalFeature(self._values, ohe_name)
        
        if (len(counter) == 2):
            # In case of binary feature one column of OHE representation can be omitted
            assert set(self._cat2label.values()) == set([0, 1])
            self._printers[self.OHE](msg_base + 'OHE senseless for binary feature "{}".'.format(self._name))
            return NumericalFeature(self._values, ohe_name)
        
            # На данный момент непонятно, почему при unique_label >= 0 возвращали константу
            """if unique_label >= 0:
                assert unique_label == 1
                self._printers[self.OHE](msg_base + 'omiting unique label for "{}" turns it constant.'.format(self._name))
                return NumericalFeature(np.zeros(len(self._values)), ohe_name)
            else:
                self._printers[self.OHE](msg_base + 'OHE senseless for binary feature "{}".'.format(self._name))
                return NumericalFeature(self._values, ohe_name)"""
        
        ohe_values = OneHotEncoder(sparse=sparse).fit_transform(self._values[:, np.newaxis])
        if sparse:
            ohe_values = ohe_values.tocsc()
        if unique_label >= 0:
            assert unique_label == len(counter) - 1
            mask = (self._values == unique_label)
            if sparse:
                last_column = ohe_values[:, unique_label].toarray().flatten()
            else:
                last_column = ohe_values[:, unique_label]
            assert np.all(last_column == mask), 'Last column of ohe feature must correspond to unique_label.'
            ohe_values = ohe_values[:, :unique_label]       
        
        feature_names = []
        feature_values = []
        for label in sorted(self._label2cat.keys()):
            if label != unique_label:
                feature_names.append(str(self._label2cat[label]))
                feature_values.append(ohe_values[:, label])
        features = [NumericalFeature(fvalues, fname) for fvalues, fname in zip(feature_values, feature_names)]
        from .aggregated_feature import AggregatedFeature
        return AggregatedFeature(features, ohe_name, verbose=self._verbose, copy=False)
        
    def get_properties(self):
        return copy.deepcopy(self._properties)

