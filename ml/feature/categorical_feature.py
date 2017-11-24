# -*- coding: utf-8 -*-

from .feature_base import *
import numpy as np
import unittest
import copy
import numbers

class CategoricalFeature(FeatureBase):
    
    ################################################################################### 
    
    def _is_label_encoded(self, values, name):
        if not self._is_numeric(values):
            return False
        # Here labels are numeric
        labels = sorted(list(set(values)))
        prev_value = labels[0]
        if prev_value != 0:
            return False
        for value in labels[1:]:
            if value != prev_value + 1:
                return False
            prev_value = value
        return True

    def _check_label_encoded(self, values, name=None, throw=True):
        if not self._is_label_encoded(values):
            if throw:
                raise ValueError("Feature {} is not label-encoded".format(name))
            return False
        return True
            
    def _check_cat2label(self, values, name, cat2label, throw=True):
        if not set(values) == set(cat2label.values()):
            if throw:
                raise ValueError('Num of values != number of labels for feature {}'.format(name))
        if not len(set(cat2label.keys())) == len(set(cat2label.values())):
            if throw:
                raise ValueError('There is no one-to-one correspondance in cat2labe for feature {}'.format(name))
         
    ###################################################################################

    def __repr__(self):
        return 'CategoricalFeature({}; {})'.format(self.name, self.values.shape)
    
    def __init__(self, values, name, cat2label=None, apply_mapping=False, verbose=0):
        """
        values - значения категориальной переменной
        name - название категориальной переменной
        cat2label - mapping для преобразования категорий в метки
        """
        super().__init__(values, name, verbose)
        assert self.sparse == False
        self.name = self._get_categorical_name(name)
        
        msg_init = 'CategoricalFeature.__init__({})'.format(name)
        
        self.cat2label = None
        self.label2cat = None
        if (cat2label is not None):
            self.printers[9](msg_init + ': initializing "cat2label"')
            if (set(self.values) != set(cat2label.values())) or apply_mapping:
                self.printers[9](msg_init + ': applying mapping "cat2label" to values')
                self.values = np.array(list(map(lambda cat: cat2label[cat], self.values)))
            self._check_cat2label(self.values, self.name, cat2label)
            self.cat2label = copy.deepcopy(cat2label)
            self.label2cat = {label:cat for cat, label in cat2label.items()}
            
        self.properties = {}
        self.properties['is_numeric'] = self._is_numeric(self.values, self.name)
        self.properties['is_label_encoded'] = self._is_label_encoded(self.values, self.name)
        if self.properties['is_label_encoded']:
            self.printers[9](msg_init + ': feature {} already label encoded'.format(name))
            self.name = self._get_label_encoded_name(self.name)
        else:
            self.printers[9](msg_init + ': label encoding feature {}'.format(name))
            self._label_encode()
        self.properties['is_constant'] = self._is_constant(self.values, self.name)
  
        # These values are used for filtering rare values
        self.threshold = None
        self.unique_label = None
            
        # Label encoding. По существу нет никакого смысла держать в незакодированном виде
        assert self.properties['is_label_encoded']
        assert self.properties['is_numeric']
        
    ##################################################################################
        
    def get_cat_values(self):
        return np.array(list(map(lambda label: self.label2cat[label], self.values)))
        
    ##################################################################################
        
    def _filter_feature(self, threshold):
        # Checking if the feature is label encoded
        if not self.properties['is_label_encoded']:
            raise ValueError('Cannot apply filtering if feature is not label encoded. Run \"filter_values\" first')
        self.name = self._get_filtered_name(self.name, threshold)
        
        # Checking if there are rare values present in the feature
        counts = Counter(self.values)
        for label in counts:
            if counts[label] <= threshold:
                self.unique_label = self.values.max() + 1
                self.threshold = threshold
                break
        if self.unique_label is None: 
            # There are no labels which occur less or equal to threshold times
            return
        
        # Some features occur less or equal threshold times. Let us find them
        rare_labels = set()
        rare_cats = set()
        # Changing rare labels to the chosen unique_label
        for n, label in enumerate(self.values):
            if counts[label] <= threshold:
                if self.cat2label is not None:
                    rare_labels.add(label)
                    rare_cats.add(self.label2cat[label])
                self.values[n] = self.unique_label
                self.properties['is_label_encoded'] = False
             
        if self.cat2label is not None:    
            if len(rare_cats) > 1:
                new_cat = '(' + '|'.join(sorted(list(rare_cats))) + ')'
            elif len(rare_cats) == 1:
                new_cat = list(rare_cats)[0]
            else:
                assert False, "rare_cats must not be empty at this point!"
                
            for label in rare_labels:
                del self.label2cat[label]
            self.label2cat[self.unique_label] = new_cat
            for cat in rare_cats:
                del self.cat2label[cat]
            self.cat2label[new_cat] = self.unique_label
            
        self.properties['is_constant'] = self._is_constant(self.values, self.name)
        self._label_encode()
        assert self.properties['is_label_encoded']
        assert self.properties['is_numeric']
        
    def get_filtered_feature(self, threshold):
        new_feature = copy.deepcopy(self)
        new_feature._filter_feature(threshold)
        return new_feature
    
    def get_counter_feature(self):
        counts = Counter(self.values)
        new_values = np.zeros(len(self.values))
        for n, value in enumerate(self.values):
            new_values[n] = counts[value]
        new_name = self._get_counter_name(self.name)
        return NumericalFeature(new_values, new_name)

    def get_loo_feature(self, Y_train, n_splits=100, random_state=235, seed=1234, shuffle=True, scale=0.05):
        """
        Предполагает, что первые len(Y_train) примеров принадлежат обучающей выборке
        """
        np.random.seed(seed)
        X_train = self.values[:len(Y_train)]
        new_values_train = np.zeros(len(X_train))

        kfold = StratifiedKFold(n_splits, shuffle=shuffle, random_state=random_state)
        probas = defaultdict(list)
        for train_indices, test_indices in kfold.split(X_train, Y_train):
            x_train, y_train = X_train[train_indices], Y_train[train_indices]
            x_test, y_test = X_train[test_indices], Y_train[test_indices]
            values = set(x_train)
            for value in values:
                mask = (x_test == value)
                new_values_train[test_indices[mask]] = np.mean(y_train[x_train == value]) 
                
        X_test = self.values[len(Y_train):]
        new_values_test = np.zeros(len(X_test))
        for value in set(X_test):
            new_values_test[X_test == value] = np.mean(Y_train[X_train == value])
        
        new_values = np.concatenate([new_values_train, new_values_test]) * \
                     np.random.normal(loc=1.0, scale=scale, size=len(self.values))
        new_name = self._get_loo_name(self.name)
        return NumericFeature(new_values, new_name)
        
    ############################################################
    ##                       Кодировщики                      ##
    ############################################################
    def _label_encode(self):
        if self.properties['is_label_encoded']:
            if not self.name.startswith(self.LE_PREFIX):
                self.name = self._get_label_encoded_name(self.name)
            return
        
        label_encoder = LabelEncoder()
        self.values = label_encoder.fit_transform(self.values)
        classes = label_encoder.classes_
        old_label2new_label = {cat:label for label, cat in enumerate(classes)}
        new_label2old_label = {label:cat for label, cat in enumerate(classes)}

        self.name = self._get_label_encoded_name(self.name)
        self.properties['is_label_encoded'] = self._is_label_encoded(self.values, self.name)
        self.properties['is_numeric'] = self._is_numeric(self.values, self.name)
        self.properties['is_constant'] = self._is_constant(self.values, self.name)
        
        if self.unique_label is not None:
            self.unique_label = old_label2new_label[self.unique_label]
            assert self.unique_label == len(old_label2new_label) - 1
            assert (self.FIL_PREFIX + '{}_'.format(self.threshold)) in self.name
            
        if self.label2cat is None:
            self.cat2label = old_label2new_label
            self.label2cat = new_label2old_label
        else:
            new_label2cat = {}
            for old_label in self.label2cat:
                new_label = old_label2new_label[old_label]
                new_label2cat[new_label] = self.label2cat[old_label]
            cat2new_label = {cat:new_label for new_label, cat in new_label2cat.items()}
            self.cat2label = cat2new_label
            self.label2cat = new_label2cat
            
        assert self.properties['is_label_encoded']
        assert self.properties['is_numeric']

    def get_label_encoded_feature(self):
        new_feature = copy.deepcopy(self)
        new_feature._label_encode()
        return new_feature
    
    def get_one_hot_encoded_feature(self, sparse=True, omit_uniques=False):
        assert self.properties['is_label_encoded']
        assert self.properties['is_numeric']
        
        if (not omit_uniques) or (self.unique_label is None):
            unique_label = -1
        else:
            unique_label = self.unique_label
        
        ohe_name = self._get_ohe_name(self.name)
        counter = Counter(self.values)
        
        if self.properties['is_constant']: 
            assert np.sum(self.values) == 0
            assert len(self.cat2label) == 1
            assert list(self.cat2label.values())[0] == 0
            self.printers[2]('CategoricalFeature[{}]: OHE of constant feature.'.format(self.name))
            return NumericalFeature(self.values, ohe_name)
        
        if (len(counter) == 2):
            assert set(self.cat2label.values()) == set([0, 1])
            if unique_label >= 0:
                assert unique_label == 1
                self.printers[2]('CategoricalFeature[{}]: omiting unique label turns to constant.'.format(self.name))
                return NumericalFeature(np.zeros(len(self.values)), ohe_name)
            else:
                self.printers[2]('CategoricalFeature[{}]: OHE senseless for binary feature.'.format(self.name))
                return NumericalFeature(self.values, ohe_name)
        
        ohe_values = OneHotEncoder(sparse=sparse).fit_transform(self.values[:, np.newaxis])
        if sparse:
            ohe_values = ohe_values.tocsc()
        if unique_label >= 0:
            assert unique_label == len(counter) - 1
            mask = (self.values == unique_label)
            if sparse:
                last_column = ohe_values[:, unique_label].toarray().flatten()
            else:
                last_column = ohe_values[:, unique_label]
            assert np.all(last_column == mask)
            ohe_values = ohe_values[:, :unique_label]       
        
        feature_names = []
        feature_values = []
        for label in sorted(self.label2cat.keys()):
            if label != unique_label:
                feature_names.append(self.label2cat[label])
                feature_values.append(ohe_values[:, label])
        features = [NumericalFeature(fvalues, fname) for fvalues, fname in zip(feature_values, feature_names)]
        return AggregatedFeature(features, ohe_name)

