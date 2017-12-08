# -*- coding: utf-8 -*-

import numpy as np
import unittest
import numbers
import copy
import scipy
from scipy.sparse import csr_matrix, csc_matrix
from collections import Counter
    
import sys
sys.path.append('../')
from ml.feature import *
        
n_features = 5
features = []
feature_names = []
size = 10
as_dataframe=True
np.random.seed(1)
for n_feature in range(n_features):
    values = np.random.randint(low=0, high=2, size=size)
    features.append(NumericalFeature(values, 'f' + str(n_feature), verbose=0))
    feature_names.append(features[-1].get_name())
features.append(NumericalFeature(np.ones(size, dtype=np.int32), name='f' + str(n_features)))

aggr_feature = AggregatedFeature(features, 'AGGR', copy=False, treat_const='none')
print('AGGREGATED_FEATURE:\n{}\n'.format(aggr_feature))

sparse = True
values = aggr_feature.get_values(sparse=sparse, as_dataframe=as_dataframe)
print('VALUES with sparse = {}:\n{}\n'.format(sparse, values))

sparse=False
values = aggr_feature.get_values(sparse=sparse, as_dataframe=as_dataframe)
print('VALUES with sparse = {}:\n{}\n'.format(sparse, values))

fnames = feature_names[1:4]
sparse = False
print('VALUES for features {}'.format(fnames))
values = aggr_feature.get_values(feature_names=fnames, sparse=sparse, as_dataframe=as_dataframe) 
print(values, '\n')
