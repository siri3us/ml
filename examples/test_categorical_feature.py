# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
sys.path.append('../')

from ml.feature import *
from ml.utils import print_columns

name = 'f'
cat_values = ['A', 'A', 'B', 'A', 'B', 'C', 'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'E']
values     = [0,   0,    1,   0,   1,   2,   0,   1,   2,   3,   0,   1,   2,   3,   4]
cat2label  = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
f = CategoricalFeature(cat_values, name, cat2label, verbose=0)

test_initial = True
test_counter = True
test_filtered = True
test_ohe = True
test_ohe_filtered = True
test_loo = True

if test_initial:
    print('INITIAL FEATURE:')
    print('True values    : ', values)
    print('Obtained values: ', list(f.get_values().flatten()))
    print('\nTrue CAT values: ', cat_values)
    print('Obtained CATs  : ', list(f.get_cat_values()))

if test_counter:
    print('\n\nCOUNTER FEATURE')
    print('Initial feature: ', f)
    counter_f = f.get_counter_feature()
    print('Counter feature: ', counter_f)
    print(counter_f.get_values().flatten())
    print('Values of initial and counter features:')
    args = [('initial', f.get_values().flatten()), ('counter', counter_f.get_values().flatten())]
    print_columns(*args)

if test_filtered:
    print('\n\nFILTERED FEATURES')
    ffs = {n:f.get_filtered_feature(n) for n in range(6)}
    for n in range(6):
        fil_feature = ffs[n].get_values().flatten()
        cat_feature = ffs[n].get_cat_values().flatten()
        ctr_feature = ffs[n].get_counter_feature().get_values().flatten()
        print('fil_feature props: ', ffs[n].get_properties())
        print('fil feature name: ', ffs[n])
        print('ctr feature name: ', ffs[n].get_counter_feature())
        args = [('fil_feature', fil_feature), ('cat_feature', cat_feature), ('ctr_feature', ctr_feature)]
        print_columns(*args)
        print('\n\n')

if test_ohe:   
    print('\n\nOHE FEATURES')
    ohe_feature = f.get_ohe_feature()
    a = f.get_ohe_feature(sparse=False).get_values(sparse=True).toarray()
    b = f.get_ohe_feature(sparse=False).get_values(sparse=False)
    c = f.get_ohe_feature(sparse=True).get_values(sparse=False)
    d = f.get_ohe_feature(sparse=True).get_values(sparse=True).toarray()
    assert np.allclose(a, b)
    assert np.allclose(b, c)
    assert np.allclose(c, d)
    assert np.allclose(d, a)
    print('Initial feature:', ohe_feature)
    print('\tvalues:\n', f.get_values().flatten())
    print('OHE feature:', ohe_feature)
    print('\tOHE values:\n', a)

if test_ohe_filtered:
    print('\n\nOHE FEATURES + FILTRATION')
    for threshold, omit_uniques in product([1, 2, 3, 4, 5], [False, True]):
        ff = f.get_filtered_feature(threshold=threshold)
        print('OHE feature with omit_uniques = {} and threshold = {}'.format(omit_uniques, threshold))
        print('Initial  feature name:', f)
        print('Filtered feature name:', ff)
        print('Initial feature values:', f.get_values().flatten())
        print('Filtered fature values:', ff.get_values().flatten())
        print('threhold =', ff._threshold, '  unique_label =', ff._unique_label)
        ff_ohe = ff.get_ohe_feature(omit_uniques=omit_uniques)
        print('FilOHE feature name:  ', ff_ohe)
        print('FilOHE feature values:\n', ff_ohe.get_values(sparse=False))
        print('FilOHE is constant:   ', ff_ohe.is_constant())
        print('\n\n')

if test_loo:
    print('\n\nLEAVE ONE OUT')
    X = np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0])
    y = np.array([0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0])
    cat_feature = CategoricalFeature(X, 'f')
    random_state = 345
    n_splits = 2
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    print_columns(('ind', np.arange(len(X))), ('X', X), ('y', y))
    for n_split, (train_indices, test_indices) in enumerate(cv.split(X, y)):
        print('\n\nn_split =', n_split)
        X_tr, y_tr = X[train_indices], y[train_indices]
        X_ts, y_ts = X[test_indices],  y[test_indices]
        X_loo_true = np.array([1/4., 2/3., 0, 2/3., 0, 2/3., 2/3., 2/3., 2/3., 0, 0, 1/4.])
        print()
        print_columns(('ind', train_indices), ('X_tr', X_tr), ('y_tr', y_tr))
        print()
        print_columns(('ind', test_indices), ('X_ts', X_ts), ('y_ts', y_ts))
        
    loo_feature = cat_feature.get_loo_feature(y, cv, alpha=0, scale=0.0)
    X_loo_found = loo_feature.get_values(sparse=False).flatten()
    print('LOO:\n')
    print_columns(('True', X_loo_true), ('Found', X_loo_found))
    np.allclose(X_loo_found, X_loo_true)
    #print(cat_feature, cat_feature)
    #print(loo_feature, loo_feature.values
