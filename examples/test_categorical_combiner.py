# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
sys.path.append('../')

from ml.feature import *
from ml.utils import print_columns

test = True
if test:
    features = {'f1': [0, 1, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0],
                'f2': [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
                'f3': [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                'f4': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    cat_features = [CategoricalFeature(features[name], name) for name in sorted(features.keys())]
    cat_combiner = CategoricalCombiner(treat_const='delete')
    new_feature = cat_combiner.get_combined_feature(cat_features)
    print('comb_feature:', new_feature)
    print('name = {}, values = {}'.format(new_feature._name, new_feature._values))
    fil_feature = new_feature.get_filtered_feature(1)
    print('fil_feature: ', fil_feature)
    print('name = {}, values = {}'.format(fil_feature.name, fil_feature._values))
    for degree in [1, 2, 3, None]:
        print('\ndegree = {}'.format(degree))
        new_features = cat_combiner.get_all_combinations(cat_features, degree=degree)
        print('new_features:', new_features)
        args = []
        for f_name, feature in new_features.items():
            args.append(('  ' + f_name, feature.get_values(False).flatten()))
        args = sorted(args, key=lambda x: x[0])
        print_columns(*args)
