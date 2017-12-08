# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from ml.feature import *


test = True
if test:      
    manager = CategoricalFeaturesManager(verbose=0)
    features = [CategoricalFeature(np.array([0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4]), 'f1'),
                CategoricalFeature(np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]), 'f2'),
                CategoricalFeature(np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]), 'f3')]
    for feature in features:
        manager.set_feature(feature)
        assert feature.shape == (14, )
        assert len(feature) == 14
    fnames = manager.get_list_of_features()
    print('SET UP: {}\n'.format(fnames))
    
    for fname in fnames:
        print('\t{} in manager = {}'.format(fname, fname in manager))
    
    print()
    df = manager.assemble_data_frame(fnames)
    print(df)
    
    print('\nDELETING AND SETTING FEATURES:')
    manager.del_feature('f1')
    print('del f1:', manager.get_list_of_features()) 
    manager.del_feature('f2')
    print('del f2:', manager.get_list_of_features())
    manager.del_feature('f3')
    print('del f3:', manager.get_list_of_features())

    for feature in features:
        manager.set_feature(feature)    
        print('set {}:'.format(feature.get_name()), manager.get_list_of_features())
    
    print('\nCOMBINING FEATURES:')
    new_feature = manager.get_combined_feature(['f1'])
    print(new_feature, new_feature.values.flatten())
    new_feature = manager.get_combined_feature(['f2'])
    print(new_feature, new_feature.values.flatten())
    new_feature = manager.get_combined_feature(['f3'])
    print(new_feature, new_feature.values.flatten())
    new_feature = manager.get_combined_feature(['f1', 'f2'])
    print(new_feature, new_feature.values.flatten())
    new_feature = manager.get_combined_feature(['f1', 'f3'])
    print(new_feature, new_feature.values.flatten())
    new_feature = manager.get_combined_feature(['f2', 'f3'])
    print(new_feature, new_feature.values.flatten())
    new_feature = manager.get_combined_feature(['f1', 'f2', 'f3'])
    print(new_feature, new_feature.values.flatten())
    print('all:', manager.get_list_of_features())
    
    print('\nFILTERED FEATURES:')
    print('\tbefore filtration')
    print('feature names:', manager.get_list_of_features())
    df = manager.assemble_data_frame()
    print(df)
    thr = 2
    manager.filter_features(threshold=2)
    #manager.add_filtered('f1', thr)
    #manager.add_filtered('f2', thr)
    #manager.add_filtered('f3', thr)
    print('\tafter filtration')
    print('feature names:', manager.get_list_of_features())
    df = manager.assemble_data_frame()
    print(df)
    
    manager = CategoricalFeaturesManager(features)

    """print('\nObtaining counters:')
    for name in ['f1', 'f2', 'f3']:
        FStest.add_counter(name)
        feature = FStest.get_feature('CTR_' + name)
        print(feature.name, feature.values)
    print('all:', FStest.get_list_of_features())
    print('cat:', FStest.get_list_of_features('CAT'))
    print('num:', FStest.get_list_of_features('NUM'))
    print('\nAssembling features')
    print(FStest.assemble(['f1', 'f2', 'f3', 'CTR_f3', 'f1+f2', 'f1+f3', 'f2+f3', 'f1+f2+f3', 'f4'], sparse=False))
    print(FStest.assemble(['FA2_f1', 'f2', 'f3', 'f1+f2', 'f1+f3', 'f2+f3'], 
                          {'FA2_f1': ['def', 'ohe'],
                           'f2': ['def', 'ohe'],
                           'f3': ['def', 'ohe']},
                          sparse=False))"""
