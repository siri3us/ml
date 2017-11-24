import numpy as np
import pandas as pd
import scipy
import copy
from collections import Counter, defaultdict

from ..core.helpers import Printer, Checker

class FeaturesStorage(object):
    def __init__(self, verbose):
        self.features = {}
        self.types = {}
        self.n_samples = None
        
        self.verbose = verbose
        self.printers = {}
        for v in range(10):
            self.printers[v] = Printer(v, self)
 
    ###################################################################
    def is_present(self, name):
        return name in self.features
    
    def check_if_present_(self, name):
        if not self.is_present(name):
            raise ValueError("FeaturesHandler: unknown feature \"{}\"".format(name))  
            
    def check_type_(self, name, TYPE):
        if TYPE != self.types[name]:
            raise ValueError('Expected type {} but got {}'.format(TYPE, self.types[name]))
        
    def is_binary(self, values):
        if len(Counter(values)) == 2:
            return True
        return False
    
    def check_feature_size_(self, feature):
        if (self.n_samples is not None) & (len(feature) != self.n_samples):
            raise ValueError("Given feature vector has size {} while must have size {}".format(
                        len(values), self.n_samples))
        if len(feature.shape) != 1:
            raise ValueError("Given feature vector must have shape of type (size, ), but has {}".format(
                feature.shape))


    ################################################################### 
    def set_feature(self, feature):
        self.check_feature_size_(feature)
        self.del_feature(feature)  
        self.n_samples = len(feature)
        self.features[feature.name] = copy.deepcopy(feature)
        if isinstance(feature, CategorialFeature):
            self.types[feature.name] = 'CAT'
        elif isinstance(feature, NumericFeature):
            self.types[feature.name] = 'NUM'
        else:
            assert False
            
    def get_feature(self, feature_name):
        self.check_if_present_(feature_name)
        return copy.deepcopy(self.features[feature_name]) 

    def del_feature(self, feature_name):
        if feature_name not in self.features:
            return False
        del self.features[feature_name]
        del self.types[feature_name]
        if len(self.features) == 0:
            self.n_samples = None
        return True
    
    def get_list_of_features(self, TYPE=None):
        if TYPE is None:
            return sorted(list(self.features.keys()))
        names = []
        for name in self.types:
            if self.types[name] == TYPE:
                names.append(name)
        return sorted(names)
    
    def __contains__(self, feature_name):
        return feature_name in self.features
    
    ###################################################################
    #    Функции комбинирования категориальных признаков              #
    ###################################################################
    def get_all_combinations(self, feature_names, degree, hash=hash, store=True):
        self.printers[2]('get_all_combinations_of({}, degree={}, store={})'.format(feature_names, degree, store))
        combined_features = {}
        for names in combinations(feature_names, degree):
            new_feature = self.get_combined_feature(names, hash, store) 
            combined_features[new_feature.name] = new_feature
        return combined_features

    def get_combined_feature(self, feature_names, hash=hash, store=True):
        if len(feature_names) < 1:
            raise ValueError('At least one feature name must be given')
        for name in feature_names:
            self.check_if_present_(name)
            self.check_type_(name, 'CAT')
        
        features = [self.features[name] for name in feature_names]
        new_feature = CategorialCombiner().get_combined_feature(features, hash=hash)
        if store:
            self.features[new_feature.name] = new_feature
            self.types[new_feature.name] = 'CAT'
        return copy.deepcopy(new_feature)

    ############################################################
    ##       Сборка итогового признакового представления      ##
    ############################################################
    def assemble_data_frame(self, feature_names):
        feature_values = []
        for feature_name in feature_names:
            self.check_if_present_(feature_name)
            feature_values.append(self.features[feature_name].values)
        return pd.DataFrame(np.vstack(feature_values).transpose(), columns=feature_names)
    
    def assemble(self, feature_names, feature_map=defaultdict(lambda x: ['def']), sparse=False):
        self.X = []
        feature_map = copy.deepcopy(feature_map)
        for name in feature_names:
            if name not in feature_map:
                feature_map[name] = ['def']
            for enc_type in feature_map[name]:
                encoded_values = self.features[name].get_values(format=enc_type, sparse=sparse)
                if encoded_values is None:
                    print("CategorialFeaturesHandeler.assemble(): feature {} is empty - omitting this feature".format(name))
                    continue
                assert len(encoded_values.shape) == 2, "Returned 1D vector, while 2D matrix have to be."
                self.X.append(encoded_values)
        if sparse:
            return scipy.sparse.hstack(self.X)
        return np.concatenate(self.X, axis=1)

    
    def add_categorized(self, name, bins, right=True):
        self.check_if_present_(name)
        self.check_type_(name, 'NUM')
        new_feature = self.features[name].get_categorized_feature(bins, right)
        self.features[new_feature.name] = new_feature
        self.types[new_feature.name] = 'CAT'
        return new_feature.name
    
    def add_filtered(self, name, threshold, applied_ohe=True):
        self.check_if_present_(name)
        self.check_type_(name, 'CAT')
        new_feature = self.features[name].get_filtered_feature(threshold, applied_ohe)
        self.features[new_feature.name] = new_feature
        self.types[new_feature.name] = 'CAT'
        return new_feature.name
        
    def add_counter(self, name):
        self.check_if_present_(name)
        self.check_type_(name, 'CAT')
        new_feature = self.features[name].get_counter_feature()
        self.features[new_feature.name] = new_feature
        self.types[new_feature.name] = 'CAT'
        return new_feature.name
    
    def add_loo(self, name, y_train, n_splits=100, random_state=235, seed=1234):
        self.check_if_present_(name)
        self.check_type_(name, 'CAT')
        new_feature = self.features[name].get_loo_feature(y_train, n_splits, random_state, seed)
        self.features[new_feature.name] = new_feature
        self.types[new_feature.name] = 'NUM'
        return new_feature.name

    def label_encode(self, name):
        self.check_if_present_(name)
        self.check_type_(name, 'CAT')        
        self.features[name].label_encode()
    
    
test = False
if test:      
    FStest = FeaturesStorage(verbose=0)
    features = [CategorialFeature(np.array([0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4]), 'f1'),
                CategorialFeature(np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]), 'f2'),
                CategorialFeature(np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]), 'f3'),
                NumericFeature(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]), 'f4')]
    for feature in features:
        FStest.set_feature(feature)
        assert feature.shape == (14, )
        assert len(feature) == 14

    #temp = features_storage.get_feature('f5')
    print('\nLists of features:')
    names = FStest.get_list_of_features()
    print('all:', names)
    print(FStest.assemble_data_frame(names))
    print('f1' in features_storage)
    print('f5' in features_storage)
    
    print('\nDeleting and setting features:')
    FStest.del_feature('f1')
    print('del f1:', FStest.get_list_of_features()) 
    FStest.del_feature('f4')
    print('del f4:', FStest.get_list_of_features())
    FStest.del_feature('f3')
    print('del f3:', FStest.get_list_of_features())
    FStest.del_feature('f2')
    print('del f2:', FStest.get_list_of_features())
    
    for feature in features:
        FStest.set_feature(feature)    
    
    num_feature = FStest.get_feature('f4')
    new_feature = num_feature.get_categorized_feature(np.linspace(0, 2, 3), right=False)
    FStest.set_feature(new_feature)
    print(new_feature, new_feature.values)
    
    print('\nCombining features:')
    new_feature = FStest.get_combined_feature(['f1'])
    print(new_feature, new_feature.values)
    new_feature = FStest.get_combined_feature(['f2'])
    print(new_feature, new_feature.values)
    new_feature = FStest.get_combined_feature(['f3'])
    print(new_feature, new_feature.values)
    new_feature = FStest.get_combined_feature(['f1', 'f2'])
    print(new_feature, new_feature.values)
    new_feature = FStest.get_combined_feature(['f1', 'f3'])
    print(new_feature, new_feature.values)
    new_feature = FStest.get_combined_feature(['f2', 'f3'])
    print(new_feature, new_feature.values)
    new_feature = FStest.get_combined_feature(['f1', 'f2', 'f3'])
    print(new_feature, new_feature.values)
    print('all:', FStest.get_list_of_features())
    
    print('\nFiltering values:')
    thr = 2
    FStest.add_filtered('f1', thr, True)
    FStest.add_filtered('f2', thr, False)
    FStest.add_filtered('f3', thr, False)
    print('all:', FStest.get_list_of_features())
    feature = FStest.get_feature('FA{}_f1'.format(thr))
    print(feature.name, feature.values)
    feature = FStest.get_feature('FN{}_f2'.format(thr))
    print(feature.name, feature.values)
    feature = FStest.get_feature('FN{}_f3'.format(thr))
    print(feature.name, feature.values)
    
    print('\nObtaining counters:')
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
                          sparse=False))
