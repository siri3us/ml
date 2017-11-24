class AggregatedFeature(FeatureBase):
    def __init__(self, features, name, exclude_const=True):
        assert all([isinstance(feature, FeatureBase) for feature in features])
        self.name = name
        self.init_feature_names = [feature.name for feature in features]
        self.features = {feature.name: copy.deepcopy(feature) for feature in features}
        if exclude_const:
            self.exclude_constant()
    
    def exclude_constant(self):
        to_delete = []
        for feature_name in self.features:
            if self.features[feature_name].is_constant():
                to_delete.append(feature_name)
        for feature_name in to_delete:
            print('Deleting constant feature "{}"'.format(feature_name))
            del self.features[feature_name]

    def is_constant(self):
        if len(self.features) == 0: # In case if all feature are excluded due to constant values
            return True
        return all([feature.is_constant() for feature in self.features.values()])
            
    def get_values(self, feature_names=None, **kwargs):
        sparse = kwargs.setdefault('sparse', True)
        if feature_names is None:
            features = [self.features[feature_name] for feature_name in self.init_feature_names 
                        if feature_name in self.features.keys()]
        else:
            features = [self.features[feature_name] for feature_name in feature_names]
        X = []
        for feature in features:
            X.append(feature.get_values(sparse=sparse))
        if sparse:
            X = scipy.sparse.hstack(X)
        else:
            X = np.concatenate(X, axis=1)
        return X
    
    def __repr__(self):
        s = 'AggregatedFeature['
        for feature_name in [feature_name for feature_name in self.init_feature_names 
                             if feature_name in self.features]:
            feature = self.features[feature_name]
            s += feature.__repr__() 
        s += ']'
        return s
