class CategorialCombiner:
    def __init__(self):
        pass

    def get_all_combinations(self, features, degree, hash=hash):
        feature_names = [feature.name for feature in features]
        print('get_all_combinations_of({}, degree={})'.format(feature_names, degree))
        combined_features = {}
        for some_features in combinations(features, degree):
            new_feature = self.get_combined_feature(some_features, hash) 
            combined_features[new_feature.name] = new_feature
        return combined_features

    def get_combined_feature(self, features, hash=hash):
        self.check_sizes_(features)
        if len(features) < 1:
            raise ValueError('At least one feature name must be given')
        if len(features) == 1:
            return copy.deepcopy(features[0])
                             
        feature_values = []
        feature_names = []
        for feature in features:
            if feature.values is None:
                raise ValueError('One of features is None')
            feature_values.append(feature.values)
            feature_names.append(feature.name)
            
        new_values = []
        for hyper_value in zip(*feature_values):
            new_values.append(hash(hyper_value))
        new_values = LabelEncoder().fit_transform((new_values))
        new_name = '+'.join(feature_names)
        return CategorialFeature(new_values, new_name)

    def check_sizes_(self, features):
        if len(Counter([len(feature) for feature in features])) != 1:
            raise ValueError('Features must have equal sizes!')
            
test = False
if test:
    features = {'f1': [0, 1, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0],
                'f2': [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
                'f3': [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]}
    cat_features = [CategorialFeature(features[name], name) for name in sorted(features.keys())]
    cat_combiner = CategorialCombiner()
    new_feature = cat_combiner.get_combined_feature(cat_features)
    print(new_feature.name, new_feature.values)
    new_feature.filter_values(1)
    print(new_feature.name, new_feature.values)
    new_feature.label_encode()
    print(new_feature.name, new_feature.values)
    new_features = cat_combiner.get_all_combinations(cat_features, degree=1)
    print(new_features)
    new_features = cat_combiner.get_all_combinations(cat_features, degree=2)
    print(new_features)
    new_features = cat_combiner.get_all_combinations(cat_features, degree=3)
    print(new_features)
