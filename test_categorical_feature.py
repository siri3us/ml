name = 'f'
cat_values = ['A', 'A', 'B', 'A', 'B', 'C', 'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'E']
values = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4]
cat2label = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
f = CategoricalFeature(cat_values, name, cat2label, verbose=9)
print(list(f.get_values().flatten()))
print(list(f.get_cat_values()))
print(f)
counter_f = f.get_counter_feature()
print(counter_f.get_values().flatten())
print('\n\n\n')
fs = {n:f.get_filtered_feature(n) for n in range(6)}
for n in range(6):
    print(fs[n])
    print(fs[n].get_values().flatten())
    print(fs[n].get_cat_values())
    print(fs[n].properties)
    print()
    
print('\n\n\n')  
print(f.get_one_hot_encoded_feature())
a = f.get_one_hot_encoded_feature(sparse=False).get_values(sparse=True).toarray()
b = f.get_one_hot_encoded_feature(sparse=False).get_values(sparse=False)
c = f.get_one_hot_encoded_feature(sparse=True).get_values(sparse=False)
d = f.get_one_hot_encoded_feature(sparse=True).get_values(sparse=True).toarray()
assert np.allclose(a, b)
assert np.allclose(b, c)
assert np.allclose(c, d)
assert np.allclose(d, a)
print(a)

threshold = 3
omit_uniques = False
ff = f.get_filtered_feature(threshold=threshold)
print(ff)
print('label =', ff.unique_label, ' threhold =', ff.threshold)
print(ff.get_values().flatten())
print(ff)

ff_ohe = ff.get_one_hot_encoded_feature(omit_uniques=omit_uniques)
print(ff_ohe)
print(ff_ohe.get_values(sparse=False))
print('ohe is constant:', ff_ohe.is_constant())

"""    values2 = [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0]
    y2      = np.array([0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0])
    cat_feature2 = CategorialFeature(values2, 'f2')
    loo_feature2 = cat_feature2.get_loo_feature(y2, 2, verbose=True)
    print(cat_feature, cat_feature2)
    print(loo_feature2, loo_feature2.values)"""
