# -*- coding: utf-8 -*-

class Dataset:
    def __init__(self, data, name=None, random_state=None):
        self.name = name
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.data = data
        self.init_random_state = random_state
        self.random_state = random_state
    def get_small_train_dataset(self, n_train_samples):
        return Dataset({
            'X_train': self.X_train[:n_train_samples],
            'y_train': self.y_train[:n_train_samples],
            'X_val': self.X_val,
            'y_val': self.y_val
        })
    def get_batch(self, n_samples):
        from sklearn.utils import resample
        X_batch, y_batch = resample(self.X_train, self.y_train, n_samples=n_samples, replace=False, 
                                    random_state=self.random_state)
        if self.random_state is not None:
            self.random_state += 1
        return X_batch, y_batch
    def __getitem__(self, key):
        return self.data[key]
