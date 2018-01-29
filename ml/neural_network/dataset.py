# -*- coding: utf-8 -*-
from builtins import range
import os, json
import numpy as np
import h5py
from collections import OrderedDict
from sklearn.utils import resample

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
        
        
class CocoDataset:
    def __init__(self, coco_dir, n_train_samples=None, pca_features=True, seed=None, verbose=False):
        self.gen = np.random.RandomState(seed)
        self.dir = coco_dir
        
        data = OrderedDict()
        caption_file = os.path.join(self.dir, 'coco2014_captions.h5')
        with h5py.File(caption_file, 'r') as f:
            for k, v in f.items():
                data[k] = np.asarray(v)

        if pca_features:
            train_feat_file = os.path.join(self.dir, 'train2014_vgg16_fc7_pca.h5')
        else:
            train_feat_file = os.path.join(self.dir, 'train2014_vgg16_fc7.h5')
        with h5py.File(train_feat_file, 'r') as f:
            data['train_features'] = np.asarray(f['features'])

        if pca_features:
            val_feat_file = os.path.join(self.dir, 'val2014_vgg16_fc7_pca.h5')
        else:
            val_feat_file = os.path.join(self.dir, 'val2014_vgg16_fc7.h5')
        with h5py.File(val_feat_file, 'r') as f:
            data['val_features'] = np.asarray(f['features'])

        dict_file = os.path.join(self.dir, 'coco2014_vocab.json')
        with open(dict_file, 'r') as f:
            dict_data = json.load(f)
            for k, v in dict_data.items():
                data[k] = v

        train_url_file = os.path.join(self.dir, 'train2014_urls.txt')
        with open(train_url_file, 'r') as f:
            train_urls = np.asarray([line.strip() for line in f])
        data['train_urls'] = train_urls

        val_url_file = os.path.join(self.dir, 'val2014_urls.txt')
        with open(val_url_file, 'r') as f:
            val_urls = np.asarray([line.strip() for line in f])
        data['val_urls'] = val_urls

        # Maybe subsample the training data
        if n_train_samples is not None:
            num_train = data['train_captions'].shape[0]
            mask = self.gen.randint(num_train, size=n_train_samples)
            data['train_captions'] = data['train_captions'][mask]
            data['train_image_idxs'] = data['train_image_idxs'][mask]
        
        self.word_to_idx = data.pop('word_to_idx')
        self.idx_to_word = data.pop('idx_to_word')
        self.data = data

        if verbose:
            # Print out all the keys and values from the data dictionary
            for k, v in self.data.items():
                if type(v) == np.ndarray:
                    print(k, type(v), v.shape, v.dtype)
                else:
                    print(k, type(v), len(v))
            print('word_to_idx', len(self.word_to_idx))
            print('idx_to_word', len(self.idx_to_word))

    def sample_minibatch(self, batch_size=100, split='train'):
        split_size = self.data['%s_captions' % split].shape[0]
        mask = self.gen.choice(split_size, batch_size)
        captions = self.data['%s_captions' % split][mask]
        image_idxs = self.data['%s_image_idxs' % split][mask]
        image_features = self.data['%s_features' % split][image_idxs]
        urls = self.data['%s_urls' % split][image_idxs]
        return captions, image_features, urls
    
    def decode_captions(self, captions):
        singleton = False
        if captions.ndim == 1:
            singleton = True
            captions = captions[None]
        decoded = []
        N, T = captions.shape
        for i in range(N):
            words = []
            for t in range(T):
                word = self.idx_to_word[captions[i, t]]
                if word != '<NULL>':
                    words.append(word)
                if word == '<END>':
                    break
            decoded.append(' '.join(words))
        if singleton:
            decoded = decoded[0]
        return decoded
