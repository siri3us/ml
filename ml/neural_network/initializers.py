# -*- coding: utf-8 -*-

import numpy as np


class Initializer:
    def __init__(self):
        pass

    
class DeterministicInitializer(Initializer):
    def __init__(self, init_value):
        self.init_value = init_value
    def __call__(self, shape=None, dtype=np.float64):
        return self.init_value.astype(dtype)


class RandomInitializer(Initializer):
    def __init__(self, seed=None):
        super().__init__()
        self.gen = np.random.RandomState(seed)

        
class ZerosInitializer(Initializer):
    def __init__(self):
        super().__init__()
    def __call__(self, shape=None, dtype=np.float64):
        if shape is None:
            return 0.0
        return np.zeros(shape, dtype=dtype)


class NormalInitializer(RandomInitializer):
    def __init__(self, seed=None):
        super().__init__(seed=seed)
    def __call__(self, shape=None, dtype=np.float64):
        stddev = 1.0
        if len(shape) == 2:
            stddev = 1. / np.sqrt(shape[0])
        if len(shape) == 4:
            stddev = 1.0 / np.sqrt(np.prod(shape[1:]))
        return self.gen.uniform(-stddev, stddev, size=shape).astype(dtype)


class NormalInitializer(RandomInitializer):
    def __init__(self, seed=None):
        super().__init__(seed=seed)
        
    def __call__(self, shape=None, dtype=np.float64):
        stddev = 1.0
        if len(shape) == 2:
            stddev = 1. / np.sqrt(shape[0])
        if len(shape) == 4:
            stddev = 1.0 / np.sqrt(np.prod(shape[1:]))
        return self.gen.normal(loc=0, scale=stddev, size=shape).astype(dtype)
