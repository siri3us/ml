# -*- coding: utf-8 -*-

import numpy as np

class Initializer:
    def __init__(self):
        pass
    
class ConstInitializer(Initializer):
    def __init__(self, value=None, dtype=np.float64):
        self.value = value
        self.dtype = dtype
    def __call__(self, shape=None):
        if self.value is None:
            if shape is None:
                return np.zeros(shape=(1, 1), dtype=self.dtype)[0, 0]
            return np.zeros(shape=shape, dtype=self.dtype)
        value = np.array(self.value, dtype=self.dtype)
        if shape is None:
            return value
        assert value.shape == shape, 'Passed shape = {} does not match init value shape = {}.'.format(
            shape, value.shape)
        return value

class NormalInitializer(Initializer):
    def __init__(self, generator, dtype=np.float64):
        assert isinstance(generator, np.random.RandomState)
        self.generator = generator
        self.dtype = dtype
    def __call__(self, shape):
        if len(shape) == 2:
            stddev = 1.0 / np.sqrt(shape[0])
        elif len(shape) == 4:
            stddev = 1.0 / np.sqrt(np.prod(shape[1:]))
        else:
            assert False
        return self.generator.normal(loc=0.0, scale=stddev, size=shape).astype(self.dtype, copy=False)

def get_kernel_initializer(init=None, dtype=None, generator=None):
    """
    Возвращает инициализатор для 
    -init: задает начальное значение инициализируемой переменной
        None:        используется нормальная инициализация
        np.ndarray:  константная инициализация заданным значением
        Initializer: передан custom-мный иницализатор
    -dtype
    -generator
    """
    if init is None:
        return NormalInitializer(generator=generator, dtype=dtype)
    if isinstance(init, np.ndarray):
        return ConstInitializer(value=init, dtype=dtype)
    assert isinstance(init, Initializer), '"init" must be of type "Initializer"'
    return init

def get_bias_initializer(init=None, dtype=None, **kwargs):
    """
    Возвращает константный инициализатор для значения вектора смещения
    -init
    -dtype
    -kwargs
    """
    return ConstInitializer(value=init, dtype=dtype)



