# -*- coding: utf-8 -*-

import numpy as np
from .initializers import *

class TensorShape:
    def __init__(self, shape=None):
        self.set_shape(shape)
    def set_shape(self, shape):
        self.initialized = False
        if shape is None:
            self.shape = None
            self.initialized = False
        elif isinstance(shape, (tuple, list)):
            self.shape = tuple(shape)
            self.initialized = True
        elif isinstance(shape, TensorShape):
            self.shape = shape.get_shape() # так как хранится в immutable tuple-ах, то проблем быть не должно
            self.initialized = True
        else:
            self.initialized = False
            raise TypeError('Unacceptable type "{}" of "shape"'.format(type(shape)))
    def get_shape(self):
        return self.shape
    def is_initialized(self):
        return self.initialized
    def __repr__(self):
        return type(self).__name__ + '(' + repr(self.shape) + ')'
    def __len__(self):
        return len(self.shape)
    def __getitem__(self, n):
        return self.shape[n]
    def __iter__(self):
        return iter(self.shape)
    def __eq__(self, other_shape):
        other_shape = TensorShape(other_shape)
        # Если хотя бы одна из форм не инициализована, то результат сравнения False
        if (not self.is_initialized()) | (not other_shape.is_initialized()):
            return False
        if len(self.shape) != len(other_shape):
            return False
        for l, r in zip(self.shape, other_shape):
            if (l == -1) | (r == -1):
                continue
            if l != r:
                return False
        return True
    def __neq__(self, other_shape):
        return not self == outher_shape


class Tensor:
    """
    Прокси класс для передачи доступа к параметрам
    
    
    """
    def __init__(self, shape=None, init_value=None, initializer=None, dtype=np.float64, name=None):
        """
        - shape: tuple, list или TensorShape
        - value: numpy array
        """
        self.initialized = False
        shape = TensorShape(shape)
        if init_value is not None:
            assert isinstance(init_value, np.ndarray)

        if shape.is_initialized() & (init_value is not None):
            init_value_shape = TensorShape(init_value.shape)
            if shape != init_value_shape:
                raise ValueError('Inconsistent parameters "shape" and "init_value" parameters '\
                                 'are passed to the {} constructor "{}": {} != {}.'.format(
                                 type(self).__name__, shape, init_value_shape))
            self.shape = init_value_shape  
            self.initializer = ConstantInitializer(value=init_value, dtype=dtype)
        elif shape.is_initialized() & (init_value is None):
            if initializer is None:
                # Используем инициализатор по умолчанию (плохая практика)
                initializer = NormalInitializer(dtype=dtype)
            if not isinstance(initializer, InitializerBase):
                raise TypeError('Parameter "initializer" passed to the {} constructor '\
                                'must have type Initializer.'.format(type(self).__name__))
            self.shape = shape
            self.initializer = initializer
        elif (not self.shape.is_initialized()) & (init_value is not None):
            self.shape = TensorShape(init_value.shape)
            self.initializer = ConstantInitializer(value=init_value, dtype=dtype)
        else:
            raise ValueError('Either "shape" or "init_value" must be provided to the {} constructor.'.format(type(self).__name__))
        self.name = name
        self.dtype = dtype
        assert isinstance(self.initializer, InitializerBase)
        assert self.shape.is_initialized()
        for dim_size in self.shape:
            assert dim_size > 0, 'All "shape" dimensions must be positive integers.'

    def __repr__(self):
        if self.initialized:
            return 'Tensor[shape={}, value=\n{}]'.format(self.shape, self.value)
        return 'Tensor[shape={}, value=None]'.format(self.shape)
    
    def initialize(self):
        """
        Инициализация аргумента value данного тензора.
        Для возможности инициализации необходимо выполнение одного из следующих условий:
        - задано начальное значение value тензора; доложно быть совместимо с параметром shape
        - задано значение shape формы тензора
        """
        self.value = self.initializer(shape=self.shape.get_shape())
        self.initialized = True
    def get_value(self):
        return self.value


class Placeholder:
    def __init__(self, shape):
    
    def get_value():
    
    def set_value(self, value):
        assert self.shape == TensorShape(value.shape)
    
    
        
class TensorConstantInitializer(ConstantInitializer):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
    def __call__(self, *args, **kwargs):
        tensor_value = super().__call__(*args, **kwargs)
        return Tensor(shape=tensor_value.shape, init_value=tensor_value, name=self.name,
                      dtype=self.dtype)

    
class TensorNormalInitializer(NormalInitializer):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
    def __call__(self, *args, **kwargs):
        tensor_value = super().__call__(*args, **kwargs)
        return Tensor(shape=tensor_value.shape, init_value=tensor_value, name=self.name,
                      dtype=self.dtype)
