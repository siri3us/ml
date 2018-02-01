# -*- coding: utf-8 -*-

import numpy as np

class Initializer:
    def __init__(self):
        pass
    
class InitializerBase:
    def __init__(self, dtype=np.float64):
        self.dtype = dtype
    def __call__(self, shape, *args, **kwargs):
        assert False
    def _check_shape(self, shape):
        if not isinstance(shape, tuple):
            raise TypeError('When provided to initializer "shape" must be a tuple of integers.')
        # TODO: check positive integers
            

class RandomInitializerBase(InitializerBase):
    def __init__(self, generator=None, seed=None, dtype=np.float64):
        super().__init__(dtype=dtype)
        self.set_generator(generator=generator, seed=seed)
    def set_generator(self, generator, seed):
        if generator is None:
            generator = np.random.RandomState(seed)
        assert isinstance(generator, np.random.RandomState)  
        self.generator = generator
        

class ConstantInitializer(InitializerBase):
    def __init__(self, value=None, dtype=np.float64):
        super().__init__(dtype=dtype)
        if value is not None:
            if not isinstance(value, np.ndarray):
                raise TypeError('When provided to the ConstantInitializer constructor "value" must be a numpy array.')
        self.value = value
    def __call__(self, shape=None, *args, **kwargs):
        """
        Либо shape, либо начальное значение value должны быть заданы.
        """
        if shape is not None:
            self._check_shape(shape)
        if (self.value is None) & (shape is None):
            raise ValueError('Either "value" or "shape" must be available when calling ConstantInitializer.')
        if self.value is None:
            return np.zeros(shape, dtype=self.dtype)
        if shape is None:
            return self.value.astype(dtype=self.dtype, copy=False)
        if shape != self.value.shape:
            raise ValueError('Inconsistent requested shape and the initial value shape: '\
                             'requested shape is {} while the initial value shape is {}'.format(shape, self.value.shape))
        return self.value.astype(dtype=self.dtype, copy=False)

    
class NormalInitializer(RandomInitializerBase):
    def __init__(self, generator=None, seed=None, dtype=np.float64, loc=None, scale=None):
        """
        - generator: если задан, seed не используется; если не задан, seed используется для создания генератора
        - loc: среднее значение нормального распределения
        - scale: стандартное отклонение нормального распределения
        """
        super().__init__(generator=generator, seed=seed, dtype=dtype)
        self.set_loc(loc)
        self.set_scale(scale)
    def set_loc(self, loc=None):
        if loc is None:
            loc = 0.0
        self.loc = loc
    def set_scale(self, scale=None):
        if scale is None:
            scale = 1.0
        self.scale = scale
    def __call__(self, shape):
        """
        Аргументы:
        - shape
        
        Возвращает массив из нормальных случайных значний:
        """
        self._check_shape(shape) 
        return self.generator.normal(loc=self.loc, scale=self.scale, size=shape).astype(self.dtype, copy=False)
   

# TODO rewrite
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



