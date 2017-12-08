# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import numbers
import copy
from scipy.sparse import csr_matrix, csc_matrix
from collections import Counter
from .exceptions import UnknownFeatureError
from ..core.helpers import Checker, Printer
      
FEATURE_PREFIXES = \
{'CAT': '',
 'NUM': '',
 'LE' : '',    # LabelEncoded feature
 'OHE': 'Ohe', # OneHotEncoded feature
 'CTR': 'Ctr', # Counter feature
 'LOO': 'Loo', # LeaveOneOut feature
 'FIL': 'Fil'}
      
class FeatureKernel:
    """
    FeatureKernel - класс, реализующий общий функционал класса FeatureBase. 
    Является одним из аттрибутов экземпляров класса FeatureBase (SparseFeatureBase и DenseFeatureBase).
    Реализуемые операции включают в себя: 
        1) проверку корректности значений признака 
        2) вывод сообщений о некорректности значений
        3) предобработку и постобработку признаков
        4) получение характеристик признаков (размера, формата и т.п.)
    """
    
    def __init__(self, owner):
        self._owner = owner
        self._printers = owner._printers
        self._info_msg = owner._info_msg
        self._error_msg = owner._error_msg
        self._warning_msg = owner._warning_msg
        self._method_msg = owner._method_msg
        
    def _is_numeric(self, values, name=None):
        """
        Возвращает True, если тип признак числовой. Иначе возвращает False. 
        Данная проверка производится только в конструкторе NumericalFeature.
        Аргументы:
            :param values - значения признака (np.ndarray, csr_matrix, csc_matrix).
            :param name   - имя признака (str).
        """
        return isinstance(values.dtype.type(), numbers.Number)
    
    def _check_numeric(self, values, name=None, throw=True):
        """
        Возвращает True, если тип признак числовой. Иначе возвращает False или вызывает исключение
        (в зависимости от параметра throw).
        Аргументы:
            :param values - значения признака (np.ndarray, csr_matrix, csc_matrix).
            :param name   - имя признака (str).
            :param throw  - если True, то вызывает исключение, если признак не числовой (bool).
        """
        if not self._is_numeric(values):
            if throw:
                error_msg = 'Feature values are not numerical! Their type is {}'.format(values.dtype)
                raise TypeError(self._error_msg(error_msg))
            return False
        return True
    
    def _is_constant(self, values, name=None):
        """
        Если значения признака идентичны для всех объектов, то возвращает True. В противном случае возвращает False. 
        Аргументы:
            :param values - значения признака в виде np.ndarray.
            :param name   - имя признака (str).
        """
        assert isinstance(values, np.ndarray)
        counter = Counter()
        for v in values:
            counter[v] += 1
            if len(counter) != 1:
                return False  
        return True
    
    def _check_constant(self, values, name=None, throw=True):
        """
        Возвращает True, если значения признака идентичны для всех объетов. Иначе возращает False или 
        вызывает исключение (в зависимости от параметра throw).
        Аргументы:
            :param values - значения признака (np.ndarray, csr_matrix, csc_matrix).
            :param name   - имя признака (str).
            :param throw  - если True, то вызывает исключение, если признак не константный (bool).
        """
        if self._is_constant(values):
            if throw:
                raise ConsantFeatureError(name, type(self).__name__)
            return False
        return True
    
    def _is_shaped(self, values, name=None):
        self._undefined_method('_is_shaped')
    def _check_shaped(self, values, name=None, throw=True):
        self._undefined_method('_check_shaped')
    
    def _get_length(self):
        self._undefined_method('_get_length')
    def _get_values(self):
        self._undefined_method('_get_values')
    def _get_dense(self):
        self._undefined_method('_get_dense')
    def _get_sparse(self):
        self._undefined_method('_get_sparse')
        
    def _preprocess(self, values, name):
        self._undefined_method('_preprocess')
        
    def _undefined_method(self, method_name):
        error_msg = 'Method "{}" of the abstract class "{}" must be redefined in derivative classes.'.format(method_name, type(self).__name__)
        assert False, error_msg


class SparseFeatureKernel(FeatureKernel):
    """
    Данный класс реализует базовые операции для работы с разряженными столбцами признаков. Наследник класса FeatureKernel.
    Переопределенные методы класса FeatureKernel:
        _is_shaped    - проверка размерности.
        _check_shaped - проверка размерности.
        _is_constant  - проверка константности.
        _preprocess   - преобразование входных данных к внутреннему формату хранения.
        _get_length   - возвращает количество объектов
        _get_values   - возвращает значения признаков в разряженном (csc_matrix) или плотном (np.ndarray) формате
        _get_dense    - возвращает DenseFeatureBase
        _get_sparse   - возвращает SparseFeatureBase
    """
    def __init__(self, owner):
        super().__init__(owner)

    ########################################################################
    def _is_shaped(self, values, name=None):
        return (len(values.shape) == 2) & (values.shape[0] == 1) & (values.shape[1] > 1)
    def _check_shaped(self, values, name=None, throw=True):
        if not self._is_shaped(values, name):
            if throw:
                error_msg = "Given sparse feature vector must have shape of type (1, size), but has {}".format(values.shape)
                raise ValueError(self._error_msg(error_msg))
            return False
        return True
    
    ################
    def _is_constant(self, values, name=None):
        values = values.toarray().flatten()
        return super()._is_constant(values, name)
    
    ########################################################################
    def _preprocess(self, values, name):
        """
        Данный метод преобразует входной разряженный формат хранения в csr_matrix размера [1, n_samples].
        Аргументы:
            :param values - значения признака (csr_matrix, csc_matrix).
            :param name   - имя признака (str).
        """
        self._printers[self._owner.METHODS](self._method_msg('_preprocess'))
        assert isinstance(values, (csr_matrix, csc_matrix))
        init_shape = values.shape
        if values.ndim != 2:
            error_msg = "Feature \"{}\" has shape {} but must have a shape of length 2.".format(name, init_shape)
            raise ValueError(self._error_msg(error_msg))
        if (init_shape[0] == 1) & (init_shape[1] > 1):
            new_values = values.tocsr() 
        elif (init_shape[0] > 1) & (init_shape[1] == 1):
            new_values = values.transpose().tocsr()
        else:
            error_msg = "Feature \"{}\" has incorrect shape {}. Must be either (1, n) or (n, 1)".format(name, shape, tuple(reversed(shape)))
            raise ValueError(self._error_msg(error_msg))
        info_msg = "Feature \"{}\" is transformed from shape {} to csr_matrix of shape {}".format(name, init_shape, new_values.shape)
        self._printers[self._owner.FORMAT_CHANGE_LEVEL](self._info_msg(info_msg))
        info_msg = '_preprocess returns {}'.format(type(new_values))
        self._printers[self._owner.METHODS](self._info_msg(info_msg))
        return new_values
    
    ########################################################################
    def _get_length(self):
        """
        Возвращает количество объектов.
        """
        return self._owner._values.shape[1]
    def _get_values(self, sparse=True, *args, **kwargs):
        """
        Возвращает значения признака в форме np.ndarray или csc_matrix.
        Аргументы:
            :param sparse - если True, то возвращает признаки в виде csc_matrix размера [n_samples, 1];
                            иначе возвращает np.ndarray размера [n_samples, 1].
        """
        if sparse:
            return self._owner._values.transpose().tocsc()
        else:
            return self._owner._values.todense().T
    def _get_dense(self):
        values = self._owner._values
        name = self._owner._name
        verbose = self._owner._verbose
        return DenseFeatureBase(values.toarray().flatten(), name, verbose)
    def _get_sparse(self):
        values = self._owner._values
        name = self._owner._name
        verbose = self._owner._verbose
        return SparseFeatureBase(values, name, verbose)


class DenseFeatureKernel(FeatureKernel):
    """
    Данный класс реализует базовые операции для работы с плотными столбцами признаков. Наследник класса FeatureKernel.
    Переопределенные методы класса FeatureKernel:
        _is_shaped    - проверка размерности.
        _check_shaped - проверка размерности.
        _preprocess   - преобразование входных данных к внутреннему формату хранения.
        _get_length   - возвращает количество объектов
        _get_values   - возвращает значения признаков в разряженном (csc_matrix) или плотном (np.ndarray) формате
        _get_dense    - возвращает DenseFeatureBase
        _get_sparse   - возвращает SparseFeatureBase
    """
    def __init__(self, owner):
        super().__init__(owner)
        
    ########################################################################
    def _is_shaped(self, values, name=None):
        return len(values.shape) == 1
    def _check_shaped(self, values, name=None, throw=True):
        if not self._is_shaped(values, name):
            if throw:
                raise ValueError(self._error_msg("Given dense feature vector must have shape of type (size, ), but has {}".format(values.shape)))
            return False
        return True
    
    ########################################################################
    def _preprocess(self, values, name):
        self._printers[self._owner.METHODS](type(self).__name__ + '._preprocess')
        assert isinstance(values, (np.ndarray, list))
        values = np.array(values)
        init_shape = values.shape
        if len(init_shape) not in [1, 2]:
            error_msg = "Feature \"{}\" has shape {} but must have a shape of length either 1 or 2.".format(
                name, init_shape)
            raise ValueError(self._error_msg(error_msg))
        if len(init_shape) == 1:
            new_values = values
        elif (init_shape[0] == 1) & (init_shape[1] > 1):
            new_values = values[0]
        elif (init_shape[0] > 1) & (init_shape[1] == 1):
            new_values = values[:, 0]
        else:
            error_msg = "Feature \"{}\" has incorrect shape {}. Must be either (n, ) or (1, n) or (n, 1)".format(
                name, shape, tuple(reversed(shape)))
            raise ValueError(self._error_msg(error_msg))
        info_msg = "Feature \"{}\" is transformed from shape {} to shape {}".format(
            name, init_shape, new_values.shape)
        self._printers[self._owner.FORMAT_CHANGE_LEVEL](self._info_msg(info_msg))
        return new_values
    
    ########################################################################
    def _get_length(self):
        return self._owner._values.shape[0]
    def _get_values(self, sparse=False, *args, **kwargs):
        if sparse:
            return csc_matrix(self._owner._values[:, np.newaxis])
        else:
            return self._owner._values[:, np.newaxis]
    def _get_dense(self):
        values = self._owner._values
        name = self._owner._name
        verbose = self._owner._verbose
        return DenseFeatureBase(values, name, verbose)
    def _get_sparse(self):
        values = csr_matrix(self._owner._values[np.newaxis, :])
        values.eliminate_zeros()
        name = self._owner._name
        verbose = self._owner._verbose
        return SparseFeatureBase(values, name, verbose)
    
    
class FeatureBase(Checker):
    """
    FeatureBase - базовый класс, контейнер для одного признака. 
    Признак хранится в виде либо разряженном, либо в плотном формате.
    Признак обладает своим именем. При проведении преобразований признака, его имя преобразуется.
    
    Уровни печати (значение verbose):
    * FORMAT_CHANGE_LEVEL - уровень, выше которого выводятся в печать сообщения о преобразованиях формата признака при инициализации
    """
    
    CAT_PREFIX = 'CAT_'
    NUM_PREFIX = 'NUM_'
    CTR_PREFIX = 'CTR_' # Counter feature
    LOO_PREFIX = 'LOO_' # LeaveOneOut feature
    FIL_PREFIX = 'FIL'   
    LE_PREFIX  = 'LE_'  # LabelEncoded feature
    OHE_PREFIX = 'OHE_' # OneHotEncoded feature

    FORMAT_CHANGE_LEVEL = 10
    METHODS = 11
    
    def is_constant(self):
        """
        Возвращает True, если значения признака идентичны для всех объетов. Иначе - False.
        """
        return self._kernel._is_constant(self._values, self._name)

    def is_numeric(self):
        """
        Возвращает True, если признак числовой. Иначе - False.
        """
        return self._kernel._is_numeric(self._values, self._name)
    
    ########################################################################
    def __init__(self, values, name, verbose=0):
        attributes = ['name', 'values', 'shape']
        setters = {}; getters = {}
        for attr in attributes:
            setters[attr] = self.__getattribute__('set_' + attr)
            getters[attr] = self.__getattribute__('get_' + attr)
        super().__setattr__('_setters', setters)
        super().__setattr__('_getters', getters)
        super().__init__()
        self._verbose = verbose
        if isinstance(values, (csr_matrix, csc_matrix)):
            self._sparse = True
            self._kernel = SparseFeatureKernel(self)
        elif isinstance(values, (np.ndarray, list)):
            self._sparse = False
            self._kernel = DenseFeatureKernel(self)
        else:
            raise TypeError('Type of "values" is unacceptable!')
        self.set_name(name)
        self.set_values(values)

    def __str__(self):
        return '[{}: {}, {}]'.format(type(self).__name__, self._name, self._shape)
    
    def __repr__(self):
        return str(self)
        
    def __len__(self):
        return self._kernel._get_length()
        
    ########################################################################
    def __getattr__(self, name):
        if name in self._getters:
            return self._getters[name]()
        raise AttributeError('Attribute "{}" not found!'.format(name))
    def __setattr__(self, name, value):
        if name in self._setters:
            return self._setters[name](value)
        return super().__setattr__(name, value)
    
    def set_name(self, name):
        self._check_type(name, 'feature_name', str)
        self._name = name
    def set_values(self, values):
        """
        Всегда сохраняется копия values.
        """
        self._printers[self.METHODS](self._method_msg('set_values({})'.format(type(values))))
        _values = self._kernel._preprocess(values, self._name)
        self._kernel._check_shaped(_values, self._name, throw=True)
        self._values = _values
        self._shape  = _values.shape
        self._printers[self.METHODS](self._info_msg('set_values setted {}'.format(type(self._values))))
    def set_shape(self, shape):
        assert False, 'Setting "shape" is not allowed'
    def get_name(self):
        return self._name
    def get_shape(self):
        return self._shape
    
    # Функции для получения значений в различных форматах
    def get_values(self, *args, **kwargs):
        return self._kernel._get_values(*args, **kwargs)
    def get_array(self):
        return self.get_values(sparse=False)
    def get_csc_matrix(self):
        return self.get_values(sparse=True)
    def get_data_frame(self):
        values = self.get_values(sparse=False).flatten()
        return pd.DataFrame({self.get_name(): values})
    def get_series(self):
        values = self.get_values(sparse=False)
        return pd.Series(values.flatten(), name=self.get_name())
    def as_data_frame(self):
        return self.get_data_frame()
    def as_series(self):
        return self.get_series()
    def as_array(self):
        return self.get_array()
    def as_csc_matrix(self):
        return self.get_csc_matrix()
        
    ########################################################################
    def _get_categorical_name(self, name):
        return FEATURE_PREFIXES['CAT'] + name
    def _get_numerical_name(self, name):
        return FEATURE_PREFIXES['NUM'] + name
    def _get_counter_name(self, name):
        return FEATURE_PREFIXES['CTR'] + name
    def _get_loo_name(self, name):
        return FEATURE_PREFIXES['LOO'] + name
    def _get_filtered_name(self, name, threshold):
        return FEATURE_PREFIXES['FIL'] + '{}_'.format(threshold) + name
    def _get_label_encoded_name(self, name):
        return FEATURE_PREFIXES['LE'] + name
    def _get_ohe_name(self, name):
        return FEATURE_PREFIXES['OHE'] + name

    ########################################################################
    def to_dense(self):
        if self._sparse:
            self._values = self._kernel._get_dense()._values
            self._kernel = DenseFeatureKernel(self)
        assert isinstance(self._values, np.ndarray)
        assert isinstance(self._kernel, DenseFeatureKernel)
    def to_sparse(self):
        if not self._sparse:
            self._values = self._kernel._get_sparse()._values
            self._kernel = SparseFeatureKernel(self)
        assert isinstance(self._values, csr_matrix)
        assert isinstance(self._kernel, SparseFeatureKernel)
    ########################################################################
    def deepcopy(self):
        """
        Выполняет глубокое копирования объекта. При необходимости копирования должна вызываться ТОЛЬКО она.
        Функция copy.deepcopy выполняет некорректное копирование из-за переопределения операции работы с атрибутами
        в классе FeatureBase.
        """
        return FeatureBase(copy.deepcopy(self._values), self._name, self._verbose)    
    

