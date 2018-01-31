# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from ..core import Checker
from .decorators import *

class Layer(Checker):
    # Checks
    # def _assert_nans(self, arr):
    #    assert not np.any(np.isnan(arr)), 'NaNs etected: {}!'.format(self)
    #  def _assert_infs(self, arr):
    #     assert not np.any(np.isinf(arr)), 'Infs detected {}!'.format(self)
    # def _check_arrays(self, *arrays):   #### Проверка входных массива данных на наличие nans и infs
    #     for arr in arrays:
    #         assert isinstance(arr, np.ndarray)
    #         self._assert_nans(arr)
    #         self._assert_infs(arr)
    
    
    def __init__(self, name=None):
        self.output = None       # Output of the layer is always kept for backpropagatoin
        self.grad_input = None   # Input gradient is saved just in case
        self.training = True     # 
        
        self._forward_enter_callback  = lambda: None
        self._forward_exit_callback   = lambda: None
        self._backward_enter_callback = lambda: None
        self._backward_exit_callback  = lambda: None
        
        # Проверки прямого распространения
        self._forward_checkers = []
        self._forward_checkers.append(self._forward_check_type)
        self._forward_checkers.append(self._forward_check_shape)
        self._forward_checkers.append(self._forward_check_value)
        # Препроцессинг прямого распространения
        self._forward_preprocessors = []
        self._forward_preprocessors.append(self._forward_preprocess_input_dtype)
        # Постпроцессинг прямого распространения
        self._forward_postprocessors = []
       
        # Проверки обратного распространения
        self._backward_checkers = []
        self._backward_checkers.append(self._backward_check_type)
        self._backward_checkers.append(self._backward_check_shape)
        self._backward_checkers.append(self._backward_check_value)
        # Препроцессинг прямого распространения
        self._backward_preprocessors = []
        self._backward_preprocessors.append(self._backward_preprocess_dtype)
        # Постпроцессинг прямого распространения
        self._backward_postprocessors = []
        self._backward_postprocessors.append(self._backward_postprocess_clip)
        
        self.layer_type_name = type(self).__name__
        #self.name = None
        #self.dtype = None         # Must be set during initialization
        #self.debug = None         # Must be set during initialization
        #self.input_shape = None   # Must be set during initialization
        #self.output_shape = None  # Must be set during initialization
        #self.grad_clip = None     # Must be set during initialization
        self.initialized = False  # Must be set to True after initialization
        
    def __repr__(self):
        return type(self).__name__
    
    # Setting callbacks
    def set_forward_enter_call(self, callback=lambda: None):
        self._forward_enter_callback = callback
    def set_forward_exit_call(self, callback=lambda: None):
        self._forward_exit_callback = callback
    def set_backward_enter_call(self, callback=lambda: None):
        self._backward_enter_callback = callback
    def set_backward_exit_call(self, callback=lambda: None):
        self._backward_exit_callback = callback
    
       
    ################################## 
    ###       Initialization       ###
    ##################################
    def _check_initialized(self):
        assert hasattr(self, 'debug')
        assert hasattr(self, 'seed')
        assert hasattr(self, 'dtype')
        assert hasattr(self, 'input_shape')
        assert hasattr(self, 'output_shape')
    def initialize(self, params):
        """Данная функция вызывается в процессе компиляции модели (метод Model.compile). Только позле инициализация уровень становится готов к работе.
        Аргументы:
        -params: словарь, содержащий параметры сети:
            debug       - 
            seed        - 
            dtype       - тип данных; все данные сети, включая входы и переменные, будут представлены в данном формате
            input_shape - размер входного массива данных
        
        Возвращаемое значение:
        -params: возвращает тот же самый словарь params; однако часть его значений меняется в процессе инициализации (см. 'Замечания для разработчиков')
        
        Замечания для разработчиков:
            Словарь params передается по ссылке. Внимание: процессе инициализации уровней сети данный словарь изменяется;
                если исходный словарь параметров не должен изменяться, то в метод initialize следует передавать копию словаря.
            С точки зрения архитектуры, реализация данной функции относится к парадигме PIMPL: вся действительная работа выполнеяется в методе _initialize"""
        self._initialize(params)
        self._check_initialized()
        self.initialized = True
        return params
    def _initialize(self, params):
        """Must be called at each layer via super()._initialize(params) at the beginning of layer.initialize() call"""
        self._initialize_debug(params)
        self._initialize_seed(params)
        self._initialize_dtype(params)
        self._initialize_grad_clip(params)
        self._initialize_name(params)
        self._initialize_input_shape(params)
        self._initialize_params(params)
        self._initialize_output_shape(params)
        return params
    def _initialize_name(self, params):
        names = params.setdefault('names', {})
        n_layers = names.setdefault(self.layer_type_name, 0)
        names[self.layer_type_name] += 1
        self.name = self.layer_type_name + str(n_layers)
        return params
    def _initialize_seed(self, params):
        # Предполагается, что каждый уровень сети снабжен собственным случайным генератором (Just in case)
        self.seed = params.setdefault('seed', 0)
        self.generator = np.random.RandomState(self.seed)
        params['seed'] += 1
        return params
    def _initialize_dtype(self, params):
        self.dtype = params.setdefault('dtype', np.float64)
        return params
    def _initialize_debug(self, params):
        self.debug = params.setdefault('debug', False)
        return params
    def _initialize_params(self, params):
        # Empty layer does not have params
        return params
    def _initialize_input_shape(self, params):
        # TODO: непонятно, какую версию использовать в базовом классе Layer
        assert 'input_shape' in params, '"input_shape" is not provided though must be.'
        input_shape = params['input_shape']
        self.input_shape = tuple([-1] + list(input_shape[1:]))
        return params
    def _initialize_output_shape(self, params):
        # По умолчанию предполагается, что размер выхода сети совпадает с его входом.
        # Для уровней, изменющих размер входа, следует перегрузить данную функцию.
        self.output_shape = self.input_shape
        return params
    def _initialize_grad_clip(self, params):
        self.grad_clip = params.setdefault('grad_clip', np.inf)
        assert self.grad_clip > 0, '"grad_clip" must be higher than 0.'
        return params
       
    ################################## 
    ###     Forward propagation    ###
    ##################################
    @check_initialized
    def forward(self, input, target=None):
        self._forward_enter_callback()            # Вызов начального callback-а прямого распространения
        self._forward_check(input, target)        # Проверка корректности входа слоя (типа, формы, значений)
        input, target = self._forward_preprocess(input, target) # Обработка входа слоя
        self._forward(input, target)              # Обновление выхода слоя (аттрибута self.output)
        self._forward_postprocess()               # Обработка выхода слоя (аттрибута self.output)
        self._forward_exit_callback()             # Вызов финального callback-а прямого распространения
        return self.output
    def _forward(self, input, target=None):
        self.output = input
    # Проверки прямого распространения    
    def _forward_check(self, input, target=None):   #### Проверка входного массива
        if self.debug:
            for checker in self._forward_checkers:
                checker(input, target)
    def _forward_check_type(self, input, target=None):
        assert isinstance(input, np.ndarray)
        if target is not None:
            assert isinstance(target, np.ndarray)
    def _forward_check_shape(self, input, target=None):          #### Проверка правильности размера входного массива данных
        pass
    def _forward_check_value(self, input, target=None):          #### Проверка корректности значений входных массивов данных при прямом распространении ошибки
        assert not np.any(np.isnan(input)), 'NaNs detected in "input" of forward propagation of layer "{}"'.format(self.name)
        assert not np.any(np.isinf(input)), 'Infs detected in "input" of forward propagation of layer "{}"'.format(self.name)
        if target is not None:
            assert not np.any(np.isnan(target)), 'NaNs detected in "target" of forward propagation of layer "{}"'.format(self.name)
            assert not np.any(np.isinf(target)), 'Infs detected in "target" of forward propagation of layer "{}"'.format(self.name)
    # Препроцессинг прямого распространения
    def _forward_preprocess(self, input, target=None):       #### Предобработка прямого распространения
        for preprocessor in self._forward_preprocessors:
            input, target = preprocessor(input, target)
        return input, target
    def _forward_preprocess_input_dtype(self, input, target=None): #### Приведение данных к требуемому типу  
        input  = self._convert_to_dtype(input)        
        return input, target
    # Постпроцессинг прямого распространения
    def _forward_postprocess(self):               #### Постобработка прямого распространения
        for postprocessor in self._forward_postprocessors:
            postprocessor()
            
    ################################## 
    ###    Backward propagation    ###
    ##################################
    @check_initialized
    def backward(self, input, grad_output):
        """
        Внимание: данную функцию нельзя переопределять. 
            Для изменения поведения уровния при forward propagation следует переопределить 
            методы update_grad_input и update_grad_param, либо метод _backward
        """
        self._backward_enter_callback()
        self._backward_check(input, grad_output)
        input, grad_output = self._backward_preprocess(input, grad_output)
        self._backward(input, grad_output)
        self._backward_postprocess()
        self._backward_exit_callback()
        return self.grad_input
    def _backward(self, input, grad_output):
        # Идиома PIMPL
        self.grad_input = grad_output
    # Проверки для обратного распространения
    def _backward_check(self, input, grad_output):
        if self.debug:
            for checker in self._backward_checkers:
                checker(input, grad_output)        
    def _backward_check_type(self, input, grad_output):
        assert isinstance(input, np.ndarray)
        assert isinstance(grad_output, np.ndarray)
    def _backward_check_shape(self, input, grad_output): #### Проверка правильности размеров входных массивов данных при обратном распространении ошибки
        pass
    def _backward_check_value(self, input, grad_output): #### Проверка корректности значений входных массивов данных при обратном распространении ошибки
        assert not np.any(np.isnan(input)), 'NaNs detected in "input" of backward propagation of layer "{}"'.format(self.name)
        assert not np.any(np.isinf(input)), 'Infs detected in "input" of backward propagation of layer "{}"'.format(self.name)
        assert not np.any(np.isnan(grad_output)), 'NaNs detected in "grad_output" of backward propagation of layer "{}"'.format(self.name)
        assert not np.any(np.isinf(grad_output)), 'Infs detected in "grad_output" of backward propagation of layer "{}"'.format(self.name)
    # Препроцессинг обратного распространения 
    def _backward_preprocess(self, input, grad_output):           # Предобработка обратного распространения
        for preprocessor in self._backward_preprocessors:
            input, grad_output = preprocessor(input, grad_output)
        return input, grad_output 
    def _backward_preprocess_dtype(self, input, grad_output):
        input = self._convert_to_dtype(input)
        grad_output = self._convert_to_dtype(grad_output)
        return input, grad_output
    # Постпроцессинг обратного распространения 
    def _backward_postprocess(self):                              # Постобработка обратного распространения
        for postprocessor in self._backward_postprocessors:
            postprocessor()                                   
    def _backward_postprocess_clip(self):  ### Gradients clipping; ограничение градиентов
        np.clip(self.grad_input, -self.grad_clip, self.grad_clip, self.grad_input)
        grad_params = self.get_grad_params(copy=False)
        for param_name, param_value in grad_params.items():
            np.clip(param_value, -self.grad_clip, self.grad_clip, param_value)
            
    ################################## 
    ###   Pre- & post- processors  ###
    ##################################
    def _convert_to_dtype(self, v):
        if v is None:
            return v
        if isinstance(v, np.ndarray):
            return v.astype(self.dtype, copy=False) 
        return v

    ################################## 
    ###         Parameters         ###
    ##################################
    # Получение параметров и градиентов
    @check_initialized
    def get_params(self, copy=False):
        return OrderedDict()
    @check_initialized
    def get_grad_params(self, copy=False):
        return OrderedDict()    
    def _make_dict_copy(self, d, copy=False):
        if copy:
            return OrderedDict([(k, v.copy()) for k, v in d.items()])
        return d
    # Выставление параметров и градиентов
    @check_initialized
    def set_params(self, new_params):
        # Для работы данного метода в классах-потомках достаточно корректной работы метода get_params в этих классах
        params = self.get_params(copy=False)
        for param_name in params:
            if param_name in new_params:
                param_shape = params[param_name].shape
                new_param_shape = new_params[param_name].shape
                assert param_shape == new_param_shape, 'Attempt to write a value with shape {} into parameter "{}" with shape {}'.format(
                    new_param_shape, param_name, param_shape)
                np.copyto(params[param_name], new_params[param_name]) 
            else:
                # TODO: create warning
                pass
    @check_initialized
    def set_grad_params(self, new_grad_params):
        # Для работы данного метода в классах-потомках достаточно корректной работы метода get_grad_params в этих классах 
        grad_params = self.get_grad_params(copy=False)
        for param_name in grad_params:
            if param_name in new_grad_params:
                param_shape = grad_params[param_name].shape
                new_param_shape = new_grad_params[param_name].shape
                assert param_shape == new_param_shape, 'Attempt to write a value with shape {} into gradient of parameter "{}" with shape {}'.format(
                    new_param_shape, param_name, param_shape)
                np.copyto(grad_params[param_name], new_grad_params[param_name])      
            else:
                # TODO: create warning
                pass
    @check_initialized
    def zero_grad_params(self):
        # Для работы данного метода в классах-потомках достаточно корректной работы метода get_grad_params в этих классах
        grad_params = self.get_grad_params(copy=False)
        for param_name, grad_param_value in grad_params.items():
            grad_param_value.fill(0)
                 
    ################################## 
    ###       Regularization       ###
    ##################################
    @check_initialized
    def get_regularizers(self):
        return OrderedDict()
    @check_initialized
    def get_regularization_loss(self):
        # Для работы данного метода в классах-потомках достаточно корректной работы метода get_regularizers в этих классах
        regularizers = self.get_regularizers()
        params = self.get_params()
        loss = 0.0
        for param_name, regularizer in regularizers.items():
            assert param_name in params, 'Regularization for unknown parameter "{}" in layer "{}".'.format(param_name, self.name)
            loss += regularizer.get_loss(params[param_name])
        return loss
        
    ################################## 
    ###   Changing operation mode  ###
    ##################################
    @check_initialized
    def train(self):
        self.training = True
    @check_initialized
    def evaluate(self):
        self.training = False
