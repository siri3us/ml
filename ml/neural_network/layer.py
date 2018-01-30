# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from ..core import Checker
from .decorators import *

class Layer(Checker):
    # Checks
    def _assert_nans(self, arr):
        assert not np.any(np.isnan(arr)), 'NaNs etected: {}!'.format(self)
    def _assert_infs(self, arr):
        assert not np.any(np.isinf(arr)), 'Infs detected {}!'.format(self)
    def _check_arrays(self, *arrays):
        for arr in arrays:
            assert isinstance(arr, np.ndarray)
            self._assert_nans(arr)
            self._assert_infs(arr)
    
    
    def __init__(self, name=None):
        self.output = None       # Output of the layer is always kept for backpropagatoin
        self.grad_input = None   # Input gradient is saved just in case
        self.training = True     # 
        
        self._forward_enter_callback  = lambda: None
        self._forward_exit_callback   = lambda: None
        self._backward_enter_callback = lambda: None
        self._backward_exit_callback  = lambda: None
        
        self.name = name
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
    
    # Initialization
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
        layer_type_name = type(self).__name__
        n_layers = names.setdefault(layer_type_name, 0)
        names[layer_type_name] += 1
        if self.name is None:
            self.name = layer_type_name + str(n_layers)
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
        assert 'input_shape' in params, '"input_shape" is not provided though must be.'
        input_shape = params['input_shape']
        self.input_shape = tuple([-1] + list(input_shape[1:]))
        return params
    def _initialize_output_shape(self, params):
        # По умолчанию предполагается, что размер выхода сети совпадает с его входом.
        # Для уровней, изменющих размер входа, следует перегрузить данную функцию.
        self.output_shape = params['input_shape']
        return params
    def _initialize_grad_clip(self, params):
        self.grad_clip = params.setdefault('grad_clip', np.inf)
        assert self.grad_clip > 0, '"grad_clip" must be higher than 0.'
        return params
       
    ################################## 
    ###     Forward propagation    ###
    ##################################
    @check_initialized
    def forward(self, input):
        self._forward_enter_callback()
        self._check_forward_input(input)
        input = self._forward_preprocess(input)
        self._forward(input)            # Finding output tensor; self.output
        self._forward_postprocess()
        self._forward_exit_callback()   # Callback during forward propagation
        return self.output
    def _check_forward_input(self, input):        #### Проверка входного массива
        if self.debug:
            self._check_arrays(input)             # Проверка входного массива данных на наличие nans и infs
            self._check_input_shape(input)        # Проверка правильности размера входного массива данных
    def _forward_preprocess(self, input):         #### Предобработка прямого распространения
        return self._convert_to_dtype(input)      # Приведение данных к требуемому типу    
    def _forward(self, input):
        self.update_output(input)                 
    def update_output(self, input):
        self.output = input
    def _forward_postprocess(self):               #### Постобработка прямого распространения
        return
    def _check_input_shape(self, input):
        pass
        
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
        self._check_backward_input(input, grad_output)
        input, grad_output = self._backward_preprocess(input, grad_output)
        self._backward(input, grad_output)
        self._backward_postprocess()
        self._backward_exit_callback()
        return self.grad_input
    def _check_backward_input(self, input, grad_output):
        if self.debug:
            self._check_arrays(input, grad_output)                   # Проверка входных массива данных на наличие nans и infs
            self._check_input_grad_output_shape(input, grad_output)  # Проверка правильности размеров входных массивов данных
    def _backward_preprocess(self, input, grad_output):          # Предобработка обратного распространения
        return self._convert_to_dtype(input, grad_output)        
    def _backward(self, input, grad_output):
        # Идиома PIMPL
        self.update_grad_input(input, grad_output)               # This updates self.grad_input
        self.update_grad_param(input, grad_output)               # This updates all grad params  
    def update_grad_input(self, input, grad_output):
        self.grad_input = grad_output
    def update_grad_param(self, input, grad_output):
        pass
    def _backward_postprocess(self):                             # Постобработка обратного распространения
        self._clip_gradients()                                   # Gradients clipping; ограничение градиентов
    def _check_input_grad_output_shape(self, input, grad_output):
        # Поведение по умолчанию предполагает, что слой сети не изменяет размер входа TODODO
        # assert input.shape == grad_output.shape, 'input.shape({}) != grad_output.shape({}) for layer "{}".'.format(input.shape, grad_output.shape, self.name)
        pass
        
    ################################## 
    ###   Pre- & post- processors  ###
    ##################################
    def _convert_to_dtype(self, *args):
        args = tuple([arg.astype(self.dtype, copy=False) for arg in args])
        if len(args) == 1:
            return args[0]
        return args
    def _clip_gradients(self):
        np.clip(self.grad_input, -self.grad_clip, self.grad_clip, self.grad_input)
        grad_params = self.get_grad_params()
        for param_name, param_value in grad_params.items():
            np.clip(param_value, -self.grad_clip, self.grad_clip, param_value)
    
    
    # Regulariation
    @check_initialized
    def get_regularization_loss(self):
        return 0.0
    
    # Getting params and gradients
    @check_initialized
    def get_params(self, copy=False):
        return OrderedDict()
    @check_initialized
    def get_grad_params(self, copy=False):
        return OrderedDict()
    @check_initialized
    def get_regularizers(self):
        return OrderedDict()    
    def _make_dict_copy(self, d, copy=False):
        if copy:
            return OrderedDict([(k, v.copy()) for k, v in d.items()])
        return d
      
    @check_initialized
    def zero_grad_params(self):
        """
        Данная функция обнуляет текущие значения градиентов
        """
        pass
    
    @check_initialized
    def set_params(self, new_params):
        params = self.get_params(copy=False)
        for param_name in new_params:
            assert param_name in params, 'Layer "{}" does not have a parameter with name "{}".'.format(self, param_name)
            np.copyto(params[param_name], new_params[param_name]) 
    @check_initialized
    def set_grad_params(self, new_grad_params):
        grad_params = self.get_grad_params(copy=False) # Getting references to current gradients
        for param_name in new_grad_params:
            assert param_name in grad_params, 'Layer "{}" does not have a parameter with name "{}".'.format(self, param_name)
            np.copyto(grad_params[param_name], new_grad_params[param_name])
            
    # Changing network mode
    @check_initialized
    def train(self):
        self.training = True
    @check_initialized
    def evaluate(self):
        self.training = False
