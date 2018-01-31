# -*- coding: utf-8 -*-

import numbers
import numpy as np

class Printer:
    def __init__(self, verbose, owner):
        """
        Класс, ответственный за распечатку сообщений. 
        Печатает сообщение, когда его уровень печати превосходит уноверь печати
        объекта класса.
        """
        self._verbose = verbose
        self._owner = owner
    def __call__(self, *args, **kwargs):
        if self._owner._verbose >= self._verbose:
            print(*args, **kwargs)

class Checker(Printer):
    def _error_msg(self, error_msg):
        return 'ERROR: ' + type(self).__name__ + ': ' + error_msg
    def _info_msg(self, info_msg):
        return 'INFO: ' + type(self).__name__ + ': ' + info_msg
    def _warning_msg(self, warning_msg):
        return 'WARNING: ' + type(self).__name__ + ': ' + warning_msg
    def _method_msg(self, method_name, method_msg):
        return type(self).__name__ + '.' + method_name + '(): ' + method_msg

    def __init__(self, distr_sum_error=1e-8):
        self._distr_sum_error = distr_sum_error
        self._printers = {}
        for v in range(20):
            self._printers[v] = Printer(v, self)
            
    ################################## 
    ###        Type checks         ###
    ##################################
    def _check_type(self, n, name, *types):
        if not isinstance(n, tuple(types)):
            type_names = [t.__name__ for t in types]
            raise TypeError('Param "{}" must have one of the types "{}", not "{}".'.format(
                name, type_names, type(n).__name__))
        return True
    def _check_numeric(self, n, name, msg=None):
        if not isinstance(n, numbers.Number):
            if msg is None: msg = 'Param "{}" must be a number'.format(name)
            raise TypeError(msg)
        return True 
    def _check_int(self, n, name, msg=None):
        self._check_numeric(n, name, msg=msg)
        if not n == int(n):
            if msg is None: msg = 'Param "{}" must be an integer number'.format(name)
            raise TypeError(msg)
        return True
    def _check_boolean(self, n, name, msg=None):
        if not isinstance(n, bool):
            if msg is None: msg = 'Param "{}" must be a boolean'.format(name)
            raise TypeError(msg)
        return True
        
    ################################## 
    ###        Value checks        ###
    ##################################
    def _check_positive(self, n, name, msg=None):
        self._check_numeric(n, name, msg=msg)
        if n <= 0:
            if msg is None: msg = 'Param "{}" must be positive'.format(name)
            raise ValueError(msg)
        return True
    def _check_nonnegative(self, n, name, msg=None):
        self._check_numeric(n, name, msg=msg)
        if n < 0:
            if msg is None: msg = 'Param "{}" must be nonegative'.format(name)
            raise ValueError(msg)
        return True
        
    ################################## 
    ###    Type + value checks     ###
    ##################################
    def _check_int_positive(self, n, name):
        self._check_int(n, name)
        self._check_positive(n, name)
        return True
    def _check_int_nonnegative(self, n, name):
        self._check_int(n, name)
        self._check_nonnegative(n, name)
        
        return True
    def _check_proba(self, n, name):
        self._check_numeric(n, name)
        if (n <= 1) & (n >= 0):
            return True
        if msg is None: msg = 'Param "{}" is not proba though it must be'.format(name)
        raise ValueError(msg)
    def _check_distr(self, distr, name):
        for i in range(len(distr)):
            self._check_proba(distr[i], name + '[{}]'.format(i))
        if not np.allclose(np.sum(distr), 1, self._distr_sum_error):
            raise ValueError("Distribution \"{}\" is not normed: sum equals {}".format(name, np.sum(distr)))
        return True
        
    def _check_array_range(self, arr, range, arr_name):
        self._check_type(arr, arr_name, np.ndarray)
        if not ((np.min(arr) >= range[0]) & (np.max(arr) <= range[1])):
            raise ValueError("All elements of the array \"{}\" must be in the range [{}, {}]".format(arr_name, range[0], range[1]))
        return True
        
    ################################## 
    ###          Setters           ###
    ##################################  
    # Класс Checker обладает специальными методами, позволяющими не только проводить проверку некоторого 
    # значения, но и сохранять это значение в качестве аттрибута. Так как данная возможность операется на 
    # использвание метода __setattr__, то требуется соблюдать особую осторожность, если данный метода 
    # переопределен в наследнике класса Checker.
    def _set_number(self, value, name):
        self._check_numeric(value, name)
        self.__setattr__(name, value)
        
    def _set_positive_number(self, value, name):
        self._check_positive(value, name)
        self.__setattr__(name, value)
        
    def _set_nonegative_number(self, value, name):
        self._check_nonnegative(value)
        self.__setattr__(name, value)
    
    def _set_positive_int(self, value, name):
        self._check_int(value, name)
        self._set_positive_number(value, name)
        
    def _set_nonegative_int(self, value, name):
        self._check_int(value, name)
        self._set_nonegative_number(value, name)
    
    def _set_proba(self, value, name):
        self._check_proba(value)
        self.__setattr__(name, value)
      
    def _set_boolean(self, value, name):
        self._check_type(value, name, bool)
        self.__setattr__(name, value)
