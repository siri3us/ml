# -*- coding: utf-8 -*-

def dtype_conversion(fn):
    def fn_(self=None, *args, **kwargs):
        dtype = self.dtype
        return fn(self, 
                  *[arg.astype(dtype, copy=False) for arg in args], 
                  **{k: v.astype(dtype, copy=False) for k, v in kwargs.items()})
    return fn_

def check_initialized(fn):
    def fn_(self=None, *args, **kwargs):
        assert self.initialized, 'Object {} must be initialized to call its methods.'.format(self)
        return fn(self, *args, **kwargs)
    return fn_

def check_compiled(fn):
    def fn_(self=None, *args, **kwargs):
        assert self.compiled, 'Model {} must be compiled before usage.'.format(self)
        return fn(self, *args, **kwargs)
    return fn_

