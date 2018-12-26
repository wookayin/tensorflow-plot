'''Miscellaneous utilities.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import six
import numpy as np


def get_class_defining_method(m):
    '''
    Get the class type that defines `m`, if it is a method.  If m is not a
    method, returns None. Should work both in python 2 and 3.

    Code originated from https://stackoverflow.com/questions/3589311/
    '''
    if inspect.ismethod(m):
        if hasattr(m, 'im_class'):
            return m.im_class
        for cls in inspect.getmro(m.__self__.__class__):
            if cls.__dict__.get(m.__name__) is m:
                return cls
        m = m.__func__

    if inspect.isfunction(m):
        try:
            cls = getattr(inspect.getmodule(m),
                          m.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
            if isinstance(cls, type):
                return cls
        except AttributeError:
            return None

    return None


# getargspec(fn)
if six.PY2:
    getargspec = inspect.getargspec

    def getargspec_allargs(func):
        argspec = getargspec(func)
        return argspec.args

else: # Python 3
    getargspec = inspect.getfullargspec

    def getargspec_allargs(func):
        argspec = getargspec(func)
        return argspec.args + argspec.kwonlyargs


def merge_kwargs(kwargs, kwargs_new):
    '''
    Merge two kwargs.

    One could simply use {**kwargs, **kwargs_new} to merge two kwargs dict,
    but we should support old python versions too.

    Moreover, values for duplicated key will be overwritten (in favor of kwargs_new).
    '''
    kwargs = kwargs.copy()
    kwargs.update(kwargs_new)
    return kwargs


_np_decode = np.vectorize(lambda b: b.decode('utf8'))

def decode_bytes_if_necessary(arg):
    """
    Decodes scalar bytes and ndarray of bytes into unicode counterparts.
    """
    if isinstance(arg, bytes):
        # Regardless of python 2 and 3, return as unicode.
        return arg.decode('utf8')
    if isinstance(arg, np.ndarray) and arg.dtype == object:
        return _np_decode(arg)
    else:
        return arg


__all__ = (
    'get_class_defining_method',
    'getargspec',
    'merge_kwargs',
    'decode_bytes_if_necessary',
)
