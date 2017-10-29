'''Miscellaneous utilities.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import six


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
else:
    getargspec = inspect.getfullargspec


__all__ = (
    'get_class_defining_method',
    'getargspec',
)
