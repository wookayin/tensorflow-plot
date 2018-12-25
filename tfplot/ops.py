''' Main plot operations. '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import re
import types

import tensorflow as tf
import numpy as np

from . import figure
from . import util
from .util import merge_kwargs, decode_bytes_if_necessary

from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot(plot_func, in_tensors, name='Plot',
         **kwargs):
    '''
    Create a TensorFlow op which draws plot in an image. The resulting
    image is in a 3-D uint8 tensor.

    Given a python function ``plot_func``, which takes numpy arrays as its
    inputs (the evaluations of ``in_tensors``) and returns a matplotlib
    `Figure` object as its outputs, wrap this function as a TensorFlow op.
    The returning figure will be rendered as a RGB-A image upon execution.

    Args:
      plot_func: a python function or callable
        The function which accepts numpy `ndarray` objects as an argument
        that match the corresponding `tf.Tensor` objects in ``in_tensors``.
        It should return a new instance of ``matplotlib.figure.Figure``,
        which contains the resulting plot image.
      in_tensors: A list of `tf.Tensor` objects.
      name: A name for the operation (optional).
      kwargs: Additional keyword arguments passed to ``plot_func`` (optional).

    Returns:
      A single `uint8` `Tensor` of shape ``(?, ?, 4)``, containing the plot
      image that ``plot_func`` computes.
    '''

    if not hasattr(plot_func, '__call__'):
        raise TypeError("plot_func should be callable")

    if not isinstance(name, six.string_types):
        raise TypeError("name should be str or unicode, " +
                        "given {}".format(type(name)))

    if not isinstance(in_tensors, (list, tuple)):
        if isinstance(in_tensors, (tf.Tensor, np.ndarray, np.number)):
            in_tensors = [in_tensors]
        else:
            raise TypeError("in_tensors should be a list of Tensors, " +
                            "given {}".format(type(in_tensors)))

    in_tensors = [tf.convert_to_tensor(t) for t in in_tensors]

    def _render_image(*args):
        # `args` is (a tuple of) python values

        # for tf.string tensors, decode into unicode if necessary.
        args = tuple(
            (decode_bytes_if_necessary(arg) if t.dtype == tf.string else arg) \
            for (arg, t) in zip(args, in_tensors)
        )
        fig = plot_func(*args, **kwargs)

        if not isinstance(fig, Figure):
            raise TypeError("The returned value should be a " +
                            "matplotlib.figure.Figure object, " +
                            "but given {}".format(type(fig)))

        # render fig into numpy array.
        image = figure.to_array(fig)
        return image

    im = tf.py_func(_render_image, in_tensors, Tout=tf.uint8,
                    name=name)
    im.set_shape([None, None, 4])
    return im


def plot_many(plot_func, in_tensors, name='PlotMany',
              max_outputs=None,
              **kwargs):
    '''
    A batch version of ``plot``.  Create a TensorFlow op which draws
    a plot for each image. The resulting images are given in a 4-D `uint8`
    Tensor of shape ``[batch_size, height, width, 4]``.

    Args:
      plot_func: A python function or callable, which accepts numpy
        `ndarray` objects as an argument that match the corresponding
        `tf.Tensor` objects in ``in_tensors``. It should return a new instance
        of ``matplotlib.figure.Figure``, which contains the resulting plot image.
        The shape (height, width) of generated figure for each plot should
        be same.
      in_tensors: A list of `tf.Tensor` objects.
      name: A name for the operation (optional).
      max_outputs: Max number of batch elements to generate plots for (optional).
      kwargs: Additional keyword arguments passed to `plot_func` (optional).

    Returns:
      A single `uint8` `Tensor` of shape ``(B, ?, ?, 4)``, containing the B
      plot images, each of which is computed by ``plot_func``, where B equals
      ``batch_size``, the number of batch elements in the each tensor from
      ``in_tensors``, or ``max_outputs`` (whichever is smaller).
    '''

    # unstack all the tensors in in_tensors
    args = []
    batch_size = None

    with tf.name_scope(name):
        for in_tensor in in_tensors:
            in_tensor = tf.convert_to_tensor(in_tensor)
            arg_unpacked = tf.unstack(in_tensor, name=in_tensor.op.name + '_unstack')
            if batch_size is not None and batch_size != len(arg_unpacked):
                raise ValueError("All tensors in in_tensors should have " +
                                 "the same batch size : %d != %d for %s" % (
                                     batch_size, len(arg_unpacked), in_tensor
                                 ))
            batch_size = len(arg_unpacked)
            args.append(arg_unpacked)

        # generate plots for each batch element
        ims = []
        for k, arg in enumerate(zip(*args)):
            if max_outputs is not None and k >= max_outputs:
                break
            im = plot(plot_func, arg, name=('Plot_%d' % k), **kwargs)
            ims.append(im)

        # combine the generated plots and use them as image summary
        im_packed = tf.stack(ims, name='PlotImages')

    return im_packed



__all__ = (
    'plot',
    'plot_many',
)
