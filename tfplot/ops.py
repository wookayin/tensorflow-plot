''' Main plot operations. '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import tensorflow as tf
import numpy as np

from . import figure
from matplotlib.figure import Figure


def plot(plot_func, in_tensors, name='Plot',
         **kwargs):
    '''
    Create a TensorFlow op which draws plot in an image. The resulting
    image is in a 3-D uint8 tensor.

    Given a python function `plot_func`, which takes numpy arrays as its
    inputs (the evaluations of `in_tensors`) and returns a matplotlib
    `figure` object as its outputs, wrap this function as a TensorFlow op.
    The returning figure will be rendered as a RGB image upon execution.

    Args:
      plot_func: A python function or callable, which accepts numpy
        `ndarray` objects as an argument that match the corresponding
        `tf.Tensor` objects in `in_tensors`. It should return a new instance
        of `matplotlib.figure.Figure`, which contains the resulting plot image.
      in_tensors: A list of `Tensor` objects.
      name: A name for the operation (optional).
      kwargs: Additional keyword arguments passed to `plot_func` (optional).

    Returns:
      A single `uint8` `Tensor` of shape `(?, ?, 3)`, containing the plot
      image that `plot_func` computes.
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

    def _render_image(*args):
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
    im.set_shape([None, None, 3])
    return im


def plot_many(plot_func, in_tensors, name='PlotMany',
              **kwargs):
    '''
    A batch version of `plot`.  Create a TensorFlow op which draws
    a plot for each image. The resulting images are given in a 4-D `uint8`
    Tensor of shape `[batch_size, height, width, 3]`.

    Args:
      plot_func: A python function or callable, which accepts numpy
        `ndarray` objects as an argument that match the corresponding
        `tf.Tensor` objects in `in_tensors`. It should return a new instance
        of `matplotlib.figure.Figure`, which contains the resulting plot image.
        The shape (height, width) of generated figure for each plot should
        be same.
      in_tensors: A list of `Tensor` objects.
      name: A name for the operation (optional).
      kwargs: Additional keyword arguments passed to `plot_func` (optional).

    Returns:
      A single `uint8` `Tensor` of shape `(batch_size, ?, ?, 3)`, containing
      the `batch_size` plot images each of which is computed by `plot_func`,
      where `batch_size` is the number of batch elements in the each tensor
      from `in_tensors`.
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
            im = plot(plot_func, arg, name=('Plot_%d' % k), **kwargs)
            ims.append(im)

        # combine the generated plots and use them as image summary
        im_packed = tf.stack(ims, name='PlotImages')

    return im_packed


__all__ = (
    'plot',
    'plot_many',
)
