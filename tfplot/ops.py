''' Main plot operations. '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import re

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



def wrap(plot_func, batch=False, name=None):
    '''
    Wrap a plot function as a TensorFlow operation. It will return a python
    function that creates a TensorFlow plot operation applying the arguments
    as input.

    For example, if `plot_func(x)` is a python function that takes two
    arrays as input, and draw a plot by returning a matplotlib Figure,
    we can wrap this function as a Tensor factory, such as:

    ```python
    tf_plot = tfplot.wrap(plot_func, name="MyPlot")
    # x, y = get_batch_inputs(batch_size=4, ...)

    plot_x = tf_plot(x)   # Tensor("MyPlot:0", shape=(4, ?, ?, 3), dtype=uint8)
    plot_y = tf_plot(y)   # Tensor("MyPlot_1:0", shape=(4, ?, ?, 3), dtype=uint8)
    ```

    Args:
      plot_func: A python function or callable to wrap. See the documentation
        of `tfplot.plot()` for details.
      batch: If True, all the tensors passed as argument will be
        assumed to be batched. Default value is False.
      name: A default name for the operation (optional). If not given, the
        name of `plot_func` will be used.

    Returns:
      A python function that will create a TensorFlow plot operation,
      passing the provied arguments.
    '''

    def _wrapped_fn(*args, **kwargs_call):
        _plot = plot_many if batch else plot
        return _plot(plot_func, list(args),
                     name=name or _clean_name(plot_func.__name__),
                     **kwargs_call)

    _wrapped_fn.__name__ = 'wrapped_fn[%s]' % plot_func
    return _wrapped_fn


def _clean_name(s):
    """
    Convert a string to a valid variable, function, or scope name.
    """
    return re.sub('[^0-9a-zA-Z_]', '', s)


__all__ = (
    'plot',
    'plot_many',
    'wrap',
)
