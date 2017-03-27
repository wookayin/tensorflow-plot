''' Main plot operations. '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

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

    def _render_image(*args):
        fig = plot_func(*args, **kwargs)

        if not isinstance(fig, Figure):
            raise TypeError("The returned value should be a " +
                            "matplotlib.figure.Figure object, " +
                            "but given {}".format(type(fig)))

        # render fig into numpy array.
        return figure.to_array(fig)

    im = tf.py_func(_render_image, in_tensors, Tout=tf.uint8,
                    name=name)
    im.set_shape([None, None, 3])
    return im


__all__ = (
    'plot',
)
