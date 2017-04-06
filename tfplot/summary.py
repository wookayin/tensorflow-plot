from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import ops


def plot(name, plot_func, in_tensors, collections=None, **kwargs):
    """
    Create a tensorflow op that outpus a `Summary` protocol buffer,
    to which a single plot operation is executed (i.e. image summary).

    Basically it is a one-liner wrapper of `tfplot.ops.plot()` and
    `tf.summary.image()` calls.

    The generated `Summary` has single image summary value containing
    an image of the plot.

    Args:
      name: The name of scope for the generated ops and the summary op.
        Will also serve as a series name prefix in TensorBoard.
      plot_func: A python function or callable, specifying the plot operation
        as in `tfplot.ops.plot()`. See the documentation at `plot()`.
      in_tensors: A list of `Tensor` objects, as in `plot()`.
      collections: Optional list of ops.GraphKeys.  The collections to add the
        sumamry to.  Defaults to [_ops.GraphKeys.SUMMARIES]
      kwargs: Optional keyword arguments passed to `plot()`.

    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer (tensorflow operation).
    """
    with tf.name_scope(name):
        im = ops.plot(plot_func, in_tensors, **kwargs)
        im = tf.expand_dims(im, axis=0)
        summary = tf.summary.image(name="ImageSummary", tensor=im, collections=collections)
    return summary


def plot_many(name, plot_func, in_tensors,
              max_outputs=3, collections=None, **kwargs):
    """
    Create a tensorflow op that outputs a `Summary` protocol buffer,
    where plots could be drawn in a batch manner. This is a batch version
    of `tfplot.summary.plot()`.

    Specifically, all the input tensors `in_tensors` to `plot_func` is
    assumed to have the same batch size. Tensors corresponding to a single
    batch element will be passed to `plot_func` as input.

    The resulting `Summary` contains multiple (up to `max_outputs`) image
    summary values, each of which contains a plot rendered by `plot_func`.

    Args:
      name: The name of scope for the generated ops and the summary op.
        Will also serve as a series name prefix in TensorBoard.
      plot_func: A python function or callable, specifying the plot operation
        as in `tfplot.ops.plot()`. See the documentation at `plot()`.
      in_tensors: A list of `Tensor` objects, the input to `plot_func()`
        but each in a batch.
      max_outputs: Max number of batch elements to generate plots for.
      collections: Optional list of ops.GraphKeys.  The collections to add the
        sumamry to.  Defaults to [_ops.GraphKeys.SUMMARIES]
      kwargs: Optional keyword arguments passed to `plot()`.

    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer (tensorflow operation).
    """

    with tf.name_scope(name=name) as scope:
        im_batch = ops.plot_many(plot_func, in_tensors, name=scope, **kwargs)
        summary = tf.summary.image(name="ImageSummary", tensor=im_batch,
                                   max_outputs=max_outputs,
                                   collections=collections)
    return summary
