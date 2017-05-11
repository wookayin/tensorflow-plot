''' Summary Op utilities. '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import ops


def plot(name, plot_func, in_tensors, collections=None, **kwargs):
    """
    Create a TensorFlow op that outpus a `Summary` protocol buffer,
    to which a single plot operation is executed (i.e. image summary).

    Basically, it is a one-liner wrapper of ``tfplot.ops.plot()`` and
    ``tf.summary.image()`` calls.

    The generated `Summary` object contains single image summary value
    of the image of the plot drawn.

    Args:
      name: The name of scope for the generated ops and the summary op.
        Will also serve as a series name prefix in TensorBoard.
      plot_func: A python function or callable, specifying the plot operation
        as in :func:`tfplot.plot`. See the documentation at :func:`tfplot.plot`.
      in_tensors: A list of `Tensor` objects, as in :func:`~tfplot.plot`.
      collections: Optional list of ``ops.GraphKeys``.  The collections to add the
        summary to.  Defaults to ``[_ops.GraphKeys.SUMMARIES]``.
      kwargs: Optional keyword arguments passed to :func:`~tfplot.plot`.

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
    Create a TensorFlow op that outputs a `Summary` protocol buffer,
    where plots could be drawn in a batch manner. This is a batch version
    of :func:`tfplot.summary.plot`.

    Specifically, all the input tensors ``in_tensors`` to ``plot_func`` is
    assumed to have the same batch size. Tensors corresponding to a single
    batch element will be passed to ``plot_func`` as input.

    The resulting `Summary` contains multiple (up to ``max_outputs``) image
    summary values, each of which contains a plot rendered by ``plot_func``.

    Args:
      name: The name of scope for the generated ops and the summary op.
        Will also serve as a series name prefix in TensorBoard.
      plot_func: A python function or callable, specifying the plot operation
        as in :func:`tfplot.plot`. See the documentation at :func:`tfplot.plot`.
      in_tensors: A list of `Tensor` objects, the input to ``plot_func``
        but each in a batch.
      max_outputs: Max number of batch elements to generate plots for.
      collections: Optional list of ``ops.GraphKeys``.  The collections to add the
        sumamry to.  Defaults to ``[_ops.GraphKeys.SUMMARIES]``.
      kwargs: Optional keyword arguments passed to :func:`~tfplot.plot`.

    Returns:
      A scalar `Tensor` of type `string`. The serialized `Summary` protocol
      buffer (tensorflow operation).
    """

    with tf.name_scope(name=name) as scope:
        im_batch = ops.plot_many(plot_func, in_tensors, name=scope,
                                 max_outputs=max_outputs,
                                 **kwargs)
        summary = tf.summary.image(name="ImageSummary", tensor=im_batch,
                                   max_outputs=max_outputs,
                                   collections=collections)
    return summary


def wrap(plot_func, _sentinel=None,
         batch=False, name=None, **kwargs):
    '''
    Wrap a plot function as a TensorFlow summary builder. It will return a
    python function that creates a TensorFlow op which evaluates to
    ``Summary`` protocol buffer with image.

    The resulting function (say ``summary_wrapped``) will have the following
    signature:

    .. code-block:: python

       summary_wrapped(name, tensor, # [more input tensors ...],
                       max_outputs=3, collections=None)

    Examples:

      Given a plot function which returns a matplotlib `Figure`,

      >>> def figure_heatmap(data, cmap='jet'):
      >>>     fig, ax = tfplot.subplots()
      >>>     ax.imshow(data, cmap=cmap)
      >>>     return fig

      we can wrap it as a summary builder function:

      >>> summary_heatmap = tfplot.summary.wrap(figure_heatmap, batch=True)

      Now, when building your computation graph, call it to build summary ops
      like ``tf.summary.image``:

      >>> heatmap_tensor
      <tf.Tensor 'heatmap_tensor:0' shape=(16, 128, 128) dtype=float32>
      >>>
      >>> summary_heatmap("heatmap/original", heatmap_tensor)
      >>> summary_heatmap("heatmap/cmap_gray", heatmap_tensor, cmap=gray)
      >>> summary_heatmap("heatmap/no_default_collections", heatmap_tensor, collections=[])


    Args:
      plot_func: A python function or callable to wrap. See the documentation
        of :func:`tfplot.plot` for details.
      batch: If True, all the tensors passed as argument will be
        assumed to be batched. Default value is False.
      name: A default name for the plot op (optional). If not given, the
        name of ``plot_func`` will be used.
      kwargs: Optional keyword arguments that will be passed by default to
        :func:`~tfplot.plot`.

    Returns:
      A python function that will create a TensorFlow summary operation,
      passing the provided arguments into plot op.
    '''

    if _sentinel is not None:
        raise RuntimeError("Invalid call: it can have only one unnamed argument, " +
                           "please pass named arguments for batch, name, etc.")

    factory_fn = ops.wrap(plot_func, batch=batch, name=name, **kwargs)
    def _summary_fn(summary_name, *args, **kwargs_call):
        plot_op = factory_fn(*args, **kwargs_call)
        return tf.summary.image(summary_name, plot_op,
                                max_outputs=kwargs_call.pop('max_outputs', 3),
                                collections=kwargs_call.pop('collections', None),
                                )

    _summary_fn.__name__ = 'summary_fn[%s]' % plot_func
    return _summary_fn


__all__ = (
    'wrap',
    'plot',
    'plot_many',
)
