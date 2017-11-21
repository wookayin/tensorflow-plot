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



def wrap(plot_func, _sentinel=None,
         batch=False, name=None,
         **kwargs):
    '''
    Wrap a plot function as a TensorFlow operation. It will return a python
    function that creates a TensorFlow plot operation applying the arguments
    as input.

    For example, if ``plot_func`` is a python function that takes two
    arrays as input, and draw a plot by returning a matplotlib Figure,
    we can wrap this function as a `Tensor` factory, such as:

      >>> tf_plot = tfplot.wrap(plot_func, name="MyPlot", batch=True)
      >>> # x, y = get_batch_inputs(batch_size=4, ...)

      >>> plot_x = tf_plot(x)
      Tensor("MyPlot:0", shape=(4, ?, ?, 4), dtype=uint8)
      >>> plot_y = tf_plot(y)
      Tensor("MyPlot_1:0", shape=(4, ?, ?, 4), dtype=uint8)

    Args:
      plot_func: A python function or callable to wrap. See the documentation
        of :func:`tfplot.plot()` for details.
      batch: If True, all the tensors passed as argument will be
        assumed to be batched. Default value is False.
      name: A default name for the operation (optional). If not given, the
        name of ``plot_func`` will be used.
      kwargs: An optional kwargs that will be passed by default to
        ``plot_func``.

    Returns:
      A python function that will create a TensorFlow plot operation,
      passing the provided arguments.
    '''

    if not hasattr(plot_func, '__call__'):
        raise TypeError("plot_func should be callable")
    if _sentinel is not None:
        raise RuntimeError("Invalid call: it can have only one unnamed argument, " +
                           "please pass named arguments for batch, name, etc.")

    if name is None:
        name = _clean_name(plot_func.__name__)

    def _wrapped_fn(*args, **kwargs_call):
        _plot = plot_many if batch else plot
        _name = kwargs_call.pop('name', name)
        return _plot(plot_func, list(args), name=_name,
                     **_merge_kwargs(kwargs, kwargs_call))

    _wrapped_fn.__name__ = 'wrapped_fn[%s]' % plot_func
    return _wrapped_fn


def wrap_axesplot(axesplot_func, _sentinel=None,
                  batch=False, name=None,
                  figsize=None, tight_layout=False, **kwargs):
    '''
    Wrap an axesplot function as a TensorFlow operation.  It will return a
    python function that creates a TensorFlow plot operation applying the
    arguments as input.

    An axesplot function ``axesplot_func`` can be either:

    - an unbounded method of matplotlib `Axes` (or `AxesSubplot`) class,
      such as ``Axes.scatter()`` and ``Axes.text()``, etc, or
    - a simple python function that takes the named argument ``ax``,
      of type `Axes` or `AxesSubplot`, on which the plot will be drawn.
      Some good examples of this family includes ``seaborn.heatmap(ax=...)``.

    The resulting function can be used as a Tensor factory. When the created
    tensorflow plot op is being executed, a new matplotlib figure which
    consists of a single `AxesSubplot` will be created, and the axes plot
    will be used as an argument for ``axesplot_func``. For example,

      >>> import seaborn.apionly as sns
      >>> tf_heatmap = tfplot.wrap_axesplot(sns.heatmap, name="HeatmapPlot", figsize=(4, 4), cmap='jet')

      >>> plot_op = tf_heatmap(attention_map, cmap)
      Tensor(HeatmapPlot:0", shape=(?, ?, 4), dtype=uint8)

    Args:
      axesplot_func: An unbounded method of matplotlib `Axes` or `AxesSubplot`,
          or a python function or callable which has the `ax` parameter for
          specifying the axis to draw on.
      batch: If True, all the tensors passed as argument will be
        assumed to be batched. Default value is False.
      name: A default name for the operation (optional). If not given, the
        name of ``axesplot_func`` will be used.
      figsize: The figure size for the figure to be created.
      tight_layout: If True, the resulting figure will have no margins for
        axis. Equivalent to calling ``fig.subplots_adjust(0, 0, 1, 1)``.
      kwargs: An optional kwargs that will be passed by default to
        ``axesplot_func``.

    Returns:
      A python function that will create a TensorFlow plot operation,
      passing the provied arguments and a new instance of `AxesSubplot` into
      ``axesplot_func``.
    '''

    if not hasattr(axesplot_func, '__call__'):
        raise TypeError("axesplot_func should be callable")
    if _sentinel is not None:
        raise RuntimeError("Invalid call: it can have only one unnamed argument, " +
                           "please pass named arguments for batch, name, etc.")

    def _create_subplots():
        if figsize is not None:
            fig, ax = figure.subplots(figsize=figsize)
        else:
            fig, ax = figure.subplots()

        if tight_layout:
            fig.subplots_adjust(0, 0, 1, 1)
        return fig, ax

    # (1) instance method of Axes -- ax.xyz()
    def _fig_axesplot_method(*args, **kwargs_call):
        fig, ax = _create_subplots()
        axesplot_func.__get__(ax)(*args, **_merge_kwargs(kwargs, kwargs_call))
        return fig

    # (2) xyz(ax=...) style
    def _fig_axesplot_fn(*args, **kwargs_call):
        fig, ax = _create_subplots()
        axesplot_func(*args, ax=ax, **_merge_kwargs(kwargs, kwargs_call))
        return fig

    method_class = util.get_class_defining_method(axesplot_func)
    if method_class is not None and issubclass(method_class, Axes):
        # (1) Axes.xyz()
        if hasattr(axesplot_func, '__self__') and axesplot_func.__self__:
            raise ValueError("axesplot_func should be a unbound method of " +
                             "Axes or AxesSubplot, but given a bound method " +
                             str(axesplot_func))
        fig_axesplot_func = _fig_axesplot_method
    else:
        # (2) xyz(ax=...)
        if 'ax' not in util.getargspec(axesplot_func).args:
            raise TypeError("axesplot_func must take 'ax' parameter to specify Axes")
        fig_axesplot_func = _fig_axesplot_fn

    if name is None:
        name = _clean_name(axesplot_func.__name__)

    def _wrapped_factory_fn(*args, **kwargs_call):
        _plot = plot_many if batch else plot
        _name = kwargs_call.pop('name', name)
        return _plot(fig_axesplot_func, list(args), name=_name,
                     **kwargs_call)

    _wrapped_factory_fn.__name__ = 'wrapped_axesplot_fn[%s]' % axesplot_func
    return _wrapped_factory_fn


def _clean_name(s):
    """
    Convert a string to a valid variable, function, or scope name.
    """
    return re.sub('[^0-9a-zA-Z_]', '', s)


def _merge_kwargs(kwargs, kwargs_new):
    kwargs = kwargs.copy()
    kwargs.update(kwargs_new)
    return kwargs



__all__ = (
    'plot',
    'plot_many',
    'wrap',
    'wrap_axesplot',
)
