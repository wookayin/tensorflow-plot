''' Main plot operations. '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import re

from . import figure
from . import util
from .ops import plot, plot_many
from .util import merge_kwargs

from biwrap import biwrap

from matplotlib.figure import Figure
from matplotlib.axes import Axes


# a dummy marker object to avoid pylint issues on decorators.
REQUIRED = object()


@biwrap
def wrap(plot_func=REQUIRED, _sentinel=None,
         batch=False, name=None,
         **kwargs):
    '''
    Wrap a plot function as a TensorFlow operation. It will return a python
    function that creates a TensorFlow plot operation applying the arguments
    as input. It can be also used as a decorator.

    For example:

      >>> @tfplot.wrap
      >>> def plot_imshow(img):
      >>>    fig, ax = tfplot.subplots()
      >>>    ax.imshow(img)
      >>>    return fig
      >>>
      >>> plot_imshow(an_image_tensor)
      Tensor("plot_imshow:0", shape=(?, ?, 4), dtype=uint8)

    Or, if ``plot_func`` is a python function that takes  numpy arrays as input
    and draw a plot by returning a matplotlib Figure,
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
        ``plot_func`` when executed inside a TensorFlow graph.

    Returns:
      A python function that will create a TensorFlow plot operation,
      passing the provided arguments.
    '''

    if plot_func == REQUIRED:
        raise TypeError("Required argument 'plot_func' (pos 1) not found")
    if not hasattr(plot_func, '__call__'):
        raise TypeError("plot_func should be callable")
    if _sentinel is not None:
        raise RuntimeError("Invalid call: it can have only one positional argument, " +
                           "please pass named arguments for batch, name, etc.")

    if name is None:
        name = _clean_name(plot_func.__name__)

    @functools.wraps(plot_func)
    def _wrapped_fn(*args, **kwargs_call):
        _plot = plot_many if batch else plot
        _name = kwargs_call.pop('name', name)
        return _plot(plot_func, list(args), name=_name,
                     **merge_kwargs(kwargs, kwargs_call))

    _wrapped_fn.__name__ = 'wrap[%s]' % plot_func.__name__
    if hasattr(plot_func, '__qualname__'):
        _wrapped_fn.__qualname__ = 'wrap[%s.%s]' % (plot_func.__module__, plot_func.__qualname__)
    return _wrapped_fn


def wrap_axesplot(axesplot_func, _sentinel=None,
                  batch=False, name=None,
                  figsize=None, tight_layout=False, **kwargs):
    '''
    DEPRECATED: Use ``tfplot.autowrap()`` instead. Will be removed
    in the next version.

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
        axesplot_func.__get__(ax)(*args, **merge_kwargs(kwargs, kwargs_call))
        return fig

    # (2) xyz(ax=...) style
    def _fig_axesplot_fn(*args, **kwargs_call):
        fig, ax = _create_subplots()
        axesplot_func(*args, ax=ax, **merge_kwargs(kwargs, kwargs_call))
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
        if 'ax' not in util.getargspec_allargs(axesplot_func):
            raise TypeError("axesplot_func must take 'ax' parameter to specify Axes")
        fig_axesplot_func = _fig_axesplot_fn

    if name is None:
        name = _clean_name(axesplot_func.__name__)

    @functools.wraps(axesplot_func)
    def _wrapped_factory_fn(*args, **kwargs_call):
        _plot = plot_many if batch else plot
        _name = kwargs_call.pop('name', name)
        return _plot(fig_axesplot_func, list(args), name=_name,
                     **kwargs_call)

    _wrapped_factory_fn.__name__ = 'wrapped_axesplot_fn[%s]' % axesplot_func
    return _wrapped_factory_fn


@biwrap
def autowrap(plot_func=REQUIRED, _sentinel=None,
             batch=False, name=None, figsize=None, tight_layout=False,
             **kwargs_default):
    """
    Wrap a function as a TensorFlow operation similar to :func:`tfplot.wrap()`
    (as a decorator or with normal function call), but provides with additional
    features such as auto-creating matplotlib figures.

      - (``fig``, ``ax``) matplotlib objects are automatically created and
        injected given that `plot_func` has a keyword argument named ``fig``
        and/or ```ax``. In such cases, we do not need to manually call
        :func:`tfplot.subplots()` to create matplotlib figure/axes objects.
        If a manual creation of ``fig, ax`` is forced, please consider
        using :func:`tfplot.wrap()` instead.

      - It can automatically handle return values of the provided `plot_func`
        function. If it returns nothing (None) but ``fig`` was automatically
        injected then the resulting figure will be drawn, or returns ``Axes``
        then the associated ``Figure`` will be used.

    Example:

      >>> @tfplot.autowrap(figsize=(3, 3))
      >>> def plot_imshow(img, *, fig, ax):
      >>>    ax.imshow(img)
      >>>
      >>> plot_imshow(an_image_tensor)
      Tensor("plot_imshow:0", shape=(?, ?, 4), dtype=uint8)

    Args:
      plot_func: A python function or callable to wrap. See the documentation
        of :func:`tfplot.plot()` for details. Additionally, if this function
        has a parameter named ``fig`` and/or ``ax``, new instances of
        ``Figure`` and/or ``AxesSubplot`` will be created and passed.

      batch: If True, all the tensors passed as argument will be
        assumed to be batched. Default value is False.
      name: A default name for the operation (optional). If not given, the
        name of ``plot_func`` will be used.

      figsize: The figure size for the figure to be created.
      tight_layout: If True, the resulting figure will have no margins for
        axis. Equivalent to calling ``fig.subplots_adjust(0, 0, 1, 1)``.
      kwargs_default: An optimal kwargs that will be passed by default to
        ``plot_func`` when executed inside a TensorFlow graph.

    """

    if plot_func == REQUIRED:
        raise TypeError("Required argument 'plot_func' (pos 1) not found")

    # check if func has `fig` or `ax` parameter
    fig_ax_mode = tuple(
        arg_name for arg_name in ('ax', 'fig') \
        if arg_name in util.getargspec_allargs(plot_func)
    )

    # check if func is an instance method of Axes, e.g. ax.scatter()
    method_class = util.get_class_defining_method(plot_func)
    is_axesplot_bind = False
    if method_class is not None and issubclass(method_class, Axes):
        if hasattr(plot_func, '__self__') and plot_func.__self__:
            raise ValueError("plot_func should be a unbound method of " +
                             "Axes or AxesSubplot, but given a bound method " +
                             str(plot_func))
        is_axesplot_bind = True


    def _create_subplots(_kwargs):
        # recognize overriding parameters for creating subplots, e.g. figsize
        _figsize = _kwargs.pop('figsize', figsize)

        fig, ax = figure.subplots(figsize=_figsize)
        return fig, ax

    # Decorates `plot_func` with additional aspects
    # (e.g. auto-injection, return value handling)
    @functools.wraps(plot_func)
    def _wrapped_plot_fn(*args, **kwargs_call):
        # (1) auto-inject fig, ax
        if fig_ax_mode or is_axesplot_bind:
            # auto-create rather than manually
            fig, ax = _create_subplots(kwargs_call)
        fig_ax_kwargs = dict(
            ([('fig', fig)] if 'fig' in fig_ax_mode else []) + \
            ([('ax', ax)] if 'ax' in fig_ax_mode else [])
        )

        # (2) body
        if is_axesplot_bind:   # e.g. Axesplot.scatter -> bind 'ax' as self
            ret = plot_func.__get__(ax)(*args, **kwargs_call)
        else:
            ret = plot_func(*args, **merge_kwargs(kwargs_call, fig_ax_kwargs))  # TODO conflict??

        # (3) return value handling
        if ret is None and fig_ax_mode:
            # even if the function doesn't return anything,
            # but we know that `fig` is what we just need to draw.
            ret = fig
        elif is_axesplot_bind:
            # for Axesplot methods, ignore the return value
            # and use the fig instance created before as target figure
            ret = fig
        elif isinstance(ret, Axes):
            ret = fig = ret.figure
        elif isinstance(ret, Figure):
            fig = ret

        if tight_layout:
            fig.subplots_adjust(0, 0, 1, 1)

        return ret

    # return the wrapper (a factory of Tensor)
    _wrapped_fn = wrap(_wrapped_plot_fn, batch=batch, name=name,
                       **kwargs_default)

    _wrapped_fn.__name__ = 'autowrap[%s]' % plot_func.__name__
    if hasattr(plot_func, '__qualname__'):
        _wrapped_fn.__qualname__ = 'autowrap[%s.%s]' % (plot_func.__module__, plot_func.__qualname__)


    # expose the unwrapped python function as well
    _wrapped_fn.__unwrapped__ = plot_func

    return _wrapped_fn


def _clean_name(s):
    """
    Convert a string to a valid variable, function, or scope name.
    """
    return re.sub('[^0-9a-zA-Z_]', '', s)


__all__ = (
    'wrap',
    'wrap_axesplot',
    'autowrap',
)
