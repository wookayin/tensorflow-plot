''' Figure utilities. '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import types

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import six
from io import BytesIO
from tensorflow import Summary

from . import mpl_figure


def subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True,
             subplot_kw=None, gridspec_kw=None, **fig_kw):
    """
    Create a figure and a set of subplots, as in `pyplot.subplots()`.

    It works almost similar to `pyplot.subplots()`, but differ from it in that
    it does not involve any side effect as pyplot does (e.g. modifying thread
    states such as current figure or current subplot).

    (docstrings inherited from `matplotlib.pyplot.subplots`)

    """
    FigureClass = fig_kw.pop('FigureClass', Figure)
    fig = FigureClass(**fig_kw)

    # attach a new Agg canvas
    if fig.canvas is None:
        FigureCanvasAgg(fig)

    # create subplots, e.g. fig.subplots() in matplotlib 2.1+
    if not hasattr(fig, 'subplots'):
        if six.PY2:
            fig.subplots = types.MethodType(mpl_figure.subplots, fig, FigureClass)
        else:
            fig.subplots = types.MethodType(mpl_figure.subplots, fig)

    axs = fig.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey,
                       squeeze=squeeze, subplot_kw=subplot_kw,
                       gridspec_kw=gridspec_kw)
    return fig, axs


# inherit and append a part of the docstring from pyplot.subplots()
subplots.__doc__ += plt.subplots.__doc__[plt.subplots.__doc__.find('Parameters'):]\
    .replace('plt.subplots', 'tfplot.subplots')


def to_array(fig):
    """
    Convert a matplotlib figure ``fig`` into a 3D numpy array.

    Example:

      >>> fig, ax = tfplot.subplots(figsize=(4, 4))
      >>> # draw whatever, e.g. ax.text(0.5, 0.5, "text")

      >>> im = to_array(fig)   # ndarray [288, 288, 3]

    Args:
      fig: A ``matplotlib.figure.Figure`` object.

    Returns:
      A numpy ``ndarray`` of shape ``(?, ?, 3)``, containing an RGB image of
      the figure.
    """

    # attach a new canvas if not exists
    if fig.canvas is None:
        FigureCanvasAgg(fig)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape((h, w, 3))
    return img


def to_summary(fig, tag):
    """
    Convert a matplotlib figure ``fig`` into a TensorFlow Summary object
    that can be directly fed into ``Summary.FileWriter``.

    Example:

      >>> fig, ax = ...    # (as above)
      >>> summary = to_summary(fig, tag='MyFigure/image')

      >>> type(summary)
      tensorflow.core.framework.summary_pb2.Summary
      >>> summary_writer.add_summary(summary, global_step=global_step)

    Args:
      fig: A ``matplotlib.figure.Figure`` object.
      tag (string): The tag name of the created summary.

    Returns:
      A TensorFlow ``Summary`` protobuf object containing the plot image
      as a image summary.
    """
    if not isinstance(tag, six.string_types):
        raise TypeError("tag must be a string type")

    # attach a new canvas if not exists
    if fig.canvas is None:
        FigureCanvasAgg(fig)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # get PNG data from the figure
    png_buffer = BytesIO()
    fig.canvas.print_png(png_buffer)
    png_encoded = png_buffer.getvalue()
    png_buffer.close()

    summary_image = Summary.Image(height=h, width=w, colorspace=3,  # RGB
                                  encoded_image_string=png_encoded)
    summary = Summary(value=[Summary.Value(tag=tag, image=summary_image)])
    return summary


__all__ = (
    'to_array',
    'to_summary',
    'subplots',
)
