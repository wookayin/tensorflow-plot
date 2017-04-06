''' Figure utilities. '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import types

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

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
        fig.subplots = types.MethodType(mpl_figure.subplots, fig, FigureClass)

    axs = fig.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey,
                       squeeze=squeeze, subplot_kw=subplot_kw,
                       gridspec_kw=gridspec_kw)
    return fig, axs


# inherit and append a part of the docstring from pyplot.subplots()
subplots.__doc__ += plt.subplots.__doc__[plt.subplots.__doc__.find('Parameters'):]\
    .replace('plt.subplots', 'tfplot.subplots')


def to_array(fig):
    """
    Convert a matplotlib figure `fig` into a 3D numpy array.

    A typical usage:

      ```python
      fig, ax = plt.subplots(figsize=(4, 4))
      # draw whatever, e.g. ax.text(0.5, 0.5, "text")
      im = to_array(fig)   # [288, 288, 3]
      ```

    Args:
      fig: A `matplotlib.figure.Figure` object.
    """

    #assert fig.canvas is not None, \
    #    'fig must have canvas -- has it been created by plt.figure() ?'
    if fig.canvas is None:
        # attach a new canvas
        FigureCanvasAgg(fig)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape((h, w, 3))
    return img


__all__ = (
    'subplots',
    'to_array',
)
