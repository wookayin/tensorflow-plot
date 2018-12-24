'''Some predefined plot functions.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .wrapper import autowrap


__all__ = (
    'probmap',
    'probmap_simple',
    'batch',
)


@autowrap
def probmap(x, cmap='jet', colorbar=True,
            vmin=None, vmax=None, axis=True, ax=None):
    '''
    Display a heatmap in color. The resulting op will be a RGBA image Tensor.

    Args:
      x: A 2-D image-like tensor to draw.
      cmap: Matplotlib colormap. Defaults 'jet'
      axis: If True (default), x-axis and y-axis will appear.
      colorbar: If True (default), a colorbar will be placed on the right.
      vmin: A scalar. Minimum value of the range. See ``matplotlib.axes.Axes.imshow``.
      vmax: A scalar. Maximum value of the range. See ``matplotlib.axes.Axes.imshow``.

    Returns:
        A `uint8` `Tensor` of shape ``(?, ?, 4)`` containing the resulting plot.
    '''
    assert ax is not None, "autowrap did not set ax"

    axim = ax.imshow(x, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        ax.figure.colorbar(axim)
    if not axis:
        ax.axis('off')

    if not axis and not colorbar:
        ax.figure.subplots_adjust(0, 0, 1, 1)
    else:
        ax.figure.tight_layout()


def probmap_simple(x, **kwargs):
    '''
    Display a heatmap in color, but only displays the image content.
    The resulting op will be a RGBA image Tensor.

    It reduces to ``probmap`` having `colorbar` and `axis` off.
    See the documentation of ``probmap`` for available arguments.
    '''
    # pylint: disable=unexpected-keyword-arg
    return probmap(x,
                   colorbar=kwargs.pop('colorbar', False),
                   axis=kwargs.pop('axis', False),
                   figsize=kwargs.pop('figsize', (3, 3)),
                   **kwargs)
    # pylint: enable=unexpected-keyword-arg


def batch(func):
    '''
    Make an autowrapped plot function (... -> RGBA tf.Tensor) work in a batch
    manner.

    Example:

      >>> p
      Tensor("p:0", shape=(batch_size, 16, 16, 4), dtype=uint8)
      >>> tfplot.contrib.batch(tfplot.contrib.probmap)(p)
      Tensor("probmap/PlotImages:0", shape=(batch_size, ?, ?, 4), dtype=uint8)
    '''
    if not hasattr(func, '__unwrapped__'):
        raise ValueError("The given function is not wrapped with tfplot.autowrap()!")

    func = func.__unwrapped__
    return autowrap(func, batch=True)
