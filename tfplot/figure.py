''' Figure utilities. '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg


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
    'to_array',
)
