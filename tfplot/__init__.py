from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
if not matplotlib.rcParams.get('backend', None):
    matplotlib.use('Agg')

from .ops import plot, plot_many
from .wrapper import wrap, wrap_axesplot
from .wrapper import autowrap

from .figure import subplots
from . import summary

from matplotlib.figure import Figure
from matplotlib.axes import Axes


__version__ = '0.3.0'
