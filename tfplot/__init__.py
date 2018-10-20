from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

from .ops import plot, plot_many, wrap, wrap_axesplot
from .figure import subplots
from . import summary

from matplotlib.figure import Figure
from matplotlib.axes import Axes


__version__ = '0.2.0'
