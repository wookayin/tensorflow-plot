""" A cherry-pick from matplotlib implementation """

# Copyright (c) 2012-2013 Matplotlib Development Team; All Rights Reserved
# http://matplotlib.org/users/license.html
#
# All the code in this file was brought from matplotlib


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import warnings

import numpy as np

from matplotlib.gridspec import GridSpec


# should be attached to matplotlib.figure.Figure()
# @see https://github.com/matplotlib/matplotlib/pull/5146
def subplots(self, nrows=1, ncols=1, sharex=False, sharey=False,
             squeeze=True, subplot_kw=None, gridspec_kw=None):
    """
    Add a set of subplots to this figure.
    Parameters
    ----------
    nrows, ncols : int, default: 1
        Number of rows/cols of the subplot grid.
    sharex, sharey : bool or {'none', 'all', 'row', 'col'}, default: False
        Controls sharing of properties among x (`sharex`) or y (`sharey`)
        axes:
            - True or 'all': x- or y-axis will be shared among all
                subplots.
            - False or 'none': each subplot x- or y-axis will be
                independent.
            - 'row': each subplot row will share an x- or y-axis.
            - 'col': each subplot column will share an x- or y-axis.
        When subplots have a shared x-axis along a column, only the x tick
        labels of the bottom subplot are visible.  Similarly, when
        subplots have a shared y-axis along a row, only the y tick labels
        of the first column subplot are visible.
    squeeze : bool, default: True
        - If True, extra dimensions are squeezed out from the returned
            axis object:
            - if only one subplot is constructed (nrows=ncols=1), the
                resulting single Axes object is returned as a scalar.
            - for Nx1 or 1xN subplots, the returned object is a 1D numpy
                object array of Axes objects are returned as numpy 1D
                arrays.
            - for NxM, subplots with N>1 and M>1 are returned as a 2D
                arrays.
        - If False, no squeezing at all is done: the returned Axes object
            is always a 2D array containing Axes instances, even if it ends
            up being 1x1.
    subplot_kw : dict, default: {}
        Dict with keywords passed to the
        :meth:`~matplotlib.figure.Figure.add_subplot` call used to create
        each subplots.
    gridspec_kw : dict, default: {}
        Dict with keywords passed to the
        :class:`~matplotlib.gridspec.GridSpec` constructor used to create
        the grid the subplots are placed on.
    Returns
    -------
    ax : single Axes object or array of Axes objects
        The added axes.  The dimensions of the resulting array can be
        controlled with the squeeze keyword, see above.
    See Also
    --------
    pyplot.subplots : pyplot API; docstring includes examples.
    """

    # for backwards compatibility
    if isinstance(sharex, bool):
        sharex = "all" if sharex else "none"
    if isinstance(sharey, bool):
        sharey = "all" if sharey else "none"
    share_values = ["all", "row", "col", "none"]
    if sharex not in share_values:
        # This check was added because it is very easy to type
        # `subplots(1, 2, 1)` when `subplot(1, 2, 1)` was intended.
        # In most cases, no error will ever occur, but mysterious behavior
        # will result because what was intended to be the subplot index is
        # instead treated as a bool for sharex.
        if isinstance(sharex, int):
            warnings.warn(
                "sharex argument to subplots() was an integer. "
                "Did you intend to use subplot() (without 's')?")

        raise ValueError("sharex [%s] must be one of %s" %
                            (sharex, share_values))
    if sharey not in share_values:
        raise ValueError("sharey [%s] must be one of %s" %
                            (sharey, share_values))
    if subplot_kw is None:
        subplot_kw = {}
    if gridspec_kw is None:
        gridspec_kw = {}

    gs = GridSpec(nrows, ncols, **gridspec_kw)

    # Create array to hold all axes.
    axarr = np.empty((nrows, ncols), dtype=object)
    for row in range(nrows):
        for col in range(ncols):
            shared_with = {"none": None, "all": axarr[0, 0],
                            "row": axarr[row, 0], "col": axarr[0, col]}
            subplot_kw["sharex"] = shared_with[sharex]
            subplot_kw["sharey"] = shared_with[sharey]
            axarr[row, col] = self.add_subplot(gs[row, col], **subplot_kw)

    # turn off redundant tick labeling
    if sharex in ["col", "all"]:
        # turn off all but the bottom row
        for ax in axarr[:-1, :].flat:
            for label in ax.get_xticklabels():
                label.set_visible(False)
            ax.xaxis.offsetText.set_visible(False)
    if sharey in ["row", "all"]:
        # turn off all but the first column
        for ax in axarr[:, 1:].flat:
            for label in ax.get_yticklabels():
                label.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)

    if squeeze:
        # Discarding unneeded dimensions that equal 1.  If we only have one
        # subplot, just return it instead of a 1-element array.
        return axarr.item() if axarr.size == 1 else axarr.squeeze()
    else:
        # Returned axis array will be always 2-d, even if nrows=ncols=1.
        return axarr
