""" Module containing the functions to produce the standard echidna
output plots - wsing ``matplotlib.pyplot`` as a backend.
"""
import matplotlib.pyplot as plt

import logging

_logger = logging.getLogger("plot")


def plot_projection(spectra, dimension, fig_num=1,
                    show_plot=True, subplots_kw={}):
    """ Plot the spectra as projected onto the dimension.

    For example dimension ``energy_mc`` will plot the spectra as
    projected onto the energy_mc dimension.

    Args:
      spectra (:class:`echidna.core.spectra`): The spectrum to plot
      dimension (string): The dimension onto which to project the
        spectrum
      fig_num (int, optional): The number of the figure. If you are
        producing multiple spectral plots automatically, this can be
        useful to ensure pyplot treats each plot as a separate figure.
        Default is 1.
      show_plot (bool, optional): Displays the plot if true. Default is
        True.
      subplots_kw (dict, optional): Dict with kewyords passed to the
        :func:`plt.subplots` function and the :func:`plt.figure`
        function. May also include dicts ``subplot_kw`` to pass to the
        :func:`add_subplot` function and ``gridspec_kw`` to pass to the
        :class:`GridSpec` instance that creates the grid for the
        subplots. See their documentation for further details

    Returns:
      plt.figure: Plot of the projection
    """
    fig, ax = plt.subplots(num=fig_num, **subplots_kw)

    # Get parameter, x-values and bin boundaries
    config = spectra.get_config()
    par = config.get_par(dimension)
    bin_boundaries = par.get_bin_boundaries()
    xs = par.get_bin_centres()

    # Project spectrum to use as weights
    projection = spectra.project(dimension)

    # Get style
    style = spectra.get_style()

    # Make histogram
    ax.hist(xs, bin_boundaries, weights=projection, histtype="step", **style)

    # Make it pretty!
    width = par.get_width()
    ax.set_xlabel("%s (%s)" % (dimension, par.get_unit()))
    ax.set_ylabel("Count per %.2g %s bin" % (width, par.get_unit()))
    ax.legend()

    # If ROI has been set, shrink x-axis
    try:
        roi = spectra.get_roi(config.get_dim(dimension))
        ax.xlim(roi)
    except KeyError:
        _logger.warning("Cannot set x-limits from ROI", exc_info=1)

    if show_plot:
        plt.show()
    return fig


def plot_stats_vs_scale(limit_results, fig_num=1, show_plot=True,
                        fmt_string="bo", xlabel="Number of Signal Decays",
                        ylabel="Test Statistic", subplots_kw={}, plot_kw={}):
    """ Plots the test statistics vs signal scales

    Args:
      limit_results (:class:`echidna.fit.fit_results.LimitResults`): The
        limit_results object which contains the data
      fig_num (int, optional): The number of the figure. If you are
        producing multiple spectral plots automatically, this can be
        useful to ensure pyplot treats each plot as a separate figure.
        Default is 1.
      show_plot (bool, optionl): Displays the plot on-screen if True
        (default)
      fmt_string (string, optional): Format string specifying the style
        of the marker/line and its colour, in :func:`plt.plot`
      xlabel (string, optional): Title of the x-axis
      ylabel (string, optional): Title of the y-axis
      subplots_kw (dict, optional): Dict with kewyords passed to the
        :func:`plt.subplots` function and the :func:`plt.figure`
        function. May also include dicts ``subplot_kw`` to pass to the
        :func:`add_subplot` function and ``gridspec_kw`` to pass to the
        :class:`GridSpec` instance that creates the grid for the
        subplots. See their documentation for further details.
      plot_kw (dict, optional): Dict with keywords passed to the
        :func:`plt.plot` function. See the its documentation for more
        details.

    Returns:
      plt.figure: Plot of the projection
    """
    fig, ax = plt.subplots(num=fig_num, **subplots_kw)

    # Set x and y values
    xs = limit_results.get_scales()
    ys = limit_results.get_full_stats()

    # Plot data
    ax.plot(xs, ys, fmt_string, **plot_kw)

    # Make it pretty!
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_plot:
        plt.show()
    return fig
