""" Module containing the functions to produce the standard echidna
output plots - wsing ``matplotlib.pyplot`` as a backend.
"""
import matplotlib.pyplot as plt
import numpy

from echidna.utilities import start_logging
logger = start_logging()


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
        logger.warning("Cannot set x-limits from ROI", exc_info=1)

    if show_plot:
        plt.show()
    return fig


def spectral_plot(dimension, fig=None, fig_num=1, subplot_kw={},
                  data=None, backgrounds=[], signals=[], xlabel=None,
                  ylabel=None, legend=True, data_errors=True,
                  plot_kw={"marker": "o",
                           "markersize": 3.5, "linestyle": ""},
                  sum_bkg_style={"color": "Red",
                                 "label": "summed background"},
                  hist_kw={"histtype": "step"},
                  legend_kw={"fontsize": "small", "numpoints": 1,
                             "fancybox": True, "framealpha": 0.5},
                  show_plot=True):
    """ Produce a spectral plot, given some data and any number of
    signals or backgrounds.

    Args:
      dimension (string): Name of spectral dimension, on which to
        project spectrum data, for spectral plot.
      fig (tuple, optional): Pre-constructed
        (:class:`plt.figure`, :class:`plt.Axes`), on which to produce
        plot.
      fig_num (int, optional): The number of the figure. If you are
        producing multiple spectral plots automatically, this can be
        useful to ensure pyplot treats each plot as a separate figure.
        Default is 1.
      subplot_kwargs (dict): Keyword arguments to pass onto
        :meth:`plt.subplots`.
      data (:class:`echidna.core.spectra.Spectra`, optional): Data
        points to plot.
      backgrounds (list, optional): Containing one
        :class:`echidna.core.spectra.Spectra` for each background to
        plot.
      signals (list, optional): Containing one
        :class:`echidna.core.spectra.Spectra` for each signal to plot.
      xlabel (string, optional): Label for x-axis. Default is to use
        spectral dimension.
      ylabel (string, optional): Label for y-axis. Default is "Events
        per <width><unit>"
      legend (bool, optional): Set to False if you do not wish to add a
        legend.
      data_errors (bool, optional): If true includes Poisson error bars
        on data points.
      plot_kw (dict, optional): Keyword arguments to pass to
        :meth:`ax.plot`.
      sum_bkg_style (dict, optional): Summed background style keyword
        arguments.
      hist_kw (dict, optional): Keyword arguments to pass to
        :meth:`ax.hist`.
      legend_kw (dict, optional): Keyword arguments to pass to
        :meth:`ax.legend`.
      show_plot (bool, optionl): Displays the plot on-screen if True
        (default)

    Returns:
      tuple: fig, ax
    """
    if fig:  # unpack
        logger.warning("Using existing fig, ax for plot")
        fig, ax = fig
    else:  # construct
        logger.warning("Generating new fig, ax using plt.subplots")
        fig, ax = plt.subplots(**subplot_kw)

    # Plot data
    x_and_bins_set = False
    if data:
        x, bins, width, unit = _get_x_bins_width(data, dimension)
        x_and_bins_set = True
        kwargs = data.get_style().copy()
        kwargs.update(plot_kw)
        y = data.project(dimension)
        if data_errors:
            errors = numpy.sqrt(y)
            ax.errorbar(x, y, yerr=errors, **kwargs)
        else:
            ax.plot(x, y, **kwargs)

    # Plot backgrounds
    bkg_sum = None
    for background in backgrounds:
        if not x_and_bins_set:
            x, bins, width, unit = _get_x_bins_width(background, dimension)
            x_and_bins_set = True
        kwargs = background.get_style().copy()
        kwargs.update(hist_kw)
        ax.hist(x, bins, weights=background.project(dimension), **kwargs)
        if bkg_sum is None:
            bkg_sum = background.project(dimension)
        else:
            bkg_sum = bkg_sum + background.project(dimension)
    kwargs = sum_bkg_style.copy()
    kwargs.update(hist_kw)
    ax.hist(x, bins, weights=bkg_sum, **kwargs)

    # Plot signal
    for signal in signals:
        if not x_and_bins_set:
            x, bins, width, unit = _get_x_bins_width(signal, dimension)
            x_and_bins_set = True
        kwargs = signal.get_style().copy()
        kwargs.update(hist_kw)
        ax.hist(x, bins, weights=signal.project(dimension), **kwargs)

    # Add labels
    if not xlabel:
        xlabel = dimension
    ax.set_xlabel(xlabel)
    if not ylabel:
        ylabel = "Events per %.2g %s" % (width, unit)
    ax.set_ylabel(ylabel)

    # Plot legend
    if legend:
        ax.legend(**legend_kw)

    if show_plot:
        plt.show()

    # Return fig, ax
    return fig, ax


def _get_x_bins_width(spectrum, dimension):
    """
    Args:
      spectrum (:class:`echidna.core.spectra.Spectra`): Spectrum for
        which to get x, bins, bin_width and unit.
      dimension (string): Name of spectral dimension, on which to
        project spectrum data, for spectral plot.

    Returns:
      tuple: containing:

        * :class:`numpy.ndarray`: bin centres (x-values)
        * :class:`numpy.ndarray`: bin boundaries (bins)
        * float: bin width
        * string: unit

    """
    par = spectrum.get_config().get_par(dimension)
    x = par.get_bin_centres()
    bins = par.get_bin_boundaries()
    width = par.get_width()
    unit = par.get_unit()
    return x, bins, width, unit


def plot_stats_vs_scale(limit_results, fig_num=1, subplots_kw={},
                        fmt_string="bo", plot_kw={}, chi_squared=2.71,
                        limit=None, xlabel="Number of signal secays",
                        ylabel="Test statistic", show_plot=True):
    """ Plots the test statistics vs signal scales

    Args:
      limit_results (:class:`echidna.fit.fit_results.LimitResults`): The
        limit_results object which contains the data
      fig_num (int, optional): The number of the figure. If you are
        producing multiple spectral plots automatically, this can be
        useful to ensure pyplot treats each plot as a separate figure.
        Default is 1.
      subplots_kw (dict, optional): Dict with kewyords passed to the
        :func:`plt.subplots` function and the :func:`plt.figure`
        function. May also include dicts ``subplot_kw`` to pass to the
        :func:`add_subplot` function and ``gridspec_kw`` to pass to the
        :class:`GridSpec` instance that creates the grid for the
        subplots. See their documentation for further details.
      fmt_string (string, optional): Format string specifying the style
        of the marker/line and its colour, in :func:`plt.plot`
      plot_kw (dict, optional): Dict with keywords passed to the
        :func:`plt.plot` function. See the its documentation for more
        details.
      chi_squared (float, optional): Chi-squared corresponding to limit
      limit (float, optional): Limit value
      xlabel (string, optional): Title of the x-axis
      ylabel (string, optional): Title of the y-axis
      show_plot (bool, optionl): Displays the plot on-screen if True
        (default)

    Returns:
      plt.figure: Plot of the projection
    """
    fig, ax = plt.subplots(num=fig_num, **subplots_kw)

    # Set x and y values
    xs = limit_results.get_scales()
    stats = limit_results.get_full_stats()
    ys = stats - numpy.nanmin(stats)

    # Plot data
    ax.plot(xs, ys, fmt_string, **plot_kw)

    # Plot chi-squared lines
    if limit:
        ax.hlines(chi_squared, 0., limit, linestyle="dashed")
        ax.vlines(limit, 0, 2.71, linestyle="dashed")

    # Make it pretty!
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_plot:
        plt.show()
    return fig


def plot_chi_squared_contour(xs, ys, chi_squareds, fig=None,
                             fig_num=1, subplot_kw={},
                             contour_kw={"levels": [1., 2., 3.],
                                         "colors": "k"},
                             xlabel="x", ylabel="y", show_plot=True):
    """ Produces a contour plot of the 2D chi-squared surface.

    Args:
      xs (numpy.ndarray): List of x-axis points on the surface grid
      ys (numpy.ndarray): List of y-axis points on the surface grid
      chi_squareds (numpy.ndarray): Two dimensional chi-squared array
        of the chi-squared surface
      fig (tuple, optional): Pre-constructed
        (:class:`plt.figure`, :class:`plt.Axes`), on which to produce
        plot.
      fig_num (int, optional): The number of the figure. If you are
        producing multiple spectral plots automatically, this can be
        useful to ensure pyplot treats each plot as a separate figure.
        Default is 1.
      subplot_kw (dict): Keyword arguments to pass onto
        :meth:`plt.subplots`
      contour_kw (dict): Keyword arguments to pass onto
        :meth:`plt.contour`
      xlabel (string, optional): Title of the x-axis
      ylabel (string, optional): Title of the y-axis
      show_plot (bool, optionl): Displays the plot on-screen if True

    Returns:
      tuple: fig, ax

    """
    if fig:  # unpack
        logger.warning("Using existing fig, ax for plot")
        fig, ax = fig
    else:  # construct
        logger.warning("Generating new fig, ax using plt.subplots")
        fig, ax = plt.subplots(num=fig_num, **subplot_kw)

    # Create meshgrid
    X, Y = numpy.meshgrid(xs, ys)
    Z = chi_squareds

    contour = ax.contour(X, Y, Z, **contour_kw)
    ax.clabel(contour, colors='k', fmt='%.2f', fontsize=12)

    # Make it pretty
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_plot:
        plt.show()

    return fig, ax
