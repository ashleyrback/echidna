""" KamLAND-Zen (plot-grab) Majoron limits script

This script:

  * Sets 90% confidence limit on the Majoron-emitting neutrinoless
    double beta decay modes (with spectral indices n = 1, 2, 3 and 7),
    using plot-grabbed data from KamLAND-Zen.

Examples:
  To use simply run the script and supply a YAML file detailing the
  spectra (data, fixed, floating) to load::

        $ python zero_nu_limit.py --from_file klz_majoron_limits_config.yaml

The ``upper_bound`` and ``lower_bound`` flags, can be used to return an
estimate on the error introduced through the plot-grabbing process.

.. note:: An example config would be::

        data_file:
            data/klz/v1.0.0/klz_data.hdf5

    ::

        fixed_dict: {
            data/klz/v1.0.0/total_b_g_klz.hdf5: 26647.1077395,
            }

    ::

        floating_list: [
            data/klz/v1.0.0/Xe136_2n2b_fig2.hdf5,
            ]

    ::

        signals_list: {
            klz_n1: data/klz/v1.0.0/Xe136_0n2b_n1_fig2.hdf5,
            klz_n2: data/klz/v1.0.0/Xe136_0n2b_n2_fig2.hdf5,
            klz_n3: data/klz/v1.0.0/Xe136_0n2b_n3_fig2.hdf5,
            klz_n7: data/klz/v1.0.0/Xe136_0n2b_n7_fig2.hdf5,
            }

    ::

        roi:
            energy:
                !!python/tuple [1.0, 3.0]

    ::

        per_bin:
            true

"""
import numpy

import echidna.utilities as utilities
logger = utilities.start_logging()  # To make sure we stat logging here first

import echidna.fit.test_statistic as test_statistic
from echidna.core.config import GlobalFitConfig
import echidna.output.store as store
import echidna.fit.fit as fit
from echidna.errors.custom_errors import CompatibilityError
import echidna.calc.decay as decay
import echidna.calc.constants as constants
import echidna.limit.limit as limit

import yaml
import json
import logging
from collections import OrderedDict
import argparse
import os


class ReadableDir(argparse.Action):
    """ Custom argparse action

    Adapted from http://stackoverflow.com/a/11415816

    Checks that hdf5 files supplied via command line exist and can be read
    """
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dirs = []
        if type(values) is str:
            prospective_dirs.append(values)
        elif type(values) is list:
            prospective_dirs = values
        else:
            raise TypeError("Invalid type for arg.")
        for prospective_dir in prospective_dirs:
            if not os.access(prospective_dir, os.R_OK):
                raise argparse.ArgumentTypeError(
                    "ReadableDir:{0} is not readable".format(prospective_dir))
        setattr(namespace, self.dest, values)  # keeps original format


def main(name, upper_bound=False, lower_bound=False,
         roi={"energy": (0.5, 4.8)}, per_bin=True, global_fit=None,
         data_file=None, fixed_dict={}, floating_list=[], sensitivity=False,
         signals_list=[], floating_backgrounds=[], signals=[], save_dir=None):
    """ The limit setting script.

    Args:
      name (string): Name of this KamLAND-Zen limit-setting session
      upper_bound (bool, optional): If True calculates the extreme
        upper-bound on the limit, due to uncertainty introduced during
        plot-grab digitisation
      lower_bound (bool, optional): If True calculates the extreme
        lower-bound on the limit, due to uncertainty introduced during
        plot-grab digitisation
      roi (dict, optional): Region Of Interest you want to fit in. The
        format of roi is e.g. {"energy": (2.4, 2.6), "radial3": (0., 0.2)}
      per_bin (bool, optional): Flag to monitor values of test
        statistic, per bin included in the fit (ROI)
      global_fit (string, optional): File from which to load
        :class:`GloabalFitConfig`
      data_file (string, optional): Path to hdf5 file containing Data
        spectrum you want to fit to
      fixed_dict (dict, optional): Dictionary containing paths to all fixed
        background spectra hdf5s, with path as key and priors (float)
        as values
      floating_list (list, optional): List of paths to hdf5s containing
        backgrounds to float
      sensitivity (bool, optional): Use expected background as data.
        Note a blank 'data' spectrum must still be supplied, which will
        then be filled with the appropriate expected background
        spectrum
      signals_list (list, optional): List of paths to hdf5s containing
        signal spectra
      floating_backgrounds (list, optional): List of background spectra
        to float
      signals (list, optional): List of signals to set limits for
      save_dir (string, optional): Directory, in which to save limit
        results

    .. warning:: floating backgrounds and signals specified via config
      will be **appended** to the lists passed as args, so if you are
      passing a spectrum to one of these arguments, make sure it is not
      also included in the config.

    """
    logger.info("Running analysis: %s" % name)

    # Set plot-grab error if required
    if upper_bound or lower_bound:
        pixel_err = 0.005
        n_pixel = 4
        plot_grab_err = numpy.sqrt(3 * (n_pixel*pixel_err)**2)

    # Set ROI
    if roi:
        logger.info("Set ROI")
        logging.getLogger("extra").info("\n%s\n" % json.dumps(roi))
        if not isinstance(roi, dict):
            logger.warning("Expecting roi be a dictionary, "
                           "not type %s" % type(roi))
    else:
        logger.warning("No ROI found, spectra will not be shrunk")

    # Set per_bin,
    logger.info("Storing per-bin information: %s" % per_bin)

    # Set test_statistic
    # This is fixed
    chi_squared = test_statistic.BakerCousinsChi(per_bin=True)
    logger.info("Using test statisitc: BakerCousinsChi")

    # Set fit_config
    if global_fit:
        global_fit_config = GlobalFitConfig.load_from_file(global_fit)
    else:  # Don't have any global fit parameters here - make blank config
        logger.warning("No global_fit path found - creating blank config")
        parameters = OrderedDict({})
        # The name set here will be the same name given to the GridSearch
        # created by the fitter, and the Summary class saved to hdf5.
        global_fit_config = GlobalFitConfig(name, parameters)
    logger.info("Using GlobalFitConfig with the following parameters:")
    logging.getLogger("extra").info(global_fit_config.get_pars())

    # Set data
    if data_file:
        logger.info("Using data spectrum %s" % data_file)
        data = store.load(data_file)
    else:
        logger.error("No data path found")
        logging.getLogger("extra").warning(
            " --> echidna can use total background as data, "
            "but a blank data spectrum should still be supplied.")
        raise ValueError("No data path found")

    # Apply plot-grab errors as appropriate
    if upper_bound:
        data_neg_errors = utilities.get_array_errors(
            data._data, lin_err=-plot_grab_err, log10=True)
        data._data = data._data + data_neg_errors
    if lower_bound:
        data_pos_errors = utilities.get_array_errors(
            data._data, lin_err=plot_grab_err, log10=True)
        data._data = data._data + data_pos_errors

    # Set fixed backgrounds
    # Create fixed_backgrounds dict with Spectra as keys and priors as values
    fixed_backgrounds = {}
    if fixed_dict:
        if not isinstance(fixed_dict, dict):
            raise TypeError(
                "Expecting dictionary with paths to fixed backgrounds as "
                "keys and num_decays for each background as values")
        for filename, num_decays in fixed_dict.iteritems():
            logger.info("Using fixed spectrum: %s (%.4f decays)" %
                        (filename, num_decays))
            spectrum = store.load(filename)

            # Add plot-grab errors as appropriate
            if upper_bound:
                spectrum_neg_errors = utilities.get_array_errors(
                    spectrum._data, lin_err=-plot_grab_err, log10=True)
                spectrum._data = spectrum._data + spectrum_neg_errors
            if lower_bound:
                spectrum_pos_errors = utilities.get_array_errors(
                    spectrum._data, lin_err=plot_grab_err, log10=True)
                spectrum._data = spectrum._data + spectrum_pos_errors

            fixed_backgrounds[spectrum] = num_decays
    else:
        logger.warning("No fixed spectra found")

    # Set floating backgrounds
    spectrum_names = [bkg.get_name() for bkg in floating_backgrounds]
    if floating_list:
        if not isinstance(floating_list, list):
            raise TypeError("Expecting list of paths to floating backgrounds")
        for filename in floating_list:
            spectrum = store.load(filename)
            if spectrum.get_name() not in spectrum_names:
                logger.info("Using floating background from: %s" % filename)
                floating_backgrounds.append(spectrum)
            else:  # Spectrum already loaded - passed via args
                logger.warning(
                    "Background %s already loaded. NOT using floating "
                    "background from: %s" % (spectrum.get_name(), filename))
    else:
        logger.warning("No floating backgrounds found")

    # Add plot-grab errors as appropriate
    for background in floating_backgrounds:
        if upper_bound:
            spectrum_neg_errors = utilities.get_array_errors(
                background._data, lin_err=-plot_grab_err, log10=True)
            background._data = background._data + spectrum_neg_errors
        if lower_bound:
            spectrum_pos_errors = utilities.get_array_errors(
                background._data, lin_err=plot_grab_err, log10=True)
            background._data = background._data + spectrum_pos_errors

    # Using default minimiser (GridSearch) so let Fit class handle this

    # Create fitter
    # No convolutions here --> use_pre_made = False
    fitter = fit.Fit(roi, chi_squared, global_fit_config, data=data,
                     fixed_backgrounds=fixed_backgrounds,
                     floating_backgrounds=floating_backgrounds,
                     per_bin=per_bin)
    logger.info("Created fitter")

    # Make data if running sensitivity study
    if sensitivity:
        data = fitter.get_data()  # Already added blank spectrum
        # Add fixed background
        data.add(fitter.get_fixed_background())
        # Add floating backgrounds - scaled to prior
        for background in fitter.get_floating_backgrounds():
            prior = background.get_fit_config().get_par("rate").get_prior()
            background.scale(prior)
            data.add(background)
        # Re-set data
        fitter.set_data(data)

    # Fit with no signal
    stat_zero = fitter.fit()
    summary = fitter.get_minimiser().get_summary()
    stat_zero = numpy.sum(stat_zero[0])
    logger.info("Calculated stat_zero (without penalty terms): %.4f" %
                stat_zero)
    logger.info("Fit summary:")
    logging.getLogger("extra").info("\n%s\n" % json.dumps(summary))
    for value in summary.values():
        stat_zero += value.get("penalty_term")
    logger.info("Calculated stat_zero (with penalty terms): %.4f" % stat_zero)

    # Load signals
    spectrum_names = [signal.get_name() for signal in signals]
    if signals_list:
        for filename in signals_list:
            signal = store.load(filename)
            if signal.get_name() not in spectrum_names:
                logger.info("Using signal spectrum from: %s" % filename)
                signals.append(signal)
            else:  # signal already loaded - passed via args
                logger.warning(
                    "Signal %s already loaded. NOT using signal "
                    "spectrum from: %s" % (signal.get_name(), filename))
    else:
        logger.error("No signal spectra found")
        raise CompatibilityError("Must have at least one signal to set limit")

    # Add plot-grab errors as appropriate
    # For signal we want to swap negative and positive fluctuations
    # The lower bound on the limit, is when all our backgrounds have
    # fluctuated down (through plot-grabbing) but the signal has
    # fluctuated up. Then the reverse is true for the upper bound,
    # backgrounds are fluctuated up and signal is fluctuated down
    for signal in signals:
        if upper_bound:
            signal_pos_errors = utilities.get_array_errors(
                signal._data, lin_err=plot_grab_err, log10=True)
            signal._data = signal._data + signal_pos_errors
        if lower_bound:
            signal_neg_errors = utilities.get_array_errors(
                signal._data, lin_err=-plot_grab_err, log10=True)
            signal._data = signal._data + signal_neg_errors

    # KamLAND-Zen limits
    klz_limits = {"Xe136_0n2b_n1": 2.6e24,
                  "Xe136_0n2b_n2": 1.0e24,
                  "Xe136_0n2b_n3": 4.5e23,
                  "Xe136_0n2b_n7": 1.1e22}

    # KamLAND-Zen detector info
    klz_detector = constants.klz_detector

    # Create converter
    converter = decay.DBIsotope(
        signal._name, klz_detector.get("Xe136_atm_weight"),
        klz_detector.get("XeEn_atm_weight"),
        klz_detector.get("Xe136_abundance"),
        decay.phase_spaces.get(signal._name),
        decay.matrix_elements.get(signal._name),
        loading=klz_detector.get("loading"),
        outer_radius=klz_detector.get("fv_radius"),
        scint_density=klz_detector.get("scint_density"))

    two_nu_rate = summary.get("Xe136_2n2b_rate").get("best_fit")
    half_life = converter.counts_to_half_life(
        two_nu_rate,
        n_atoms=converter.get_n_atoms(
            target_mass=klz_detector.get("target_mass")),
        livetime=klz_detector.get("livetime"))
    logger.info("Fitted 2nu rate is: %.4g" % half_life)

    # Loop through signals and set limit for each
    for signal in signals:
        # Reset GridSearch - with added signal rate parameter
        fitter.get_minimiser().reset_grids()

        # Create converter
        converter = decay.DBIsotope(
            signal._name, klz_detector.get("Xe136_atm_weight"),
            klz_detector.get("XeEn_atm_weight"),
            klz_detector.get("Xe136_abundance"),
            decay.phase_spaces.get(signal._name),
            decay.matrix_elements.get(signal._name),
            loading=klz_detector.get("loading"),
            outer_radius=klz_detector.get("fv_radius"),
            scint_density=klz_detector.get("scint_density"))
        klz_limit = klz_limits.get(signal._name)

        # Create limit setter
        limit_setter = limit.Limit(signal, fitter, save_dir=save_dir)

        limit_scaling = limit_setter.get_limit(
            store_fits=True, store_spectra=True)
        signal.scale(limit_scaling)
        half_life = converter.counts_to_half_life(
            limit_scaling,
            n_atoms=converter.get_n_atoms(
                target_mass=klz_detector.get("target_mass")),
            livetime=klz_detector.get("livetime"))

        logging.getLogger("extra").info(
            "\n########################################\n"
            "Signal: %s\n"
            "Calculated limit scaling of %.4g\n"
            " --> equivalent to %.4f events\n" %
            (signal._name, limit_scaling, signal.sum()))
        logging.getLogger("extra").info(
            "Calculated limit half life of %.4g y\n"
            " --> KamLAND-Zen equivalent limit: %.4g y\n"
            "########################################\n" %
            (half_life, klz_limit))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KamLAND-Zen (plot-grab) Majoron limits script")
    parser.add_argument("--from_file", action=ReadableDir,
                        help="Path to config file containing kw values")
    args = parser.parse_args()

    main_kw = yaml.load(open(args.from_file, "r"))
    logging.getLogger("extra").debug("\n\n%s\n" % yaml.dump(main_kw))

    try:
        main(**main_kw)
    except Exception:
        logger.exception("echidna failed with the following error:")
