import numpy

from echidna.utilities import start_logging
from echidna.settings import user_profile
from echidna.fit.fit_results import LimitResults
from echidna.fit.minimise import GridSearch
from echidna.errors.custom_errors import LimitError, CompatibilityError
from echidna.output import store

import logging
import yaml
import datetime
import json
import copy
from tqdm import tqdm


class Limit(object):
    """ Class to handle main limit setting.

    Args:
      signal (:class:`echidna.core.spectra.Spectra`): signal spectrum you wish
        to obtain a limit for.
      fitter (:class:`echidna.limit.fit.Fit`): The fitter used to set a
        a limit with.
      signal_config (:class:`echidna.core.config.Config`, optional): If None
        then the config from the signal arg is used.
      shrink (bool, optional): If set to True, :meth:`shrink` method is
        called on the signal spectrum before limit setting, shrinking to
        ROI.
      store_all (bool, optional): If set to True, all fit results for all
        signal scales are stored.
      save_dir (string, optional): Specify directory to save output, e.g.
        :class:`LimitResults`. Default is to use directory specified in
        :obj:`echidna.settings.user_profile`.

    Attributes:
      _logger (:class:`logging.Logger`): The output logger.
      _fitter (:class:`echidna.limit.fit.Fit`): The fitter used to set a
        a limit with.
      _signal (:class:`echidna.core.spectra.Spectra`): Signal spectrum you wish
        to obtain a limit for.
      _limit_results (:class:`echidna.fit.fit_results.LimitResults`): Limit
        results instance to report limit fit results
      _save_dir (string): Directory to save output, e.g. :class:`LimitResults`
    """
    def __init__(self, signal, fitter, signal_config=None,
                 shrink=True, store_all=False, save_dir=None):
        self._logger = start_logging(name="Limit")
        self._fitter = fitter
        self._fitter.check_fit_config(signal)
        self._fitter.set_signal(signal, shrink=shrink)
        self._signal = signal
        name = signal.get_name() + "_limit_fit_config"
        limit_config = signal.get_fit_config()
        fitter_config = fitter.get_fit_config()
        signal_config = signal.get_config()
        name = signal.get_name() + "_limit_results"
        self._limit_results = LimitResults(fitter_config, signal_config,
                                           limit_config, name)
        self._logger.info("Setting limit with the following parameters:")
        logging.getLogger("extra").info(
            yaml.dump(fitter_config.dump(basic=True)))
        if not save_dir:
            save_dir = user_profile.get("hdf5_save_path")
        self._save_dir = save_dir

    def get_array_limit(self, array, limit=2.71):
        """ Get the limit from an array containing statisics

        Args:
          array (:class:`numpy.array`): The array you want to set a limit for.
          limit (float, optional): The value of the test statisic which
            corresponds to the limit you want to set. The default is 2.71
            which corresponds to 90% CL when using a chi-squared test
            statistic.

        Raises:
          CompatibilityError: If the length of the array is not equal to the
            number of signal scalings.
          LimitError: If all values in the array are below limit.

        Returns:
          float: The signal scaling at the limit you are setting.
        """
        counts = self._signal.get_fit_config().get_par("rate").get_values()
        if len(counts) != len(array):
            raise CompatibilityError("Array length and number of signal "
                                     "scalings is different.")
        i = 0
        if not isinstance(array[0], float):  # is array
            array = self.sum_entries(array)
        for entry in array:
            if entry > limit:
                return counts[i]
            i += 1
        raise LimitError("Unable to find limit. Max stat: %s, Limit: %s"
                         % (array[-1], limit))

    def get_limit(self, limit=2.71, min_stat=None, store_limit=True,
                  store_fits=False, store_spectra=False, limit_fname=None):
        """ Get the limit using the signal spectrum.

        Args:
          limit (float, optional): The value of the test statisic which
            corresponds to the limit you want to set. The default is 2.71
            which corresponds to 90% CL when using a chi-squared test
            statistic.
          stat_zero (float or :class:`numpy.ndarray`, optional): Enables
            calculation of e.g. delta chi-squared. Include values of
            test statistic for zero signal contribution, so these can be
            subtracted from the values of the test statistic, with signal.
          store_limit (bool, optional):  If True (default) then a hdf5 file
            containing the :class:`echidna.fit.fit_results.LimitResults`
            object is saved.
          store_fits (bool, optional): If True then :class:`GridSearch`
            objects at each signal scale are stored in the
            :class:`echidna.fit.fit_results.LimitResults` object.
            Default is False.

              .. warning:: Only available for GridSearch

          store_spectra (bool, optional): If True then the spectra used for
            fitting are saved to hdf5. Default is False.
          limit_fname (string): Filename to save the
            `:class:`echidna.fit.fit_results.LimitResults` to.

        Raises:
          LimitError: If all values in the array are below limit.

        Returns:
          float: The signal scaling at the limit you are setting.
        """
        par = self._signal.get_fit_config().get_par("rate")

        min_stat_shape = None
        if min_stat:  # If supplied specific stat_zero use this
            self._logger.warning("Overriding min_stat with supplied value")
            logging.getLogger("extra").warning(" --> %s" % min_stat)
            if type(min_stat) is numpy.ndarray:
                min_stat_shape = min_stat.shape
        else:  # check zero signal stat in case its not in self._stats
            self._fitter.remove_signal()
            fit_stats = self._fitter.fit()
            if type(fit_stats) is tuple:
                fit_stats = fit_stats[0]
            if type(fit_stats) is numpy.ndarray:
                min_stat_shape = fit_stats.shape
            min_stat = numpy.sum(fit_stats)
            self._logger.info(
                "Calculated stat_zero (without penalty term): %.4f" % min_stat)
            fit_results = copy.copy(self._fitter.get_minimiser())
            if fit_results:
                summary = fit_results.get_summary()
                self._logger.info("Fit summary:")
                logging.getLogger("extra").info("\n%s\n" % json.dumps(summary))
                for value in summary.values():
                    min_stat += value.get("penalty_term")
            self._logger.info(
                "Calculated stat_zero (with penalty terms): %.4f" % min_stat)

        # Create stats array
        if min_stat_shape:
            shape = self._signal.get_fit_config().get_shape() + min_stat_shape
        else:
            shape = self._signal.get_fit_config().get_shape()
        stats = numpy.zeros(shape, dtype=numpy.float64)
        self._logger.debug("Creating stats array with shape %s" % str(shape))

        # Loop through signal scalings
        self._logger.debug("Testing signal scalings:\n\n")
        logging.getLogger("extra").debug(str(par.get_values()))

        # Progress bar
        for i, scale in tqdm(enumerate(par.get_values())):
            self._logger.debug("signal scale: %.4g" % scale)
            if not numpy.isclose(scale, 0.):
                if self._fitter.get_signal() is None:
                    self._fitter.set_signal(self._signal, shrink=False)
                self._fitter._signal.scale(scale)
            else:  # want no signal contribution
                self._fitter.remove_signal()
                self._logger.warning(
                    "Removing signal in fit for scale %.4g" % scale)

            fit_stats = self._fitter.fit()
            if type(fit_stats) is tuple:
                fit_stats = fit_stats[0]
            stats[i] = fit_stats

            fit_results = self._fitter.get_minimiser()
            if fit_results:
                results_summary = fit_results.get_summary()
                for par_name, value in results_summary.iteritems():
                    self._limit_results.set_best_fit(i, value.get("best_fit"),
                                                     par_name)
                    self._limit_results.set_best_fit_err(
                        i, value.get("best_fit_err"), par_name)
                    self._limit_results.set_penalty_term(
                        i, value.get("penalty_term"), par_name)
                if store_fits:
                    # Make new GridSearch to store as deepcopy doesn't work
                    fit_config = copy.copy(fit_results.get_fit_config())
                    spectra_config = copy.copy(
                        fit_results.get_spectra_config())
                    try:
                        grid_search = GridSearch(
                            fit_config, spectra_config, per_bin=True)

                        # Set stats
                        grid_search.set_stats(copy.deepcopy(
                            fit_results._stats))

                        # Set penalty terms
                        grid_search.set_penalty_terms(copy.deepcopy(
                            fit_results._penalty_terms))

                        # Save
                        self._limit_results.set_fit_result(i, grid_search)
                    except AttributeError:
                        self._logger.warning(
                            "Cannot store fit_results of type %s" %
                            type(fit_results))
                    except Exception:
                        raise

        # Set stats in limit_results
        self._limit_results._stats = stats
        stats = self._limit_results.get_full_stats()  # Adds penalty terms

        # Convert stats to delta - subtracting minimum
        if numpy.nanmin(stats) < numpy.sum(min_stat):
            # for sensitivity these should be the same, but when fitting to
            # data these may differ
            prev_min_stat = numpy.sum(min_stat)
            min_stat = numpy.nanmin(stats)
            self._logger.warning("Updating min_stat based on numpy.nanmin")
            logging.getLogger("extra").warning(
                " --> was %.4f now %.4f" % (prev_min_stat, min_stat))
        stats -= numpy.sum(min_stat)

        # Also want to know index of minimum
        min_bin = numpy.argmin(stats)

        log_text = ""
        log_text += "\n===== Limit Summary ====="
        try:
            # Slice from min_bin upwards
            limit_index = numpy.where(stats[min_bin:] > limit)[0][0] + min_bin
            self._limit_results.set_limit_index(limit_index)
            limit = par.get_values()[limit_index]
            self._limit_results.set_limit(limit)

            # Save results
            self._save_results(store_limit=store_limit,
                               store_spectra=store_spectra,
                               limit_fname=limit_fname)

            # Write logging output
            log_text += "\nLimit found at:\n"
            log_text += "Signal Decays: %.4g\n" % limit
            for parameter in self._fitter.get_fit_config().get_pars():
                cur_par = self._fitter.get_fit_config().get_par(parameter)
                log_text += "--- systematic: %s ---\n" % parameter
                log_text += ("Best fit: %4g\n" %
                             self._limit_results.get_best_fit(limit_index,
                                                              parameter))
                if cur_par.get_prior():
                    log_text += ("Prior: %.4g\n" %
                                 cur_par.get_prior())
                if cur_par.get_sigma():
                    log_text += ("Sigma: %.4g\n" %
                                 cur_par.get_sigma())
                log_text += ("Penalty term: %.4g\n" %
                             self._limit_results.get_penalty_term(limit_index,
                                                                  parameter))
            log_text += "----------------------------\n"
            log_text += ("Test statistic: %.4f\n" %
                         numpy.sum(stats[limit_index]))
            log_text += "N.D.F.: 1\n"  # Only fit one dof currently
            logging.getLogger("extra").info("\n%s\n" % log_text)
            return limit
        except IndexError:
            # Slice from min_bin upwards
            limit_index = numpy.argmax(stats[min_bin:]) + min_bin
            self._limit_results.set_limit_index(limit_index)
            limit = par.get_values()[limit_index]
            self._limit_results.set_limit(limit)

            # Save results
            self._save_results(store_limit=store_limit,
                               store_spectra=store_spectra,
                               limit_fname=limit_fname)

            # Write logging output
            log_text += "\nNo limit found:\n"
            log_text += "Signal Decays (at max stat): %.4g\n" % limit
            for parameter in self._fitter.get_fit_config().get_pars():
                cur_par = self._fitter.get_fit_config().get_par(parameter)
                log_text += "--- systematic: %s ---\n" % parameter
                log_text += ("Best fit: %4g\n" %
                             self._limit_results.get_best_fit(limit_index,
                                                              parameter))
                if cur_par.get_prior():
                    log_text += ("Prior: %.4g\n" %
                                 cur_par.get_prior())
                if cur_par.get_sigma():
                    log_text += ("Sigma: %.4g\n" %
                                 cur_par.get_sigma())
                log_text += ("Penalty term: %.4g\n" %
                             self._limit_results.get_penalty_term(limit_index,
                                                                  parameter))
            log_text += "----------------------------\n"
            log_text += ("Test statistic: %.4f\n" %
                         numpy.sum(stats[limit_index]))
            log_text += "N.D.F.: 1\n"  # Only fit one dof currently
            logging.getLogger("extra").info("\n%s\n" % log_text)
            raise LimitError("Unable to find limit. Max stat: %s, Limit: %s"
                             % (stats.max(), limit))

    def _save_results(self, store_limit=True,
                      store_spectra=False, limit_fname=None):
        """ Internal method to save limit results

        Args:
          store_limit (bool, optional):  If True (default) then a hdf5 file
            containing the :class:`echidna.fit.fit_results.LimitResults`
            object is saved.
          store_spectra (bool, optional): If True then the spectra used for
            fitting are saved to hdf5. Default is False.
          limit_fname (string): Filename to save the
            `:class:`echidna.fit.fit_results.LimitResults` to.

        """
        if store_limit:
            if limit_fname:
                if limit_fname[-5:] != '.hdf5':
                    limit_fname += '.hdf5'
            else:
                timestamp = datetime.datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S")
                fname = (self._limit_results.get_name() +
                         "_" + timestamp + ".hdf5")
                limit_fname = self._save_dir + fname
            store.dump_limit_results(limit_fname, self._limit_results)
            self._logger.info("Saved summary of %s to file %s" %
                              (self._limit_results.get_name(),
                               limit_fname))
        if store_spectra:
            fname = self._fitter.get_data()._name + "_data.hdf5"
            store.dump(self._save_dir + fname, self._fitter.get_data())
            if self._fitter.get_fixed_background():
                fname = (self._fitter.get_fixed_background()._name +
                         "_fixed.hdf5")
                store.dump(self._save_dir + fname,
                           self._fitter.get_fixed_background())
            if self._fitter.get_floating_backgrounds():
                for background in self._fitter.get_floating_backgrounds():
                    fname = background._name + "_float.hdf5"
                    store.dump(self._save_dir + fname, background)
            fname = self._signal._name + "_signal.hdf5"
            store.dump(self._save_dir + fname, self._signal)

    def get_statistics(self):
        """ Get the test statistics for all signal scalings.

        Returns:
          :class:`numpy.array`: Of test statisics for all signal scalings.
        """
        signal_config = self._signal.get_fit_config()
        stats = []
        for scale in signal_config.get_par("rate").get_values():
            if not numpy.isclose(scale, 0.):
                self._signal.scale(scale)
                self._fitter.set_signal(self._signal, shrink=False)
            else:
                self._fitter.remove_signal()
            stats.append(self._fitter.fit())
        return numpy.array(stats)

    def sum_entries(self, array):
        """ Sums entries of an array which contains arrays as entries.

        Args:
          array (:class:`numpy.array`): The array you want to sum the
            elements of.

        Returns:
          :class:`numpy.array`: The input array with its entries summed.
        """
        new_array = []
        for entry in array:
            new_array.append(entry.sum())
        return numpy.array(new_array)
