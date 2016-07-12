import echidna.output.store as store
from echidna.fit.minimise import GridSearch, Minuit
from echidna.errors.custom_errors import CompatibilityError, ParameterError
from echidna.core.config import GlobalFitConfig

import numpy
import logging
import collections
import os
import yaml
import copy

ncount = 0


class Fit(object):
    """ Class to handle fitting.

    .. warning:: The :class:`Fit` initialisation will try to set
      atrributes using the values provided. If a value is not provided
      echidna will attempt to set the default. If this is not possible
      a warning will be raised and you will have to set this attribute
      manually before calling :meth:`fit`.

    Args:
      roi (dictionary): Region Of Interest you want to fit in. The format of
        roi is e.g. {"energy": (2.4, 2.6), "radial3": (0., 0.2)}
      test_statistic (:class:`echidna.limit.test_statistic.TestStatistic`): An
        appropriate class for calculating test statistics.
      fit_config (:class:`FitConfig`, optional): Config class for fit -
        usually loaded from file.
      data (:class:`echidna.core.spectra.Spectra`): Data spectrum you want to
        fit.
      fixed_background (dict, optional): Dictionary containing all fixed
        backgrounds, with :class:`echidna.core.spectra.Spectra` as keys
        and priors (float) as values.
      floating_backgrounds (list, optional): one
        :class:`echidna.core.spectra.Spectra` for each background to float.
      signal (:class:`echidna.core.spectra.Spectra`):
        A spectrum of the signal that you are fitting.
      shrink (bool, optional): If set to True (default),
        :meth:`shrink` method is called on all spectra shrinking them to
        the ROI.
      minimiser (:class:`echidna.limit.minimiser.Minimiser`, optional): Object
        to handle the minimisation.
      pre_made_base_dir (string, optional): Directory in which pre-made
        convolved spectra are stored.
      single_bin (bool, optional): Flag for a single bin fit (e.g. simple
        counting experiment).
      per_bin (bool, optional): Flag to monitor values of test
        statistic, per bin included in the fit (ROI)

    Attributes:
      _logger (loggging.Logger): Logger for :class:`Fit` class.
      _checked
      _roi (dictionary): Region Of Interest you want to fit in. The format of
        roi is e.g. {"energy": (2.4, 2.6), "radial3": (0., 0.2)}
      _test_statistic (:class:`echidna.limit.test_statistic.TestStatistic`): An
        appropriate class for calculating test statistics.
      _fit_config (:class:`FitConfig`): Config class for fit -
        usually loaded from file.
      _data (:class:`echidna.core.spectra.Spectra`): Data spectrum you want to
        fit.
      _fixed_background (:class:`echidna.core.spectra.Spectra`):
        A spectrum containing all fixed backgrounds.
      _floating_backgrounds (list): one :class:`echidna.core.spectra.Spectra`
        for each background to float.
      _signal (:class:`echidna.core.spectra.Spectra`):
        A spectrum of the signal that you are fitting.
      _minimiser (:class:`echidna.limit.minimiser.Minimiser)`: Object to
        handle the minimisation.
      _checked (bool): If True then the fit class is ready to be used.
      _pre_made_dir (string): Directory in which pre-made convolved
        spectra are stored.
    """
    def __init__(self, roi, test_statistic, fit_config=None, data=None,
                 fixed_backgrounds=None, floating_backgrounds=None,
                 signal=None, shrink=True, per_bin=False, minimiser=None,
                 pre_made_base_dir=None, single_bin=False):
        self._logger = logging.getLogger("Fit")
        self._checked = False
        self.set_roi(roi)

        self.set_test_statistic(test_statistic)

        if not fit_config:
            parameters = collections.OrderedDict({})
            fit_config = GlobalFitConfig("global_fit_config", parameters)
        self.set_fit_config(fit_config)

        if data:
            self.set_data(data)
        else:  # set both as None for now
            self._data = None
            self._data_pars = None
            self._logger.warning("Data has not been set. This must be set "
                                 "manually before running the fit.")

        if fixed_backgrounds:  # spectra_dict in expected form
            self.make_fixed_background(fixed_backgrounds)  # sets attribute
        else:  # set both as None for now
            self._fixed_background = None
            self._fixed_pars = None
            self._logger.warning("Fixed background has not been set. Either "
                                 "fixed background or at least one floating "
                                 "is required to run the fit.")

        if floating_backgrounds:
            self.set_floating_backgrounds(floating_backgrounds)
        else:  # Set both as None for now
            self._floating_backgrounds = None
            self._floating_pars = None
            self._logger.warning("No floating background has been set. Either "
                                 "fixed background or at least one floating "
                                 "is required to run the fit.")

        self._global_dict = {}

        # Now all floating backgrounds are loaded, check fit par values
        for par in self.get_fit_config().get_spectra_pars():
            par.check_values()  # raises an error if prior is not in values

        if signal:
            self.set_signal(signal)
        else:  # Set both as None for now
            self._signal = None
            self._signal_pars = None
            self._logger.warning("No signal has been set.")

        if shrink:
            self.shrink_all()

        self._per_bin = per_bin

        # Try to set minimiser
        # Will only work if check_all_spectra passes.
        try:
            self.set_minimiser(minimiser)
        except CompatibilityError as detail:
            self._minimiser = None
            self._logger.warning("Minimiser could not be set because: %s" %
                                 detail)
        except IndexError as detail:
            self._minimiser = None
            self._logger.warning("Minimiser could not be set because: %s" %
                                 detail)
        self._pre_made_base_dir = pre_made_base_dir
        self._single_bin = single_bin

    def append_fixed_background(self, spectra_dict, shrink=True):
        ''' Appends the fixed background with more spectra.

        Args:
          spectra_dict (dict): Dictionary containing spectra as keys and
            prior counts as values.
          shrink (bool, optional): If set to True (default), :meth:`shrink`
            method is called on the spectra shrinking it to the ROI.
        '''
        for spectrum, scaling in spectra_dict.iteritems():
            # Copy so original spectra is unchanged
            spectrum = copy.copy(spectrum)
            if shrink:
                self.shrink_spectra(spectrum)
            spectrum.scale(scaling)
            self._fixed_background.add(spectrum)

    def check_all_spectra(self):
        """ Ensures that all spectra can be used for fitting.

        Raises:
          CompatibilityError: If the data spectra exists and its roi pars have
            not been set.
          CompatibilityError: If the data spectrum has not been set.
          CompatibilityError: If neither fixed background nor at least
            one floating background, has been set.
          CompatibilityError: If the fixed background spectra exists and its
            roi pars have not been set.
          CompatibilityError: If the signal spectra exists and its
            roi pars have not been set.
          CompatibilityError: If the floating backgrounds spectra exists and
            their roi pars have not been set.
          CompatibilityError: If the floating backgrounds spectra exists and
            their roi pars have not been set.
          CompatibilityError: If the floating backgrounds spectra exists and
            length of their roi pars is different to the number of floating
            backgrounds.
        """
        if self._data:
            if not self._data_pars:
                raise CompatibilityError("data roi pars have not been set.")
            self.check_spectra(self._data)
        else:
            raise CompatibilityError("data spectrum has not been set.")

        if (self._fixed_background is None and
                self._floating_backgrounds is None):
            raise CompatibilityError("Must provide either fixed background "
                                     "or at least one floating background "
                                     "to fit to data")
        if self._fixed_background:
            if not self._fixed_pars:
                raise CompatibilityError("fixed background roi pars have not "
                                         "been set.")
            self.check_spectra(self._fixed_background)

        if self._signal:
            if not self._signal_pars:
                raise CompatibilityError("signal roi pars have not been set.")

        if self._floating_backgrounds:
            if not self._floating_pars:
                raise CompatibilityError("floating background roi pars have "
                                         "not been set.")
            if len(self._floating_pars) != len(self._floating_backgrounds):
                raise CompatibilityError("Different number of sets of roi "
                                         "pars as the number of floating "
                                         "backgrounds.")
            for background in self._floating_backgrounds:
                if background.get_fit_config():
                    self.check_fit_config(background)

    def check_fit_config(self, spectra):
        """ Checks that a spectra has a fit config.

        Args:
          spectra (:class:`echidna.core.spectra.Spectra`): Spectra you want to
            check.

        Raises:
          CompatibilityError: If spectra has no fit config
        """
        if not spectra.get_fit_config():
            raise CompatibilityError("%s has no fit config" % spectra._name)
        for par in spectra.get_fit_config().get_pars():
            parameter = spectra.get_fit_config().get_par(par)
            parameter.check_values()

    def check_fitter(self):
        """ Checks that the Fit class is ready to be used for fitting.

        Raises:
          IndexError: If fit config contains no parameters
          AttributeError: If :attr:`_minimiser` has not been set.
          ValueError: If (un)expected per_bin flag in minimiser.
          ValueError: If (un)expected integer value for num_bins, in
            fit_results.
          ValueError: If (un)expected per_bin flag in test_statistic.
        """
        self.check_all_spectra()

        # Check fit parameters
        if self._floating_backgrounds:
            if len(self.get_fit_config().get_pars()) == 0:
                raise IndexError("No parameters found in fit config.")
        else:
            if len(self.get_fit_config().get_spectra_pars()) != 0:
                self._logger.error(
                    "Spectral fit parameters: %s" %
                    str(self.get_fit_config().get_spectra_pars()))
                raise ValueError("Expected 0 spectral fit pars for "
                                 "no floating backgrounds")

        # Check minimiser and fit results
        if self._minimiser is None:
            raise AttributeError("Minimiser has not been set.")

        # Check per_bin propagation
        if self._per_bin:
            if not self._minimiser._per_bin:
                raise ValueError("Expected per_bin True flag in minimiser")
            if not self._test_statistic._per_bin:
                raise ValueError("Expected per_bin True flag in "
                                 "test_statistic")

        self._checked = True

        self._logger.info("Fitter checked!")
        self._logger.info("Running fit with the following parameters:")
        logging.getLogger("extra").info(
            yaml.dump(self.get_fit_config().dump(basic=True)))
        for par in self.get_fit_config().get_pars():
            parameter = self.get_fit_config().get_par(par)
            self._logger.debug("Parameter %s, with values:\n\n" % par)
            logging.getLogger("extra").debug(str(parameter.get_values()))

    def check_roi(self, roi):
        """ Checks the ROI used to fit.

        Args:
          roi (dict): roi you want to check.

        Raises:
          TypeError: If roi is not a dict
          TypeError: If value in roi dict is not a list or a tuple
          CompatibilityError: If the length of a value in the roi dict is
            not 2.
        """
        if not isinstance(roi, dict):
            raise TypeError("roi must be a dictionary of parameter values")
        for dim in roi:
            if not isinstance(roi[dim], (tuple, list)):
                raise TypeError("roi must be a dictionary of tuples or lists")
            if len(roi[dim]) != 2:
                raise CompatibilityError("%s entry (%s) in roi must contain a"
                                         " low and high value in a tuple or"
                                         " list" % (dim, self._roi[dim]))
            if roi[dim][0] > roi[dim][1]:  # Make sure low is first
                roi[dim] = roi[dim][::-1]  # Reverses list/tuple

    def check_spectra(self, spectra):
        """ Checks the spectra you want to fit.

        Args:
          spectra (:class:`echidna.core.spectra.Spectra`): Spectra you want to
            check.

        Raises:
          ValueError: If roi low value and spectra low value are not equal.
          ValueError: If roi high value and spectra high value are not equal.
        """
        for dim in self._roi:
            dim_type = spectra.get_config().get_dim_type(dim)
            par = dim + "_" + dim_type
            if not numpy.isclose(self._roi[dim][0],
                                 spectra.get_config().get_par(par)._low):
                raise ValueError("roi %s low (%s) not equal to spectra %s"
                                 " low (%s)"
                                 % (dim, self._roi[dim][0], dim,
                                    spectra.get_config().get_par(par)._low))
            if not numpy.isclose(self._roi[dim][1],
                                 spectra.get_config().get_par(par)._high):
                raise ValueError("roi %s high (%s) not equal to spectra %s"
                                 " high (%s)"
                                 % (dim, self._roi[dim][0], dim,
                                    spectra.get_config().get_par(par)._high))

    def get_data(self):
        """ Gets the data you are fitting.

        Returns:
          :class:`echidna.core.spectra.Spectra`: The data you are fitting.
        """
        return self._data

    def get_fit_config(self):
        """
        """
        return self._fit_config

    def get_fixed_background(self):
        """ Gets the fixed background you are fitting.

        Returns:
          :class:`echidna.core.spectra.Spectra`: The fixed background you are
            fitting.
        """
        return self._fixed_background

    def get_floating_backgrounds(self):
        """ Gets the floating backgrounds you are fitting.

        Returns:
          list: The floating backgrounds you are fitting.
        """
        return self._floating_backgrounds

    def get_minimiser(self):
        """ Gets the minimiser you are using.

        Returns:
          :class:`echidna.limit.minimise.Minimiser`: The minimiser
        """
        return self._minimiser

    def get_test_statistic(self):
        """ Gets the class instance you are using to calculate the test
        statistic used in the fit.

        Returns:
          (:class:`echidna.limit.test_statistic.TestStatistic`): The class
            instance used to calculate test statistics.
        """
        return self._test_statistic

    def get_roi(self):
        """ Gets the region of interest (roi)

        Returns:
          dict: The region of interest
        """
        return self._roi

    def get_roi_pars(self, spectra):
        """ Get the parameters of a spectra that contain the roi.

        Args:
          :class:`echidna.core.spectra.Spectra`: The spectra you want to obtain
            the roi parameters for.

        Returns:
          list: Of the names of the spectra parameters which contain the roi.
        """
        pars = []
        for dim in self._roi:
            dim_type = spectra.get_config().get_dim_type(dim)
            par = dim + "_" + dim_type
            pars.append(par)
        return pars

    def get_signal(self):
        """ Gets the signal you are fitting.

        Returns:
          :class:`echidna.core.spectra.Spectra`: The fixed background you are
            fitting.
        """
        return self._signal

    def fit(self):
        """ Gets the value of the test statistic used for fitting.

        Returns:
          float: The resulting test statisic.
        """
        if not self._checked:
            self.check_fitter()

        if not self._floating_backgrounds:  # Use fixed only
            observed = self._data.nd_project(self._data_pars)
            expected = self._fixed_background.nd_project(self._fixed_pars)
            if self._signal:
                expected += self._signal.nd_project(self._signal_pars)
            if self._single_bin:
                expected = numpy.sum(expected)
                observed = numpy.sum(observed)
            return self._test_statistic.compute_statistic(observed.ravel(),
                                                          expected.ravel())
        else:  # Pass to minimiser
            if self._minimiser is None:
                raise AttributeError("Minimiser is not set.")
            if type(self._minimiser) is GridSearch:
                return self._minimiser.minimise(self._funct_grid,
                                                self._test_statistic)
            elif type(self._minimiser) is Minuit:
                return self._minimiser.minimise(self._funct_root,
                                                self._test_statistic)

    def _funct_grid(self, *args):
        """ Callable to pass to minimiser.

        Args:
          args (list): List of fit parameter values to test in the
            current iteration.

        Returns:
          tuple: containing:

            :class:`numpy.ndrray`: Values of the test statistic given
              the current values of the fit parameters.
            float: Total penalty term to be applied to the test
              statistic.

        Raises:
          ValueError: If :attr:`_floating_backgrounds` is None. This
            method requires at least one floating background.
          ParameterError: If it is attempted to load a pre-made spectrum
            after other fit parameters have been applied to the spectrum.

        .. note:: This method should not be called directly, it is
          intended to be passed to an appropriate minimiser and
          called from within the minimisation algorithm.

        .. note:: This method should not be used if there are no
          floating backgrounds.

        """
        if self._floating_backgrounds is None:
            raise ValueError("The _funct method can only be used " +
                             "with at least one floating background")

        # Update parameter current values
        for index, value in enumerate(args):
            par = self._fit_config.get_par_by_index(index)
            par.set_current_value(value)

        # Loop over all floating backgrounds
        observed = self._data.nd_project(self._data_pars)
        if self._fixed_background:
            expected = self._fixed_background.nd_project(self._fixed_pars)
        else:
            expected = None
        global_pars = self._fit_config.get_global_pars()
        cur_val = ""
        for parameter in global_pars:
            cur_val += parameter._name + str(parameter._current_value) + "_"
        for spectrum, floating_pars in zip(self._floating_backgrounds,
                                           self._floating_pars):
            # Apply global parameters first
            if global_pars:
                background_name = spectrum.get_background_name()
                if background_name not in self._global_dict:
                    self._global_dict[background_name] = {}
                if cur_val not in self._global_dict[background_name]:
                    fit_config = spectrum._fit_config
                    applied = False
                    load = False
                    load_pars = []
                    for parameter in global_pars:
                        if parameter._pre_made:  # Load pre-made spectrum
                            if applied:
                                raise ParameterError(
                                    "Trying to load pre-made after a "
                                    "parameter has been applied. Rearrange "
                                    "parameter order in config.")
                            load_pars.append(parameter)
                            load = True
                        else:
                            applied = True
                            if load:
                                spectrum = self.load_pre_made(spectrum,
                                                              load_pars)
                                load = False
                            spectrum = parameter.apply_to(spectrum)
                    if load:
                        spectrum = self.load_pre_made(spectrum,
                                                      load_pars)
                    spectrum._fit_config = fit_config
                    # Shrink to roi
                    self._data.shrink_to_self(spectrum)
                    # rebin
                    spectrum.rebin(self._data._data.shape)
                    self._global_dict[background_name][cur_val] = spectrum
                else:
                    spectrum = self._global_dict[background_name][cur_val]

            # Apply spectrum-specific parameters
            if spectrum._fit_config:
                for par_name in spectrum._fit_config.get_pars():
                    parameter = spectrum._fit_config.get_par(par_name)
                    spectrum = parameter.apply_to(spectrum)

            # Spectrum should now be fully convolved/scaled
            if expected is not None:
                expected += spectrum.nd_project(floating_pars)
            else:
                expected = spectrum.nd_project(floating_pars)

        # Add signal, if required
        if self._signal:
            if global_pars:
                num_decays = self._signal._num_decays
                background_name = self._signal.get_background_name()
                if background_name not in self._global_dict:
                    self._global_dict[background_name] = {}
                if cur_val not in self._global_dict[background_name]:
                    fit_config = self._signal._fit_config
                    applied = False
                    load = False
                    load_pars = []
                    signal = None
                    for parameter in global_pars:
                        if parameter._pre_made:  # Load pre-made spectrum
                            if applied:
                                raise ParameterError(
                                    "Trying to load pre-made after a "
                                    "parameter has been applied. Rearrange "
                                    "parameter order in config.")
                            load_pars.append(parameter)
                            load = True
                        else:
                            applied = True
                            if load:
                                signal = self.load_pre_made(self._signal,
                                                            load_pars)
                                load = False
                            if signal:
                                signal = parameter.apply_to(signal)
                            else:
                                signal = parameter.apply_to(self._signal)
                    if load:
                        if applied:
                            raise ParameterError(
                                "Trying to load pre-made after a parameter "
                                "has been applied. Rearrange parameter order "
                                "in config.")
                        signal = self.load_pre_made(self._signal,
                                                    load_pars)
                    signal._fit_config = fit_config
                    self._data.shrink_to_self(signal)
                    signal.rebin(self._data._data.shape)
                    self._global_dict[background_name][cur_val] = signal
                else:
                    signal = self._global_dict[background_name][cur_val]
                signal.scale(num_decays)
            else:
                signal = self._signal
            expected += signal.nd_project(self._signal_pars)

        # If single bin - sum over expected and observed
        if self._single_bin:
            expected = numpy.sum(expected)
            observed = numpy.sum(observed)

        # Calculate value of test statistic
        test_statistic = self._test_statistic.compute_statistic(
            observed.ravel(), expected.ravel())

        # Add penalty terms
        total_penalty = 0.
        for parameter in self._fit_config.get_pars():
            par = self._fit_config.get_par(parameter)
            current_value = par.get_current_value()
            prior = par.get_prior()
            sigma = par.get_sigma()
            # If sigma is explicitly None add no penalty term
            if (sigma is not None):
                total_penalty += self._test_statistic.get_penalty_term(
                    current_value, prior, sigma)

        # Check for per_bin flag
        if self._per_bin:
            test_statistic = test_statistic.reshape(spectrum._data.shape)

        return test_statistic, total_penalty

    def _funct_root(self, npar, gin, chisq, args, iflag):
        """ Callable to pass to minimiser.

        Args:
          npar (int): Number of parameters you are minimising.
          gin (list): Gradient of paramters you are minimising.
          chisq (float): The test statistic you are minimising.
          args (list): Contains the parameters you are minimising.
          iflag (int): Denotes various ROOT error states.

        Returns:
          float: The test statistic you are minimising.

        Raises:
          ValueError: If :attr:`_floating_backgrounds` is None. This
            method requires at least one floating background.
          ParameterError: If it is attempted to load a pre-made spectrum
            after other fit parameters have been applied to the spectrum.

        .. note:: This method should not be called directly, it is
          intended to be passed to an appropriate minimiser and
          called from within the minimisation algorithm.

        .. note:: This method should not be used if there are no
          floating backgrounds.

        """
        global ncount
        if self._floating_backgrounds is None:
            raise ValueError("The _funct method can only be used " +
                             "with at least one floating background")

        # Update parameter current values
        for i, par in enumerate(self._fit_config.get_pars()):
            par = self._fit_config.get_par(par)
            print par._name, i, args[i]
            par.set_current_value(args[i])

        # Loop over all floating backgrounds
        observed = self._data.nd_project(self._data_pars)
        if self._fixed_background:
            expected = self._fixed_background.nd_project(self._fixed_pars)
        else:
            expected = None
        global_pars = self._fit_config.get_global_pars()
        cur_val = ""
        for parameter in global_pars:
            cur_val += parameter._name + str(parameter._current_value) + "_"
        spec = None
        for spectrum, floating_pars in zip(self._floating_backgrounds,
                                           self._floating_pars):
            print "Applying systs to", spectrum._name
            spec = None
            # Apply global parameters first
            if global_pars:
                fit_config = spectrum._fit_config
                bkgnd_name = spectrum.get_background_name()
                i = 0
                for parameter in global_pars:
                    if parameter._pre_made:
                        continue
                    if 'resolution' in parameter._name:
                        if i != 0:
                            raise ParameterError("Resolution must be first "
                                                 "global systematic.")
                        spectra = []
                        # Working for ly not % resolution:
                        int_val = int(parameter._current_value)
                        num_decays = spectrum._num_decays
                        if float(int_val) == parameter._current_value:
                            if bkgnd_name not in self._global_dict:
                                self._global_dict[bkgnd_name] = {}
                            if str(int_val) not in\
                                    self._global_dict[bkgnd_name]:
                                spec = self.load_pre_made(spectrum,
                                                          [parameter],
                                                          cur_val=int_val)
                                spectrum.shrink_to_self(spec)
                                spec.rebin(spectrum._data.shape)
                                spec.scale(num_decays)
                                self._global_dict[bkgnd_name][str(int_val)] =\
                                    spec
                            else:
                                spec =\
                                    self._global_dict[bkgnd_name][str(int_val)]
                        else:
                            for ly_idx in range(4):
                                str_int = str(int_val + ly_idx - 2)
                                if bkgnd_name not in self._global_dict:
                                    self._global_dict[bkgnd_name] = {}
                                if str_int not in \
                                        self._global_dict[bkgnd_name]:
                                    spec = self.load_pre_made(
                                        spectrum, [parameter],
                                        cur_val=str_int)
                                    if spec:
                                        spec.scale(num_decays)
                                        spectrum.shrink_to_self(spec)
                                        spec.rebin(spectrum._data.shape)
                                        spectra.append(spec)
                                        self._global_dict[bkgnd_name][str_int]\
                                            = spec
                                else:
                                    spectra.append(
                                        self._global_dict[bkgnd_name][str_int])
                            spec = parameter.apply_to(
                                spectra, parameter._name.split('_')[-1])
                    else:
                        if spec:
                            spec = parameter.apply_to(spec)
                        else:
                            spec = paramter.apply_to(spectrum)
                    i += 1
                spec._fit_config = fit_config
                # Shrink to roi
                print "shrinking to roi"
                self._data.shrink_to_self(spec)
                # rebin
                spec.rebin(self._data._data.shape)
                spec.scale(num_decays)
            # Apply spectrum-specific parameters
            if not spec:
                spec = spectrum
            if spec._fit_config:
                for par_name in spec._fit_config.get_pars():
                    parameter = spec._fit_config.get_par(par_name)
                    spec = parameter.apply_to(spec)

            # Spectrum should now be fully convolved/scaled
            if expected is not None:
                expected += spec.nd_project(floating_pars)
            else:
                expected = spec.nd_project(floating_pars)
        # Add signal, if required
        if self._signal:
            if global_pars:
                num_decays = self._signal._num_decays
                bkgnd_name = self._signal.get_background_name()
                fit_config = self._signal._fit_config
                signal = None
                spec = None
                i = 0
                print "Applying systs to signal"
                for parameter in global_pars:
                    if parameter._pre_made:
                        continue
                    if 'resolution' in parameter._name:
                        if i != 0:
                            raise ParameterError("Resolution must be first "
                                                 "global systematic.")
                        spectra = []
                        # Working for ly not % resolution:
                        int_val = int(parameter._current_value)
                        num_decays = self._signal._num_decays
                        if float(int_val) == parameter._current_value:
                            if bkgnd_name not in self._global_dict:
                                self._global_dict[bkgnd_name] = {}
                            if str(int_val) not in\
                                    self._global_dict[bkgnd_name]:
                                signal = self.load_pre_made(self._signal,
                                                            [parameter],
                                                            cur_val=int_val)
                                self._floating_backgrounds[0].\
                                    shrink_to_self(signal)
                                signal.rebin(
                                    self._floating_backgrounds[0]._data.shape)
                                signal.scale(num_decays)
                                self._global_dict[bkgnd_name][str(int_val)] =\
                                    signal
                            else:
                                signal =\
                                    self._global_dict[bkgnd_name][str(int_val)]
                        else:
                            for ly_idx in range(4):
                                str_int = str(int_val + ly_idx - 2)
                                if bkgnd_name not in self._global_dict:
                                    self._global_dict[bkgnd_name] = {}
                                if str_int not in \
                                        self._global_dict[bkgnd_name]:
                                    spec = self.load_pre_made(
                                        self._signal, [parameter],
                                        cur_val=str_int)
                                    if spec:
                                        spec.scale(num_decays)
                                        self._floating_backgrounds[0].\
                                            shrink_to_self(spec)
                                        spec.rebin(
                                            self._floating_backgrounds[0].
                                            _data.shape)
                                        spectra.append(spec)
                                        self._global_dict[bkgnd_name][str_int]\
                                            = spec
                                else:
                                    spectra.append(
                                        self._global_dict[bkgnd_name][str_int])
                            signal = parameter.apply_to(
                                spectra, parameter._name.split('_')[-1])
                    else:
                        if signal:
                            signal = parameter.apply_to(signal)
                        else:
                            signal = parameter.apply_to(self._signal)
                    i += 1
                signal._fit_config = fit_config
                print "shrinking to roi"
                self._data.shrink_to_self(signal)
                signal.rebin(self._data._data.shape)
                signal.scale(num_decays)
            else:
                signal = self._signal
            expected += signal.nd_project(self._signal_pars)

        # If single bin - sum over expected and observed
        if self._single_bin:
            expected = numpy.sum(expected)
            observed = numpy.sum(observed)

        # Calculate value of test statistic
        test_statistic = self._test_statistic.compute_statistic(
            observed.ravel(), expected.ravel())

        # Add penalty terms
        total_penalty = 0.
        for parameter in self._fit_config.get_pars():
            par = self._fit_config.get_par(parameter)
            current_value = par.get_current_value()
            prior = par.get_prior()
            sigma = par.get_sigma()
            # If sigma is explicitly None add no penalty term
            if (sigma is not None):
                total_penalty += self._test_statistic.get_penalty_term(
                    current_value, prior, sigma)

        # Check for per_bin flag
        if self._per_bin:
            test_statistic = test_statistic.reshape(spectrum._data.shape)

        chisq[0] = test_statistic + total_penalty
        ncount += 1

    def load_pre_made(self, spectrum, global_pars, cur_val=None):
        """ Load pre-made convolved spectra.

        This method is used to load a pre-made spectra convolved with
        certain resolution, energy-scale or shift values, or a
        combination of two or more at given values.

        The method loads the loads the correct spectra from HDF5s,
        stored in the given directory.

        Args:
          spectrum (:class:`echidna.core.spectra.Spectra`): Spectrum
            to convolve.
          global_pars (list): Of global parameters that you want to pre load.
          cur_val (float, optional): To overwrite the current value stored in
            the parameter.

        Returns:
          (:class:`echidna.core.spectra.Spectra`): Convolved spectrum,
            ready for applying further systematics or fitting.
        """
        # Locate spectrum to load from HDF5
        # Start with base spectrum name
        filename = os.path.basename(spectrum._location)
        if self._pre_made_base_dir:
            directory = self._pre_made_base_dir
        else:
            directory = os.path.dirname(spectrum._location) + '/'

        # Add current value of each global parameter
        for par in global_pars:
            dim = par._dimension
            added_dim = False
            if dim not in directory:
                added_dim = True
                directory += dim + '/'
            directory, filename = par.get_pre_convolved(directory, filename,
                                                        added_dim,
                                                        cur_val=cur_val)
        # Load spectrum from hdf5
        num_decays = spectrum._num_decays
        fit_config = spectrum._fit_config
        orig_num_decays = None
        if hasattr(spectrum, '_orig_num_decays'):
            orig_num_decays = spectrum._orig_num_decays
        if os.path.exists(directory + filename):
            spectrum = store.load(directory + filename)
        else:
            return
        if orig_num_decays:
            spectrum._num_decays = orig_num_decays
        spectrum.scale(num_decays)
        spectrum._fit_config = fit_config
        return spectrum

    def make_fixed_background(self, spectra_dict, shrink=True):
        ''' Makes a spectrum for fixed backgrounds and stores it in the class.

        Args:
          spectra_dict (dict): Dictionary containing spectra as keys and
            prior counts as values.
          shrink (bool, optional): If set to True (default), :meth:`shrink`
            method is called on the spectra shrinking it to the ROI.
        '''
        first = True
        for spectrum, scaling in spectra_dict.iteritems():
            # Copy so original spectra is unchanged
            self._logger.debug("Adding Spectra with name %s to"
                               "_fixed_background" % spectrum.get_name())
            spectrum = copy.deepcopy(spectrum)
            if first:
                first = False
                if shrink:
                    self.shrink_spectra(spectrum)
                spectrum.scale(scaling)
                total_spectrum = spectrum
                total_spectrum._name = "Fixed Background"
            else:
                if shrink:
                    self.shrink_spectra(spectrum)
                spectrum.scale(scaling)
                total_spectrum.add(spectrum)
        if shrink:
            self._fixed_background = total_spectrum  # No need to check
            self._fixed_pars = self.get_roi_pars(total_spectrum)
        else:
            self.set_fixed_background(total_spectrum, shrink)

    def remove_signal(self):
        """ Removes the signal spectra from the class.
        """
        self._signal = None

    def set_data(self, data, shrink=True):
        """ Sets the data you want to fit.

        Args:
          data (:class:`echidna.core.spectra.Spectra`): Data spectrum you
            want to fit.
          shrink (bool, optional): If set to True (default), :meth:`shrink`
            method is called on the spectra shrinking it to the ROI.
        """
        if shrink:
            self.shrink_spectra(data)
        else:
            self.check_spectra(data)
        self._data = data
        self._logger.debug("Setting Spectra with name %s as data" %
                           data.get_name())
        self._data_pars = self.get_roi_pars(data)

    def set_fit_config(self, fit_config):
        """
        Args:
          fit_config (:class:`echidna.core.spectra.FitConfig`): Config
            for fit.

        Raises:
          TypeError: If fit_config is not of type :class:`FitConfig`.
        """
        if isinstance(fit_config, GlobalFitConfig):
            self._fit_config = fit_config
        else:
            raise TypeError("fit_config type (%s) is invalid" %
                            type(fit_config))

    def set_fixed_background(self, fixed_background, shrink=True):
        """ Sets the fixed background you want to fit.

        Args:
          fixed_background (:class:`echidna.core.spectra.Spectra`): The
            fixed background spectrum you want to fit.
          shrink (bool, optional): If set to True (default) :meth:`shrink`
            method is called on the spectra shrinking it to the ROI.
        """
        if shrink:
            self.shrink_spectra(fixed_background)
        else:
            self.check_spectra(fixed_background)
        self._fixed_background = fixed_background
        self._fixed_pars = self.get_roi_pars(fixed_background)

    def set_floating_backgrounds(self, floating_backgrounds):
        """ Sets the floating backgrounds you want to fit.

        Args:
          floating_backgrounds (list): List of backgrounds you want to float
            in the fit.
        """
        floating_pars = []
        for background in floating_backgrounds:
            self._logger.debug("Adding Spectra with name %s to"
                               "_floating_backgrounds" % background.get_name())
            if background.get_fit_config():
                self.check_fit_config(background)
                # Spectrum has a valid fit config, add to GlobalFit Config
                self._fit_config.add_config(background.get_fit_config())
            floating_pars.append(self.get_roi_pars(background))
        self._floating_backgrounds = floating_backgrounds
        self._floating_pars = floating_pars

    def set_minimiser(self, minimiser=None):
        """ Sets the minimiser to use in fitting.

        Args:
          minimiser (:class:`echidna.limit.minimise.Minimiser`, optional): The
            minimiser to use in the fit.

        Raises:
          IndexError: If fit config contains no parameters.
        """
        self.check_all_spectra()  # All spectra should be set and checked first
        if minimiser:
            self._minimiser = minimiser
            self._logger.debug("Setting %s as minimiser" %
                               minimiser.get_name())
        else:  # Use default GridSearch
            if self._per_bin:
                if self._floating_backgrounds:
                    self._minimiser = GridSearch(
                        fit_config=self._fit_config,
                        spectra_config=self._floating_backgrounds[0].
                        get_config(),
                        name=self._fit_config.get_name(),
                        # This assumes fitting over all dimensions
                        per_bin=self._per_bin)
                else:
                    self._minimiser = GridSearch(
                        fit_config=self._fit_config,
                        spectra_config=self._data.get_config(),
                        name=self._fit_config.get_name(),
                        # This assumes fitting over all dimensions
                        per_bin=self._per_bin)
            else:
                self._minimiser = GridSearch(
                    fit_config=self._fit_config,
                    spectra_config=self._data.get_config(),
                    name=self._fit_config.get_name(),
                    per_bin=self._per_bin)
            self._logger.debug("Created GridSearch (%s) to use as minimiser" %
                               self._minimiser.get_name())

    def set_test_statistic(self, test_statistic):
        """ Sets the method you want to use to calculate test statistics in
          the fit.

        Args:
          test_statistic (:class:`echidna.limit.test_statistic.TestStatistic`):
            An appropriate class for calculating test statistics.
        """
        self._test_statistic = test_statistic
        self._logger.debug("Set _test_statistic as %s" % str(test_statistic))

    def set_roi(self, roi):
        """ Sets the region of interest you want to fit in.

        Args:
          roi (dictionary): The Region Of Interest you want to fit in.
            The format of roi is
            e.g. {"energy": (2.4, 2.6), "radial3": (0., 0.2)}
        """
        self.check_roi(roi)
        self._roi = roi
        self._logger.debug("Set _roi as %s" % str(roi))
        self._checked = False  # Must redo checks for a new roi

    def set_signal(self, signal, shrink=False):
        """ Sets the signal you want to fit.

        Args:
          signal (:class:`echidna.core.spectra.Spectra`): The signal
            spectrum you want to fit.
          shrink (bool, optional): If set to True
            :meth:`shrink` method is called on the spectra shrinking
            it to the ROI.
        """
        if shrink:
            self.shrink_spectra(signal)
        self._signal = signal
        self._logger.debug("Set _signal as Spectra with name %s" %
                           signal.get_name())
        self._signal_pars = self.get_roi_pars(signal)

    def set_pre_made_dir(self, directory):
        """ Sets the directory in which pre-made convolved spectra are
        stored.

        Args:
          directory (string): Directory in which pre-made spectra are
            located.
        """
        # Check that it is a valid directory
        if not os.path.isdir(directory):
            raise ValueError("Supplied directory %s is not a valid path."
                             % directory)
        self._pre_made_dir = directory

    def shrink_all(self):
        """ Shrinks all the spectra used in the fit to the roi.
        """
        if self._data:
            self.shrink_spectra(self._data)
        if self._fixed_background:
            self.shrink_spectra(self._fixed_background)
        if self._signal:
            self.shrink_spectra(self._signal)
        if self._floating_backgrounds:
            for background in self._floating_backgrounds:
                self.shrink_spectra(background)

    def shrink_spectra(self, spectra):
        """ Shrinks the spectra used in the fit to the roi.

        Args:
          spectra (:class:`echidna.core.spectra.Spectra`): Spectra you want to
            shrink to the roi.
        """
        shrink = {}
        for dim in self._roi:
            dim_type = spectra.get_config().get_dim_type(dim)
            par_low = dim + "_" + dim_type + "_low"
            par_high = dim + "_" + dim_type + "_high"
            shrink[par_low], shrink[par_high] = self._roi[dim]
        spectra.shrink(**shrink)
