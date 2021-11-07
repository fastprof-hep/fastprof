"""
Classes defining the result of likelihood fits

  * :class:`FitResult` : stores the result of a
    Maximumum-likelihood (ML) fit

  * :class:`PLRData` : stores the information
    relative to a hypothesis test based on the
    profile-likelihood ratio (PLR), including
    the results of the 2 fits involved

  * :class:`Raster` : stores PLR data for a
    set of tested hypotheses

"""

import numpy as np

from .base import Serializable
from .core import Model, Parameters, ModelPOI
from .minimizers import OptiMinimizer

class POIHypo(Serializable) :
  """Class describing an hypothesis on some or all of the model POIs

  Attributes:
    pars (dict)  : the hypothesis parameter values, in { name: value } format
  """

  def __init__(self, hypo : dict = {}) :
    """Initialize the `POIHypo` object

    Args:
      hypo : the hypothesis parameters, in { name: value } format
    """
    super().__init__()
    self.pars = { par: val for par, val in hypo.items() }

  def __contains__(self, par : str) -> bool :
    """Tests if a parameter is present

      Args:
        par : name of a parameter (either POI or NP)

      Returns:
        True if a parameter of this name is present, False otherwise
    """
    return par in self.pars

  def __getitem__(self, par : str) -> float :
    """Implement [] lookup of POI and NP names

      Args:
        par : name of a parameter (either POI or NP)

      Returns:
        The value of the parameter
    """
    return self.pars[par]

  def keys(self)   : return self.pars.keys()
  def values(self) : return self.pars.values()
  def items(self)  : return self.pars.items()

  def load_dict(self, sdict : dict) -> 'FitResult' :
    """Loads the object data

    Args:
      sdict : a dictionary of markup data from which to load the object information
    Returns:
      self
    """
    for par_name, par_value in sdict.items() : self.pars[par_name] = float(par_value)
    return self

  def fill_dict(self, sdict : dict) :
    """Saves the object data

    Args:
      sdict : a dictionary of markup data in which to store the object information
    """
    for par_name, par_value in self.pars.items() : sdict[par_name] = par_value

  def __str__(self) -> str :
    """A description string

    Returns:
      A description string for the object
    """
    return '\n'.join([ '%s = %g' % (par_name, par_value) for par_name, par_value in self.pars.items() ])


# -------------------------------------------------------------------------
class FitParameter(ModelPOI) :
  """Class representing a fit parameter

  Attributes:
     name          (str)   : the name of the parameter
     value         (float) : the value of the parameter (either a best-fit value or a fixed hypothesis value)
     error         (float) : the uncertainty on the parameter value
     min_value     (float) : the lower bound of the allowed range of the parameter
     max_value     (float) : the upper bound of the allowed range of the parameter
     initial_value (float) : the initial value of the parameter when performing fits to data
     unit          (str)   : the unit in which the parameter is expressed
  """

  def __init__(self, name : str = '', value : float = None, error : float = None,
               min_value : float = None, max_value : float = None,
               initial_value : float = None, unit : str = '') :
    """Initialize object attributes

      Missing arguments are set to None.

      Args:
        name          : the name of the parameter
        value         : the value of the parameter (either a best-fit value or a fixed hypothesis value)
        error         : the uncertainty on the parameter value
        min_value     : the lower bound of the allowed range of the parameter
        max_value     : the upper bound of the allowed range of the parameter
        initial_value : the initial value of the parameter when performing fits to data
        unit          : the unit in which the parameter is expressed
    """
    super().__init__(name, min_value, max_value, initial_value)
    self.value = value
    self.error = error

  def __str__(self) :
    """Provides a description string

      Returns:
        The object description
    """
    s = "'%s' :" % self.name
    if self.min_value is not None and self.max_value is not None : s +=' (min = %g, max = %g)' % (self.min_value, self.max_value)
    if self.initial_value is not None : s += ' init = %g' % self.initial_value
    if self.value is not None : s += ' %g' % self.value
    if self.error is not None : s += ' +/- %g'  % self.error
    if self.unit is not None : s += ' %s' % self.unit
    return s

  def load_dict(self, sdict) :
    """load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        FitParameter: self
    """
    super().load_dict(sdict)
    self.value     = self.load_field('value'    , sdict,  None, [int, float])
    self.error     = self.load_field('error'    , sdict,  None, [int, float])
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    super().fill_dict(sdict)
    if self.value is not None : sdict['value'] = self.unnumpy(self.value)
    if self.error is not None : sdict['error'] = self.unnumpy(self.error)


class FitResult(Serializable) :
  """Class describing the result of a ML fit

  Attributes:
    name    (str)   : a name for the object
    fitpars (dict)  : the best-fit parameters in { name: FitParameter } format
    nll     (float) : the best-fit NLL value
    model   (Model) : the statistical model
  """

  def __init__(self, name : str = '', fitpars : Parameters = None, nll : float = None, model : Model = None, hypo : POIHypo = None) :
    """Initialize the `FitResult` object

    Args:
      name    : a name for the object
      fitpars : the best-fit parameter values
      nll     : the best-fit NLL value
      model   : the statistical model
      hypo    : the fit hypothesis, for fixed-POI fits.
                If provided, the POI values are copied from
                `hypo` to the object (default: None)
    """
    super().__init__()
    self.name = name
    self.fitpars = {}
    if not fitpars is None and not model is None :
      for name in model.pois : self.fitpars[name] = FitParameter(name, value=fitpars[name])
      for name in model.nps  : self.fitpars[name] = FitParameter(name, value=model.nps[name].unscaled_value(fitpars[name]))
    if hypo is not None :
      for name, value in hypo.items() : self.fitpars[name] = FitParameter(name, value=value)
    self.nll = nll
    self.model = model

  def pars(self) -> Parameters :
    """The best-fit parameters

    Returns:
      best-fit parameters, as a :class:`Parameters` object
    """
    par_dict = { par.name : par.value for par in self.fitpars.values() }
    return Parameters(par_dict, model=self.model).set_from_dict(par_dict, unscaled_nps=True)

  def set_poi_values_and_ranges(self, minimizer : OptiMinimizer) :
    """Set POI initial values and ranges from the fit

    Updates the initial values and ranges of the POIs
    in the minimizer objects to those stored in the fit

    Args:
      minimizer : minimizer object to update
    """
    if minimizer.init_pois is None :
      minimizer.init_pois = self.pars()
    else :
      minimizer.init_pois.set_from_dict(self.pars().dict(pois_only=True))
    minimizer.bounds = { poi_name : (self.fitpars[poi_name].min_value, self.fitpars[poi_name].max_value) for poi_name in self.model.pois }
    return self

  def load_dict(self, sdict : dict) -> 'FitResult' :
    """Loads the object data

    Args:
      sdict : a dictionary of markup data from which to load the object information
    Returns:
      self
    """
    for par_name, par_dict in sdict['fit_pars'].items() :
      fitpar = FitParameter(par_name).load_dict(par_dict)
      self.fitpars[fitpar.name] = fitpar
    self.nll = sdict['nll']
    return self

  def fill_dict(self, sdict : dict) :
    """Saves the object data

    Args:
      sdict : a dictionary of markup data in which to store the object information
    """
    sdict['fit_pars'] = { par.name : par.dump_dict() for par in self.fitpars.values() }
    sdict['nll'] = self.nll

  def __str__(self) -> str :
    """A description string

    Returns:
      A description string for the object
    """
    return  "  Fit '%s' : nll = %g, pars :\n%s" % (self.name, self.nll, str(self.pars()))


class PLRData(Serializable) :
  """Class storing the information relative to a PLR-based hypothesis test

  The information consists of the hypothesis under test, the results of
  the two fits involved (fixed-POI and free-POI), and the associated
  p-values and test statistic values.

  Attributes:
    name  (str)   : a name for the object
    hypo (Parameters) : the tested hypothesis
    free_fit (FitResult) : the result of the free-POI fit
    hypo_fit (FitResult) : the result of the fixed-POI fit
    test_statistics (dict) : a dict of computed test statistic values
    pvs (dict) : a dict of computed p-values
    asimov (PLRData)  : a pointer to another :class:`PLRData` object
                        storing the information on an Asimov test
    pois (dict) : the free-fit POI values, in the format { par_name : par_value }
    model   (Model) : the statistical model
  """

  def __init__(self, name = '', hypo : POIHypo = None, free_fit = None, hypo_fit = None, test_statistics = None, pvs = None, asimov = None, model = None, full_hypo = None) :
    """Initialize the `FitResult` object

    Args:
      name     : a name for the object
      hypo     : the tested hypothesis
      free_fit : the result of the free-POI fit
      hypo_fit : the result of the fixed-POI fit
      test_statistics : a dict of computed test statistic values
      pvs      : a dict of computed p-values
      asimov   : a pointer to another :class:`PLRData` object
                 storing the information on an Asimov test
      model   : the statistical model
    """
    super().__init__()
    self.name = name
    self.hypo = hypo
    self.free_fit = free_fit
    self.hypo_fit = hypo_fit
    self.test_statistics = test_statistics if test_statistics != None else {}
    self.pvs = pvs if pvs != None else {}
    self.asimov = asimov
    self.model = model
    self.full_hypo = full_hypo
    self.pois = {}
    self.update()

  def update(self) :
    """Private method to fill internal information

    The relevant information is the `pois` attribute, which provides
    quick access to the best-fit values for all POIs, and the `tmu`
    value, which is computed from the fit results and stored in the
    `test_statistics` array under the `tmu` key.
    """
    if self.hypo is not None and self.model is not None : self.pois = { name : self.model.pois[name] for name in self.hypo.keys() }
    if self.free_fit is not None and self.hypo_fit is not None and not 'tmu' in self.test_statistics : self.compute_tmu()

  def compute_tmu(self) :
    """Computes the basic PLR test statistic `tmu`.

    `tmu` is defined as :math:`-2\log(L(hypo_poi)/L(free_poi))` and
    is computed directly from the fit results. The value is stored
    under the `tmu` key in `test_statistics`.
    """
    self.test_statistics['tmu'] = 2*(self.hypo_fit.nll - self.free_fit.nll)

  def set_asimov(self, asimov_plr_data : 'PLRData', local_key : str = 'tmu_A0') :
    """Sets Asimov values

    Sets the `asimov` pointer to another :class:`PLRData` object
    containing the PLR information of a fit to an Asimov dataset.
    The Asimov `tmu` value is also stored in `test_statistics`
    under the specified key (default: `tmu_A0`)

    Args:
      asimov_plr_data : an object containing the Asimov PLR data
      local_key : the key under which to store the Asimov `tmu`
    """
    self.test_statistics[local_key] = asimov_plr_data.test_statistics['tmu']
    self.asimov = asimov_plr_data

  def load_dict(self, sdict : dict) -> 'PLRData' :
    """Loads the object data

    Args:
      sdict : a dictionary of markup data from which to load the object information
    Returns:
      self
    """
    self.hypo = POIHypo().load_dict(sdict['hypo'])
    self.full_hypo = Parameters(sdict['full_hypo'], model=self.model) if 'full_hypo' in sdict else None
    self.free_fit = FitResult('free_fit', model=self.model).load_dict(sdict['free_fit'])
    self.hypo_fit = FitResult('hypo_fit', model=self.model, hypo=self.hypo).load_dict(sdict['hypo_fit'])
    self.test_statistics = sdict['test_statistics'] if 'test_statistics' in sdict else {}
    self.pvs = sdict['pvs'] if 'pvs' in sdict else {}
    self.update()
    return self

  def fill_dict(self, sdict : dict) :
    """Save the object data

    Args:
      sdict : a dictionary of markup data in which to save the object information
    Returns:
      self
    """
    sdict['hypo'] = self.hypo.dump_dict()
    if self.full_hypo is not None : sdict['full_hypo'] = self.full_hypo.dict()
    sdict['free_fit'] = self.free_fit.dump_dict()
    sdict['hypo_fit'] = self.hypo_fit.dump_dict()
    sdict['test_statistics'] = self.test_statistics
    sdict['pvs'] = self.pvs

  def __str__(self) -> str :
    """A description string for the object

    Returns:
      A string describing the object contents
    """
    s = "Profile-likelihood ratio data '%s' for hypothesis : %s" % (self.name, str(self.hypo))
    s += '\n  test statistics : %s' % str(self.test_statistics)
    s += '\n  p-values : %s' % str(self.pvs)
    s += '\n  Unconditional fit:' + str(self.free_fit)
    s += '\n  Conditional fit:' + str(self.hypo_fit)
    return s


class Raster(Serializable) :
  """Class describing PLR test results over multiple hypotheses

  Attributes:
    name     (str)   : a name for the object
    model    (Model) : the statistical model
    plr_data (dict)  : the PLR information for each hypothesis
        in the form { hypo : PLRData }, mapping
        :class:`Parameters` objects to :class:`PLRData`.
    use_global_best_fit (bool) : if `True`, update the free_fit field
        of all the stored :class:`PLRData` with the one providing the
        best fit (smallest `nll`)
    fill_missing (bool) : if `True`, fill the `tmu` values at each
        hypothesis with the value computed from the stored `nll` values.
  """

  def __init__(self, name : str = '', plr_data : dict = None, use_global_best_fit : bool = True, fill_missing : bool = True, filename : str = None,
              load_asimov : str = 'asimov', model : Model = None) :
    """Initialize the `Raster` object

    Args:
      name  : a name for the object
      plr_data : the PLR information for each hypothesis
          in the form { POIHypo : PLRData }, mapping
          :class:`Parameters` objects to :class:`PLRData`.
      use_global_best_fit : if `True`, update the free_fit field
          of all the stored :class:`PLRData` with the one providing the
          best fit (smallest `nll`)
      fill_missing  : if `True`, fill the `tmu` values at each
          hypothesis with the value computed from the stored `nll` values.
      filename : if not `None`, load the plr_data information from the
                 specified file.
      load_asimov : if not `None`, also load Asimov information from `filename`
        and set the `asimov` pointers in the :classs:`PLRData` objects to this
        information.
      model : the statistical model
    """
    super().__init__()
    self.name = name
    self.model = model
    self.plr_data = plr_data if plr_data != None else {}
    self.use_global_best_fit = use_global_best_fit
    self.fill_missing = fill_missing
    if filename is not None : self.load_with_asimov(filename, 'asimov') if load_asimov is not None else self.load(filename)
    if self.use_global_best_fit : self.set_global_best_fit()
    if self.fill_missing : self.compute_tmu()

  @classmethod
  def have_compatible_pois(cls, hypos : list) -> list :
    """Utility function to check the compatilibity of a list
       of hypotheses.

    If the hypotheses are compatible for the purpose of building
    a raster -- i.e. are defined in terms of the same POIs, in the
    same order -- then returns the list of these POIs.
    Otherwise, returns None.
    
    Args:
      cls : class object input (not actually needed here)
      hypos : list of hypotheses to test
    Returns:
      the list of POIs, or None
    """
    if len(hypos) <= 1 : return True
    pois = list(hypos[0].keys())
    if any([ list(hypo.keys()) != pois for hypo in hypos[1:] ]) : return None
    return pois
      
  def key_value(self, key : str, hypo : Parameters) -> float :
    """Utility function to retrieve numerical values from the PLR data

    Method called by :meth:`print` to access various numerical values
    in the PLR data object indexed by `hypo`. `key` can be either a
    key in the `pvs` and `test_statistics` collections, a best-fit
    parameter value in the form 'best\_' + par_name, or a hypothesis
    parameter value in the form par_name.

    Args:
      key : string indexing a particular numerical value
      hypo : hypothesis for which to retrieve the value
    Returns:
      the numerical value
    """
    if not hypo in self.plr_data : raise KeyError('While trying to access key %s, hypo %s was not found in raster %s.' % (key, str(hypo), self.name))
    for poi in self.pois() :
      if key == poi and poi in hypo : return hypo[poi]
      if key == 'best_' + poi : return self.plr_data[hypo].free_fit.fitpars[poi].value
    if key in self.plr_data[hypo].pvs :
      value = self.plr_data[hypo].pvs[key]
      return value if not isinstance(value, tuple) else value[0]
    if key in self.plr_data[hypo].test_statistics : return self.plr_data[hypo].test_statistics[key]
    raise KeyError('No data found for key %s in hypo %s in raster %s.' % (key, str(hypo), self.name))

  def is_filled(self, key : str, only_pv : bool = False, only_ts : bool = False) :
    """Utility function to check if all the data is available for a given key 

    Args:
      key : string indexing a particular entry
      as_pv : check among p-value entries
      as_ts : check among test statistic entries
    Returns:
      True if the information for `key` is filled in all the raster hypotheses,
      False otherwise.
    """
    for hypo in self.plr_data :
      if (key not in self.plr_data[hypo].pvs and not only_ts) and (key not in self.plr_data[hypo].test_statistics and not only_pv) :
        return False
    return True

  def pois(self) -> dict :
    """Shortcut method to the list of POIs

    Returns the POIs from the first raster point, as
    a { par_name : par_value } dictionary.

    Returns:
      POIs as a { par_name : par_value } dictionary.
    """
    hypo_pois = self.have_compatible_pois(list(self.plr_data.keys()))
    return { poi_name : self.model.pois[poi_name] for poi_name in hypo_pois }

  def set_global_best_fit(self) :
    """Set the same best-fit results across all points

    In theory, all the points should always share the same
    best-fit results, since this does not depend on the
    hypothesis. However they can differ due to numerical
    effects (the starting point of the fits are normally
    at the hypothesis point, and this can affect the result).
    This method finds the best minimum (smallest `nll`) among
    all the best-fit points, and sets the `nll` and best-fit
    parameter values to this one for all raster points
    """
    if len(self.plr_data) == 0 : return
    nlls = { hypo : plr_data.free_fit.nll for hypo, plr_data in self.plr_data.items() }
    best_hypo = min(nlls, key=nlls.get)
    for plr_data in self.plr_data : plr_data.free_fit = self.plr_data[best_hypo].free_fit

  def set_asimov(self, asimov_raster : 'Raster', local_key : str = 'tmu_A0') :
    """Sets Asimov values for all scan points

    Calls :meth:`PLRData.set_asimov` at each raster point:
    sets the `asimov` pointer at each point to the
    corresponding :class:`PLRData` object in `asimov_raster`
    The Asimov `tmu` value is also stored in `test_statistics`
    array of each point under the specified key (default: `tmu_A0`)

    Args:
      asimov_plr_data : an object containing the Asimov PLR data
      local_key : the key under which to store the Asimov `tmu`
    """
    if len(self.plr_data) != len(asimov_raster.plr_data) :
      raise KeyError('Cannot set Asimov scan data with a different number of hypotheses (%d instead of %d).' % (len(self.plr_data), len(asimov_raster.plr_data)))
    for plr_data, asimov_data in zip(self.plr_data.values(), asimov_raster.plr_data.values()) :
      plr_data.set_asimov(asimov_data, local_key)

  def compute_tmu(self) :
    """Computes the PLR test statistic `tmu`.

    Calls :meth:`PLRData.compute_tmu` at each raster point.
    `tmu` is defined as :math:`-2\log(L(hypo_poi)/L(free_poi))` and
    is computed directly from the fit results. The value is stored
    under the `tmu` key in `test_statistics`.
    """
    for plr_data in self.plr_data.values() : plr_data.compute_tmu()

  def load_dict(self, sdict : dict) -> 'Raster' :
    """Loads the object data

    Args:
      sdict : a dictionary of markup data from which to load the object information
    Returns:
      self
    """
    for i, plr_dict in enumerate(sdict[self.name]) :
      plr_data = PLRData('%s_%d' % (self.name, i), model=self.model).load_dict(plr_dict)
      self.plr_data[plr_data.hypo] = plr_data
    if self.use_global_best_fit : self.set_global_best_fit()
    if self.fill_missing : self.compute_tmu()
    return self

  def fill_dict(self, sdict : dict) :
    """Saves the object data

    Args:
      sdict : a dictionary of markup data in which to store the object information
    """
    sdict[self.name] = []
    for plr_data in self.plr_data.values() :
      sdict[self.name].append(plr_data.dump_dict())

  def load_with_asimov(self, filename : str, asimov_key : str = 'asimov') -> 'Raster' :
    """Loads the object data and Asimov information

    Args:
      filename : the file to load from
      asimov_key : the key under which the Asimov information is stored
    Returns:
      self
    """
    self.load(filename)
    asimov_raster = Raster(asimov_key, model=self.model).load(filename)
    self.set_asimov(asimov_raster)
    return self

  def __str__(self) -> str :
    """Provides a description string for the object

    Returns:
      a description string
    """
    s = ''
    s += 'PLR data : '
    for hypo, plr_data in self.plr_data.items() :
      s += '\nHypo :' + str(hypo)
      s += '\n' + str(plr_data)
    return s

  def print(self, keys : list = None, verbosity : int = 0, other : 'Raster' = None) -> str :
    """Print a full description of the stored information

    Prints out the PLR information for all hypotheses. The fields
    are displayed in columns, with a line for each hypothesis.
    The displayed information is governed by the `verbosity` argument:
    If `verbosity=1`, the p-value, `cls` and `clb` values are printed,
    while for `verbosity=2` the `tmu` and best-fit values of all POIs
    are also shown. Values that are not available are omitted.
    If `verbosity=0`, nothing is printed by default.
    In addition to the above, the values for particular fields can be
    displayed by providing the corresponding keys in the `keys`
    argument (see :meth:`key_value` for the available values).
    If `print_limits` is true, the interpolated 95% CL limit is also
    shown
    If the raster `other` is not `None`, its values are shown alongside
    those of the current raster for comparison

    Args:
      keys : list of key values for fields to display
      verbosity : verbosity level -- 0,1 or 2 (default: 1)
      print_limits: if True, diplay interpolated 95% CL limits
      other : another raster object for which to display the
              same results (default: None)
    Returns:
      the description string
    """
    if len(self.plr_data) == 0 : return ''
    plr_template = list(self.plr_data.values())[0]
    if keys is None :
      keys = []
      if verbosity == 0 : verbosity = 1
    if verbosity > 0 :
      keys = list(self.pois().keys()) + keys
      if 'pv' in plr_template.pvs : keys += [ 'pv' ]
      if 'cls' in plr_template.pvs : keys += [ 'cls' ]
      if 'clb' in plr_template.pvs : keys += [ 'clb' ]
      if verbosity > 1 : keys.extend([ 'tmu' ] + [ 'best_' + k for k in self.pois().keys() ])
    s = ''
    for key in keys :
      s += '| %-15s ' % key
      if not other is None and not key in self.pois().keys() : s += '| %-15s ' % ('%s (%s)' % (key, other.name))
    for hypo, plr_data in self.plr_data.items() :
      s += '\n'
      for key in keys :
        s += '| %-15g ' % self.key_value(key, hypo)
        if not other is None and not key in self.pois().keys() : s += '| %-15g ' % other.key_value(key, hypo)
    if verbosity > 2 :
      for hypo, plr_data in self.plr_data.items() :
        s += '\n------------------------------------------------'
        s += '\nHypo :' + str(hypo)
        s += '\n' + str(plr_data)
    print(s)
    return s
