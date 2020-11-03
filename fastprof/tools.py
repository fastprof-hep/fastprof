"""Module defining utility classes for reporting
  profile-likelihood test results over multiple
  hypotheses.

  The classes are

  * :class:`FitResult` : stores the result of a
    Maximumum-likelihood (ML) fit

  * :class:`PLRData` : stores the information
    relative to a hypothesis test based on the
    profile-likelihood ratio (PLR), including
    the results of the 2 fits involved

  * :class:`Raster` : stores PLR data for a
    set of tested hypotheses

  * :class:`TestStatisticCalculator` : perform
    test statistic and p-value computations using
    :class:`PLRData` storage. Derived classes are
    defined for the main test statistics (for now,
    :class:`test_statistics.QMu` and
    :class:`test_statistics.QMuTilda`.

"""

import json
import math
import scipy
import numpy as np
from abc import abstractmethod

from .core import Model, Data, Parameters, JSONSerializable, ModelPOI
from .minimizers import NPMinimizer, POIMinimizer, OptiMinimizer
from .test_statistics import QMu, QMuTilda


class FitResult(JSONSerializable) :
  """Class describing the result of a ML fit

  Attributes:
    name    (str)   : a name for the object
    fitpars (dict)  : the best-fit parameters in { name: value } format
    nll     (float) : the best-fit NLL value
    model   (Model) : the statistical model
  """

  def __init__(self, name : str = '', fitpars : Parameters = None, nll : float = None, model : Model = None, hypo : Parameters = None) :
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
      for name in model.pois : self.fitpars[name] = ModelPOI(name, value=fitpars[name])
      for name in model.nps  : self.fitpars[name] = ModelPOI(name, value=model.nps[name].unscaled_value(fitpars[name]))
    if not hypo is None :
      for name, value in hypo.dict(pois_only = True).items() : self.fitpars[name] = ModelPOI(name, value=value)
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

  def load_jdict(self, jdict : dict) -> 'FitResult' :
    """Loads the object data

    Args:
      jdict : a dictionary of JSON data from which to load the object information
    Returns:
      self
    """
    for par_name, par_dict in jdict['fit_pars'].items() :
      fitpar = ModelPOI(par_name).load_jdict(par_dict)
      self.fitpars[fitpar.name] = fitpar
    self.nll = jdict['nll']
    return self

  def fill_jdict(self, jdict : dict) :
    """Saves the object data

    Args:
      jdict : a dictionary of JSON data in which to store the object information
    """
    jdict['fit_pars'] = { par.name : par.dump_jdict() for par in self.fitpars }
    jdict['nll'] = self.nll

  def __str__(self) -> str :
    """A description string

    Returns:
      A description string for the object
    """
    return  "  Fit '%s' : nll = %g, pars :\n%s" % (self.name, self.nll, str(self.pars()))


class PLRData(JSONSerializable) :
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

  def __init__(self, name = '', hypo = None, free_fit = None, hypo_fit = None, test_statistics = None, pvs = None, asimov = None, model = None) :
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
    self.pois = {}
    self.update()

  def update(self) :
    """Private method to fill internal information

    The relevant information is the `pois` attribute, which provides
    quick access to the best-fit values for all POIs, and the `tmu`
    value, which is computed from the fit results and stored in the
    `test_statistics` array under the `tmu` key.
    """
    if self.hypo and self.free_fit : self.pois = { name : self.free_fit.fitpars[name] for name in self.hypo.model.pois.keys() }
    if self.free_fit is not None and self.hypo_fit is not None and not 'tmu' in self.test_statistics : self.compute_tmu()

  def hypo_pars(self) -> Parameters :
    """Provides the hypothesis as a :class:`Parameters` object

    Returns:
      The hypothesis definition as a :class:`Parameters` object
    """
    return Parameters(model=self.model).set_from_dict(self.hypo)

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

  def load_jdict(self, jdict : dict) -> 'PLRData' :
    """Loads the object data

    Args:
      jdict : a dictionary of JSON data from which to load the object information
    Returns:
      self
    """
    self.hypo = Parameters(jdict['hypo'], model=self.model)
    self.free_fit = FitResult('free_fit', model=self.model).load_jdict(jdict['free_fit'])
    self.hypo_fit = FitResult('hypo_fit', model=self.model, hypo=self.hypo).load_jdict(jdict['hypo_fit'])
    self.test_statistics = jdict['test_statistics'] if 'test_statistics' in jdict else {}
    self.pvs = jdict['pvs'] if 'pvs' in jdict else {}
    self.update()
    return self

  def fill_jdict(self, jdict : dict) :
    """Save the object data

    Args:
      jdict : a dictionary of JSON data in which to save the object information
    Returns:
      self
    """
    jdict['free_fit'] = self.free_fit.dump_jdict()
    jdict['hypo_fit'] = self.hypo_fit.dump_jdict()
    jdict['test_statistics'] = self.test_statistics
    jdict['pvs'] = self.pvs

  def __str__(self) -> str :
    """A description string for the object

    Returns:
      A string describing the object contents
    """
    s = "Profile-likelihood ratio data '%s' for hypothesis:" % self.name + str(self.hypo.dict(pois_only=True))
    s += '\n  test statistics : %s' % str(self.test_statistics)
    s += '\n  p-values : %s' % str(self.pvs)
    s += '\n  Unconditional fit:' + str(self.free_fit)
    s += '\n  Conditional fit:' + str(self.hypo_fit)
    return s


class Raster(JSONSerializable) :
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
          in the form { hypo : PLRData }, mapping
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

  def pois(self) -> dict :
    """Shortcut method to the list of POIs

    Returns the POIs from the first raster point, as
    a { par_name : par_value } dictionary.

    Returns:
      POIs as a { par_name : par_value } dictionary.
    """
    return list(self.plr_data.values())[0].pois if len(self.plr_data) > 0 else None

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

  def contour(self, pv_key : str = 'cls', target_pv : float = 0.05, order : int = 3, log_scale : bool = True, with_error : bool = False) -> float :
    """Compute a contour at a predefined p-value level

    The contour is obtained by interpolating the p-values at each raster
    point to identify the points in the space of hypothesis values
    where the p-value corresponds to the target p-value.

    The set of interpolated p-values is made up of elements from the
    `self.pvs` dict of each :class:`PLRData` object, at the key
    position provided by the argument `pv_key`.

    The current implementation works in 1D only, with defaults designed
    to compute a :math:`CL_s` upper limit at 95% CL. The interpolation is
    performed by the :meth:`Raster.interpolate_limit` method.

    Args:
      pv_key : the key of the selected p-values in :class:`PLRData`
      target_pv : the target p-value
      order : the order of the interpolation (see :meth:`Raster.interpolate_limit`)
      log_scale : if `True`, interpolate in the log of the p-values. If `False`
         (default), interpolate the p-values directly (see :meth:`Raster.interpolate_limit`)
      with_error : if `True`, return a triplet with (nominal limit, limit + 1sigma error,
         limit - 1sigma error), where the uncertainties are propagated from those of the p-values.

    Returns:
      self
    """
    if len(self.pois()) > 1 : raise ValueError('Cannot interpolate limit in more than 1 dimension.')
    poi_name = list(self.pois())[0]
    hypos = []
    values = []
    if with_error :
      values_up = []
      values_dn = []
    for hypo, plr_data in self.plr_data.items() :
      hypos.append(hypo[poi_name])
      if not pv_key in plr_data.pvs : raise KeyError("P-value '%s' not found at hypo %s." % (pv_key, str(hypo.dict(pois_only=True))))
      if not with_error :
        values.append(plr_data.pvs[pv_key])
      else :
        if not isinstance(plr_data.pvs[pv_key], tuple) or len(plr_data.pvs[pv_key]) < 2 : raise ValueError("p-value data at key '%s' in hypo %s does not contain error bands." %  (pv_key, str(hypo.dict(pois_only=True))))
        values.append(plr_data.pvs[pv_key][0])
        values_up.append(plr_data.pvs[pv_key][0] + plr_data.pvs[pv_key][1])
        values_dn.append(plr_data.pvs[pv_key][0] - plr_data.pvs[pv_key][1])
    limit = self.interpolate_limit(hypos, values, target_pv, order, log_scale, name = 'nominal %s[%s]' % (pv_key, poi_name))
    if not with_error : return limit
    limit_up = self.interpolate_limit(hypos, values_up, target_pv, order, log_scale, name = 'nominal+err %s[%s]' % (pv_key, poi_name))
    limit_dn = self.interpolate_limit(hypos, values_dn, target_pv, order, log_scale, name = 'nominal-err %s[%s]' % (pv_key, poi_name))
    return (limit, limit_up, limit_dn)

  def interpolate_limit(self, hypos : list, pvs : list, target_pv : float = 0.05, order : int = 3, log_scale : bool = True, name : str = 'pv') -> float :
    """Perform a one-dimensional interpolation to compute a limit

    Takes 2 lists of same size, corresponding to the `x` (`hypos)
    and `y` (`pvs`) dimensions, and interpolates to find the values
    giving pv=target_pv.

    If multiple values are found, the first one is returned. If no
    value is found, returns `None`.

    Uses the `InterpolatedUnivariateSpline` method from `scipy`, with
    spline order specified by the `order` parameter. If `log_scale` is
    `True`, the interpolation is performed in the log of the p-values.

    Args:
      hypos : list of POI values
      pvs   : list of p-values
      target_pv : the target p-value
      order : the order of the interpolation (see :meth:`Raster.interpolate_limit`)
      log_scale : if `True`, interpolate in the log of the p-values. If `False`
         (default), interpolate the p-values directly (see :meth:`Raster.interpolate_limit`)

    Returns:
      The interpolated solution
    """
    interp_hypos  = []
    interp_pvs = []
    for hypo, pv in zip(hypos, pvs) :
      if log_scale :
        if pv <= 0 : continue
        value = math.log(pv/target_pv)
      else :
        value = pv - target_pv
      interp_pvs.append(value)
      interp_hypos.append(hypo)
    if len(interp_hypos) < 2 :
      print('Cannot interpolate using %d point(s), giving up' % len(interp_hypos))
      return None
    if len(interp_hypos) < order + 1 :
      order = len(interp_hypos) - 1
      print('Reducing interpolation order to %d to match the number of available points' % order)
    finder = scipy.interpolate.InterpolatedUnivariateSpline(interp_hypos, interp_pvs, k=order)
    if order == 3 :
      roots = finder.roots()
    else :
      print('Root-finding not supported yet for non-cubic splines, failing')
      return None
    if len(roots) == 0 :
      print("No solution found for %s = %g." % (name, target_pv))
      #print("No solution found for %s = %g. Interpolation set:" % (name, target_pv))
      #print([(h,v) for h,v in zip(interp_hypos, interp_pvs)])
      return None
    if len(roots) > 1 :
      print('Multiple solutions found for %s = %g (%s), returning the first one' % (name, target_pv, str(roots)))
    return roots[0]

  def load_jdict(self, jdict : dict) -> 'Raster' :
    """Loads the object data

    Args:
      jdict : a dictionary of JSON data from which to load the object information
    Returns:
      self
    """
    for i, plr_dict in enumerate(jdict[self.name]) :
      plr_data = PLRData('%s_%d' % (self.name, i), model=self.model).load_jdict(plr_dict)
      self.plr_data[plr_data.hypo] = plr_data
    if self.use_global_best_fit : self.set_global_best_fit()
    if self.fill_missing : self.compute_tmu()
    return self

  def fill_jdict(self, jdict : dict) :
    """Saves the object data

    Args:
      jdict : a dictionary of JSON data in which to store the object information
    """
    jdict[self.name] = []
    for plr_data in self.plr_data.values() :
      jdict[self.name].append(plr_data.dump_jdict())

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
      s += '\nHypo :' + str(hypo.dict(pois_only=True))
      s += '\n' + str(plr_data)
    return s

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
    if not hypo in self.plr_data : raise KeyError('While trying to access key %s, hypo %s was not found in raster %s.' % (key, str(hypo.dict(pois_only=True)), self.name))
    for poi in self.pois() :
      if key == poi : return hypo[poi]
      if key == 'best_' + poi : return self.plr_data[hypo].free_fit.fitpars[poi].value
    if key in self.plr_data[hypo].pvs :
      value = self.plr_data[hypo].pvs[key]
      return value if not isinstance(value, tuple) else value[0]
    if key in self.plr_data[hypo].test_statistics : return self.plr_data[hypo].test_statistics[key]
    raise KeyError('No data found for key %s in hypo %s in raster %s.' % (key, str(hypo.dict(pois_only=True)), self.name))

  def print(self, keys : list = None, verbosity : int = 0, print_limits : bool = True, other : 'Raster' = None) -> str :
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
    if print_limits and len(self.pois()) == 1 and 'cls' in plr_template.pvs :
      limit = self.contour('cls', 0.05)
      limit_str = '%g' % limit if limit != None else 'not computable'
      s += '\n' + "Asymptotic 95%% CLs limit for raster '%s' = %s" % (self.name, limit_str)
      if not other is None :
        limit = other.contour('cls', 0.05)
        limit_str = '%g' % limit if limit != None else 'not computable'
        s += '\n' + "Asymptotic 95%% CLs limit for raster '%s' = %s" % (other.name, limit_str)
    if verbosity > 2 :
      for hypo, plr_data in self.plr_data.items() :
        s += '\nHypo :' + str(hypo.dict(pois_only=True))
        s += '\n' + str(plr_data)
    print(s)
    return s


class TestStatisticCalculator :
  """Base class for test statistic computations

  Base class for classes implementing the computation
  of various test statistics. This includes

  * computing the `tmu` test statistic from a linear
     model (included in the minimizer) and a dataset

  * computing other test statistics from PLR data obtained
    either from the previous step, or externally,
    e.g. using :mod:`fit_ws.py`

  * Filling p-value information in PLR data
    objects based on the test statistic
    information.

  Each step operates on a :class:`PLRData` object,
  which is either created or filled with more data.

  The test statistics themselves are defined in the
  :mod:`test_statistics.py` module. Each type
  corresponds to a separate class derived from
  :class:`TestStatisticCalculator`.

  Attributes:
    minimizer (POIMinimizer) : a minimizer object
      (see :mod:`minimizers.py`).
  """

  def __init__(self, minimizer : POIMinimizer) :
    """Initialize the `TestStatisticCalculator` object

    Args:
      minimizer : a minimizer algorithm.
    """
    self.minimizer = minimizer

  @classmethod
  def poi(self, plr_data : PLRData) -> str  :
    """Return the name of the POI in PLR data

    Note that for now only the single-POI case is
    supported.

    Args:
      plr_data : an object storing PLR information
    Returns:
      the POI name
    """
    if len(plr_data.pois) != 1 : raise ValueError('Can currently only compute test statistics for a single POI, here %s has %d.' % (plr_data.name, len(plr_data.pois)))
    return list(plr_data.pois)[0]

  @abstractmethod
  def fill_pv(self, plr_data : PLRData) :
    """Fills p-value information from PLR data

    Abstract method to be reimplemented in derived classes
    corresponding to a specific test statistic.

    Args:
      plr_data : an object storing PLR information
    """
    pass

  def compute_fast_plr(self, hypo : Parameters, data : Data, name : str = 'fast') -> PLRData :
    """Compute `tmu` from a dataset

    The computation is performed in the linear approximation
    using `self.minimizer` (which contains the model
    information) on the provided dataset, at the specified
    hypothesis value.
    A class:`PLRData` object is created with the computed
    PLR information, and this is then used to compute
    the `tmu` value.

    Args:
      hypo : the hypothesis for which to compute the PLR
      data : the input dataset
      name : name of the output :class:`PLRData` object
    Returns:
      an object containing the PLR information
    """
    plr_data = PLRData(name, hypo, model=data.model)
    tmu = self.minimizer.tmu(hypo, data, hypo)
    plr_data.test_statistics['tmu'] = tmu
    plr_data.free_fit = FitResult('free_fit', self.minimizer.free_pars, self.minimizer.free_nll, model=data.model)
    plr_data.hypo_fit = FitResult('hypo_fit', self.minimizer.hypo_pars, self.minimizer.hypo_nll, model=data.model)
    plr_data.update() # includes `tmu` computation
    return plr_data

  def compute_fast_q(self, hypo : Parameters, data : Data, name : str = 'fast') -> PLRData :
    """Compute `tmu` and the associated p-value from a dataset

    Computes `tmu` using the :meth:`compute_fast_plr`
    method above, and adds the Asimov dataset values and
    the p-value computation from :meth:`fill_pv`.

    Args:
      hypo : the hypothesis for which to compute the PLR
      data : the input dataset
      name : name of the output :class:`PLRData` object
    Returns:
      an object containing the PLR information
    """
    fast_plr_data = self.compute_fast_plr(hypo, data, name)
    asimov = data.model.generate_expected(0, NPMinimizer(data))
    asimov_plr_data = self.compute_fast_plr(hypo, asimov, 'fast_asimov')
    fast_plr_data.set_asimov(asimov_plr_data)
    self.fill_pv(fast_plr_data)
    return fast_plr_data

  def fill_all_pv(self, raster : Raster) :
    """Compute p-values in a raster containing PLR values

    Takes as input a raster object containing PLR values at each
    hypothesis point and compute the associated p-values

    Args:
      raster : a raster object
    Returns:
      self
    """
    for plr_data in raster.plr_data.values() : self.fill_pv(plr_data)

  def compute_fast_results(self, hypos : list, data : Data, init_values : dict = {}, name : str = 'fast') -> Raster :
    """Compute fast PLR and p-values for a set of hypotheses

    Builds a raster object with the same hypothesis points as the one
    provided in input, containing PLR data and p-values computing
    using a linear model.

    Args:
      raster : a raster object
      data   : the input dataset
      name   : name of the output :class:`PLRData` objects
    Returns:
      a raster object containing the fast PLR and p-value results
    """
    fast_plr_data = {}
    for i, hypo in enumerate(hypos) :
      if hypo in init_values :
        init_values[hypo].set_poi_values_and_ranges(self.minimizer)
      fast_plr_data[hypo] = self.compute_fast_q(hypo, data, '%s_%g' % (name, i))
    fast = Raster(name, fast_plr_data)
    self.fill_all_pv(fast)
    return fast

  def recompute_raster(self, raster : Raster, data : Data, name : str = 'fast') -> Raster :
    """Compute fast PLR and p-values for a set of hypotheses

    Builds a raster object with the same hypothesis points as the one
    provided in input, containing PLR data and p-values computing
    using a linear model.

    Args:
      raster : a raster object
      data   : the input dataset
      name   : name of the output :class:`PLRData` objects
    Returns:
      a raster object containing the fast PLR and p-value results
    """
    return self.compute_fast_results(raster.plr_data.keys(), data, { hypo : plr_data.free_fit for hypo, plr_data in raster.plr_data.items() }, name)


class QMuCalculator(TestStatisticCalculator) :
  """Calculator class for :math:`q_{\mu}`

  Implements test statistic and p-value
  computations in :class:`PLRData` objects for
  the :math:`q_{\mu}` test statistic defined in
  the :mod:`test_statistics.py` module.
  """

  def __init__(self, minimizer : POIMinimizer) :
    """Initialize the `QMuCalculator` object

    Args:
      minimizer : a minimizer algorithm.
    """
    super().__init__(minimizer)

  @classmethod
  def make_q(cls, plr_data : PLRData) -> QMu :
    """Builds a :class:`QMu` test statistic object from PLR data

    Args:
      plr_data : an object containing PLR information
    Returns:
      the test statistic
    """
    return QMu(test_poi = plr_data.hypo[cls.poi(plr_data)], tmu = plr_data.test_statistics['tmu'],
               best_poi = plr_data.free_fit.fitpars[cls.poi(plr_data)].value, tmu_Amu = plr_data.test_statistics['tmu_A0'])

  def fill_pv(self, plr_data : PLRData) -> 'QMuCalculator' :
    """Fills p-value information from PLR data

    Builds a :class:`QMu` test statistic from the :class:`PLRData`
    object provided as input, computes the associated test statistic
    value and p-value, and stores these in the :class:`PLRData`
    object.

    Args:
      plr_data : an object storing PLR information
    Returns:
      self
    """
    try :
      # since we use tmu_Amu to compute CLb, we need tmu_Amu = tmu_A0 (computed from an Asimov with mu'=0)
      q = self.make_q(plr_data)
      plr_data.test_statistics['q_mu'] = q.value()
      plr_data.pvs['pv' ] = q.asymptotic_pv()
      plr_data.pvs['cls'] = q.asymptotic_cls()
      plr_data.pvs['clb'] = q.asymptotic_clb()
    except Exception as inst:
      print("q_mu computation failed for PLR '%s', hypothesis %s, with exception below:" % (plr_data.name, plr_data.hypo.dict(pois_only=True)))
      raise(inst)
    return self


class QMuTildaCalculator(TestStatisticCalculator) :
  """Calculator class for :math:`\tilde{q}_{\mu}`

  Implements test statistic and p-value
  computations in :class:`PLRData` objects for
  the :math:`\tilde)q}_{\mu}` test statistic defined
  in the :mod:`test_statistics.py` module.
  """
  def __init__(self, minimizer) :
    """Initialize the `QMuTildaCalculator` object

    Args:
      minimizer : a minimizer algorithm.
    """
    super().__init__(minimizer)
    self.qs = []

  @classmethod
  def make_q(cls, plr_data) :
    """Builds a :class:`QMuTilda` test statistic object from PLR data

    Args:
      plr_data : an object containing PLR information
    Returns:
      the test statistic
    """
    return QMuTilda(test_poi = plr_data.hypo[cls.poi(plr_data)], tmu = plr_data.test_statistics['tmu'],
                    best_poi = plr_data.free_fit.fitpars[cls.poi(plr_data)].value, tmu_Amu = plr_data.test_statistics['tmu_A0'],
                    tmu_A0 = plr_data.test_statistics['tmu_A0'])

  def fill_pv(self, plr_data) :
    """Fills p-value information from PLR data

    Builds a :class:`QMuTilda` test statistic from the :class:`PLRData`
    object provided as input, computes the associated test statistic
    value and p-value, and stores these in the :class:`PLRData`
    object.

    Args:
      plr_data : an object storing PLR information
    Returns:
      self
    """
    try :
      # since we use tmu_Amu to compute CLb, we need tmu_Amu = tmu_A0 (computed from an Asimov with mu'=0)
      q = self.make_q(plr_data)
      plr_data.test_statistics['q~mu'] = q.value()
      plr_data.pvs['pv' ] = q.asymptotic_pv()
      plr_data.pvs['cls'] = q.asymptotic_cls()
      plr_data.pvs['clb'] = q.asymptotic_clb()
    except Exception as inst:
      print("q~mu computation failed for computation '%s', hypothesis %s, with exception below:" % (plr_data.name, plr_data.hypo.dict(pois_only=True)))
      raise(inst)
    return self


class ParBound :
  """Class to define and enforce parameter bounds

  Defines upper and lower bounds on a model parameter,
  and implements a test method applied to :class:`Parameters`
  objects.

  Atttributes:
    par    (str)   : parameter name
    minval (float) : parameter lower bound (`None` if no bound)
    maxval (float) : parameter upper bound (`None` if no bound)
  """

  def __init__(self, par : str, minval : float = None, maxval : float = None) :
    """Initialize the `QMuTildaCalculator` object

    Defines a selection minval <= par <= maxval.
    Both bounds are optional and can be omitted by passing `None` as
    the corresponding argument (also default).

    Args:
      par    : parameter name
      minval : parameter lower bound (`None` for no bound, default)
      maxval : parameter upper bound (`None` for no bound, default)
    """
    self.par = par
    self.minval = minval
    self.maxval = maxval
  def test(self, pars : Parameters) -> bool :
    """Applies the selection to a :class:`Parameters` object

    Args:
      pars : a set of model parameter
    Returns:
     `True` if the parameters pass the selection, `False` if they fail.
    """
    try :
      return (pars[self.par] >= self.minval if self.minval != None else True) and (pars[self.par] <= self.maxval if self.maxval != None else True)
    except KeyError :
      return True
  def __str__(self) -> str :
    """Provides a description string for the object

    Returns:
      a description string
    """
    smin = '%s >= %g' % (self.par, self.minval) if self.minval != None else ''
    smax = '%s <= %g' % (self.par, self.maxval) if self.maxval != None else ''
    if smin == '' : return smax
    if smax == '' : return smin
    return smin + ' and ' + smax
  def __repr__(self) -> str:
    """Provides a description string for the object

    Needed in addition to :meth:`__str__` to print out correctly
    lists of :class:`ParBound` objects.

    Returns:
      a description string
    """
    return self.__str__()


def process_setvals(setvals : str, model : Model, apply : bool = False) -> dict :
  """Parse a set of model POI value assignments

  The input string is expected in the form

  par1=val1,par2=val2,...

  The return value is then the dict

  { "par1" : val1, "par2" : val2, ...}

  The assignments are parsed, and not applied to the model
  parameters (the values of which are not stored in the model)

  Exception are raised if the `parX` are not POIs of the
  model, or if the `valX` are not float values.

  Args:
    setvals : a string specifying parameter assignements
    model   : model containing the parameters

  Returns:
    the parsed POI assignments, as a dict in the form { par_name : par_value }
  """
  par_dict = {}
  try:
    sets = [ a.replace(' ', '').split('=') for a in setvals.split(',') ]
  except Exception as inst :
    print(inst)
    raise ValueError("ERROR : invalid variable assignment specification '%s'." % setvals)
  for (var, val) in sets :
    if not var in model.pois and not var in model.nps : raise ValueError("Parameter '%s' not defined in model." % var)
    try :
      float_val = float(val)
    except ValueError as inst :
      raise ValueError("Invalid numerical value '%s' in assignment to variable '%s'." % (val, var))
    par_dict[var] = float_val
  return par_dict


def process_setranges(setranges : str, model : Model) :
  """Parse a set of POI range assignments

  The input string is expected in the form

  par1:[min1]:[max1],par2:[min2]:[max2],...

  where either the `minX` or the `maxX` values can be omitted (but not the ':' separator!)
  to indicate open ranges. The

  The parameter ranges are applied directly on the model
  parameters

  Exception are raised if the `parX` parameters are not defined
  in the model, or if the `minX` or `maxX` are not float values.

  Args:
    setvals : a string specifying the parameter ranges
    model   : model containing the parameters
  """
  try:
    sets = [ v.replace(' ', '').split(':') for v in setranges.split(',') ]
    for (var, minval, maxval) in sets :
      if not var in model.pois : raise ValueError("Parameter of interest '%s' not defined in model." % var)
      if minval != '' :
        try :
          float_minval = float(minval)
        except ValueError as inst :
          raise ValueError("Invalid numerical value '%s' for the lower bound of variable '%s'." % (minval, var))
        model.pois[var].min_value = float_minval
        print("INFO : setting lower bound of %s to %g" % (var, float_minval))
      if maxval != '' :
        try :
          float_maxval = float(maxval)
        except ValueError as inst :
          raise ValueError("Invalid numerical value '%s' for the upper bound of variable '%s'." % (maxval, var))
        model.pois[var].max_value = float_maxval
        print("INFO : setting upper bound of %s to %g" % (var, float_maxval))
  except Exception as inst :
    print(inst)
    raise ValueError("ERROR : invalid parameter range specification '%s'." % setranges)
