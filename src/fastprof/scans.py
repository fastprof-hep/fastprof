"""
  Utility classes to plot parameter scans
  and extract numerical results

  The underlying data in scan classes is a raster of points in POI-space 
  for which confidence level (CL) values or p-values (PV) have been computed.
  This information is stored in a :class:`Raster` object that is stored in the scan class.
  
  The purpose of the scan classes is to either extract meaningful information
  (limits or confidence intervals on the POIs) from the rasters, or plot these
  results.

  The classes are:

  * :class:`Scan` : base class defining the basic interface

  * :class:`Scan1D` : base class for 1-dim scans, deriving from :class:`Scan` and
    defining functions to compute intersections and minima.

  * :class:`UpperLimitScan` : scan class deriving from :class:`Scan1D` to compute upper limits

  * :class:`PLRScan1D` : scan class deriving from :class:`Scan1D` to compute
    confidence intervals on a single parameter

  * :class:`PLRScan2D` : scan class deriving from :class:`Scan` to compute
    2-dim confidence contours on a pair of parameters.
"""

from abc import abstractmethod
import math
import scipy
import numpy as np
import matplotlib.pyplot as plt

from .core import Model, Data, Parameters, ModelPOI
from .fit_data import PLRData, Raster
from .calculators import TestStatisticCalculator


class Scan :
  """Base class for scans over a raster of points

  Defines the basic interface for interacting with a raster:
  identifying the POIs and extracting information at each point
  
  Attributes:
    name (str) : the name of the scan
    raster (Raster) : the raster containing the information
    key (str) : the key identifying the quantity of interest in the raster
                (usually a CL or a PV)
  """
  def __init__(self, raster : Raster, key : str, calculator : TestStatisticCalculator = None, name : str = '') :
    """Initialize the object

      If `calculator` is not None, the calculator is initialized from the raster
      as a side effect of the Scan initialization.

      Args:
        raster : the raster containing the information
        key : the key identifying the quantity of interest in the raster
        calculator : a calculator to be initialized
        name : the scan name (if '', use the raster name)
    """
    self.raster = raster
    self.name = name
    if calculator : calculator.fill_all_pv(raster)
    if not raster.is_filled(key) :
      raise KeyError("No p-value information with key '%s' found in raster '%s'." % (key, raster.name)) 
    self.key = key

  def find_poi(self, poi_name : str, index : int = 0) -> ModelPOI :
    """Identify the POIs

      Return a POI in the raster, identified either from
      its name or its index among the POIs. By default, return
      the first POI. Name-based identification has priority.

      Args:
        poi_name : a POI name
        index : a POI index

      Returns:
        the specified POI
    """
    raster_pois = self.raster.pois
    if poi_name is not None :
      if poi_name in raster_pois :
        return raster_pois[poi_name]
      else :
        raise KeyError("POI '%s' is not defined in raster '%s'." % (poi_name, raster.name))
    else :
      return raster_pois[list(raster_pois.keys())[index]]

  def value(self, plr_data, with_variation : int = 0) :
    """Extract a value from the raster

      Return the target value (specified by self.key) from a PLRData object
      containing information from one point in the raster

      Args:
        plr_data : the object containing the per-point data
        with_variation : if non-zero, return the +n sigma variation
            instead of the central value.

      Returns:
        the specified value
    """
    raw_value = plr_data.pvs[self.key] if self.key in plr_data.pvs else plr_data.test_statistics[self.key]
    if not isinstance(raw_value, tuple) or len(raw_value) < 2 :
      if with_variation == 0 :
        return raw_value
      else :
        raise('Cannot return %+g sigma variation on %s since no error information is provided.' % (with_variation, self.key))
    else :
      return raw_value[0] + with_variation*raw_value[1]

  def minimum(self) :
    """Return the minimum value in the raster

      Returns the mininum value of the target (specified by self.key)
      among all PLRData objects in the raster.
      
      Returns:
        the minimum value
    """
    return min([ (hypo, self.value(plr_data)) for hypo, plr_data in self.raster.plr_data.items() ], key = lambda x : x[1])

  def maximum(self) :
    """Return the maximum value in the raster

      Returns the maximum value of the target (specified by self.key)
      among all PLRData objects in the raster.
      
      Returns:
        the maximum value
    """
    return max([ (hypo, self.value(plr_data)) for hypo, plr_data in self.raster.plr_data.items() ], key = lambda x : x[1])


class Scan1D (Scan) :
  """Base class for 1D scans over a raster of points

  Defines the basic interface for dealing with 1D curves,
  in particular computing extrema and crossings with 
  specified levels.

  Attributes:
    name (str) : the name of the scan
    raster (Raster) : the raster containing the information
    key (str) : the key identifying the quantity of interest in the raster
                (usually a CL or a PV)
    poi (str) : the POI object
  """

  def __init__(self, raster : Raster, key : str, poi_name : str = None, calculator : TestStatisticCalculator = None, name : str = '') :
    """Initialize the object

      Calls :meth:`Scan.__init__` and identifies the one POI.

      Args:
        raster : the raster containing the information
        key : the key identifying the quantity of interest in the raster
        poi_name : the name of the POI
        calculator : a calculator to be initialized
        name : the scan name (if '', use the raster name)
    """
    super().__init__(raster, key, calculator, name)
    self.poi = self.find_poi(poi_name)

  def crossings(self, pv_level : float = 0.05, order : int = 3, log_scale : bool = True, with_errors : bool = False) -> float :
    """Compute the crossing points at a predefined p-value level

    The contour is obtained by interpolating the p-values at each raster
    point to identify the points in the space of hypothesis values
    where the p-value corresponds to the target p-value.

    The set of interpolated p-values is made up of elements from the
    `self.pvs` dict of each :class:`PLRData` object, at the key
    position provided by the argument `self.key`.

    The current implementation works in 1D only, with defaults designed
    to compute a :math:`CL_s` upper limit at 95% CL. The interpolation is
    performed by the :meth:`Raster.interpolate_limit` method.

    Args:
      pv_level : the target p-value for the crossings
      order : the order of the interpolation (see :meth:`Raster.interpolate_limit`)
      log_scale : if `True`, interpolate in the log of the p-values. If `False`
         (default), interpolate the p-values directly (see :meth:`Raster.interpolate_limit`)
      with_errors : if `True`, returns for each crossing a triplet with (nominal , nominal + 1sigma error,
         nominal - 1sigma error), where the uncertainties are propagated from those of the p-values.

    Returns:
      A list of crossings : if `with_errors` is False, a list of floats giving the crossing positions, and
                            if `with_errors` is True, a list of (nominal , nominal + 1sigma error,
                            nominal - 1sigma error) values
    """
    if not with_errors :
      hypos, values = self.points(with_errors)       
      return self.interpolate_crossings(hypos, values, pv_level, order, log_scale, self.name)
    hypos, (values_nom, values_up, values_dn) = self.points(with_errors)
    crossings_nom = self.interpolate_crossings(hypos, values_nom, pv_level, order, log_scale, self.name)
    crossings_up  = self.interpolate_crossings(hypos, values_up , pv_level, order, log_scale, self.name)
    crossings_dn  = self.interpolate_crossings(hypos, values_dn , pv_level, order, log_scale, self.name)
    if len(crossings_up) != len(crossings_nom) : raise ValueError('Number of +1sigma crossings (%d) does not match the number of nominal crossings (%d).' % (len(crossings_up), len(crossings_nom)))
    if len(crossings_dn) != len(crossings_nom) : raise ValueError('Number of -1sigma crossings (%d) does not match the number of nominal crossings (%d).' % (len(crossings_dn), len(crossings_nom)))
    return [ (crossing_nom, crossing_up, crossing_dn) for crossing_nom, crossing_up, crossing_dn in zip(crossings_nom, crossings_up, crossings_dn) ]

  def minima(self, order : int = 3) -> float :
    """Compute the minimum value of a test statistic

    Args:
      order : the order of the interpolation (see :meth:`Raster.interpolate_limit`)

    Returns:
      the list of minima
    """
    hypos, values = self.points()
    found_minima = self.interpolate_minima(hypos, values, order, self.name)
    return found_minima if len(found_minima) > 0 else np.array([ values[np.argmin(hypos)] ])
    
  def points(self, with_errors : bool = False) -> tuple :
    """Collect the raster information into a set of points

    The output is a pair of lists, the first for the poi values (X-axis)
    and the second for the result values (Y-axis). If `with_errors` is True,
    the second list is triplet of lists for (central, +1sigma, -1sigma) values.

    Args:
      with_errors : if `True`, return y-values with an error band

    Returns:
      a tuple of lists as specified above
    """
    poi_values = []
    result_values = []
    if with_errors :
      result_values_up = []
      result_values_dn = []
    for hypo, plr_data in self.raster.plr_data.items() :
      poi_values.append(hypo[self.poi.name])
      if not with_errors :
        result_values.append(self.value(plr_data))
      else :
        result_values.append(self.value(plr_data))
        result_values_up.append(self.value(plr_data, +1))
        result_values_dn.append(self.value(plr_data, -1))
    return (poi_values, (result_values, result_values_up, result_values_dn)) if with_errors else (poi_values, result_values)

  def spline(self, order : int = 3) :
    """Compute a spline over the raster points

    The spline is computed from the points returned
    by :meth:`Scan1D.points` above.

    Args:
      order : the spline order

    Returns:
      the spline curve
    """
    pts = self.points()
    return scipy.interpolate.InterpolatedUnivariateSpline(pts[0], pts[1], k=order)

  def resample(self, n : int = 100, order : int = 3) :
    """Resample the raster points over a finer grid

    First interpolates the existing points using the spline
    obtained from :meth:`Scan1D.spline` above at the specified
    order, and uses this to compute result values over
    a finer grid of points covering the same range as the
    raster.
    
    The return value is a pair of lists, the first for the poi values (X-axis)
    and the second for the result values (Y-axis)

    Args:
      n : the number of points to use over the resampling
      order : the spline order

    Returns:
      pair of lists giving the poi and result values for the new sampling
    """
    pts = self.points()
    grid = np.linspace(pts[0][0], pts[0][-1], n, endpoint=True)
    spl = self.spline(order)
    return (grid, spl(grid))

  @classmethod
  def interpolate_crossings(cls, xs : list, ys : list, target : float, order : int = 3, log_scale : bool = True, name : str = '') -> list :
    """Perform a one-dimensional interpolation between points to find crossing positions

    Takes 2 lists of same size, corresponding to the `x` and `y`
    dimensions, and interpolates to find the value giving y=target.

    Returns the list of all solutions.
    
    Uses the `InterpolatedUnivariateSpline` method from `scipy`, with
    spline order specified by the `order` parameter. If `log_scale` is
    `True`, the interpolation is performed in the log of the p-values.

    Args:
      xs : list of `x` values for the input points
      ys : list of `y` values for the input points
      target : the target value for the crossings
      order : the order of the interpolation (see :meth:`Raster.interpolate_limit`)
      log_scale : if `True`, interpolate in the log of the p-values. If `False`
         (default), interpolate the p-values directly (see :meth:`Raster.interpolate_limit`)
      name : a name for the task, to use in error reporting
    Returns:
      list of crossings
    """
    interp_xs = []
    interp_ys = []
    for x, y in zip(xs, ys) :
      if log_scale :
        if y <= 0 : continue
        value = math.log(y/target)
      else :
        value = y - target
      interp_ys.append(value)
      interp_xs.append(x)
    if len(interp_xs) < 2 :
      print('Cannot interpolate using %d point(s) while computing %s, giving up.' % (len(interp_xs), name))
      return None
    if len(interp_xs) < order + 1 :
      order = len(interp_xs) - 1
      print('Reducing interpolation order to %d to match the number of available points while computing %s.' % (order, name))
    spline = scipy.interpolate.InterpolatedUnivariateSpline(interp_xs, interp_ys, k=order)
    if order == 3 :
      roots = spline.roots()
    else :
      print('Root-finding not supported yet for non-cubic splines, failing in computation %s.' % name)
      return []
    return roots

  @classmethod
  def interpolate_minima(cls, xs : list, ys : list, order : int = 4, name : str = '') -> list :
    """Perform a one-dimensional interpolation between points to find crossing positions

    Takes 2 lists of same size, corresponding to the `x` and `y`
    dimensions, and interpolates to find the position of the 
    (interpolated) minimum y.
    
    Uses the `InterpolatedUnivariateSpline` method from `scipy`, with
    spline order specified by the `order` parameter, and computes the
    derivative spline to find the minima.
    
    Note that for now only quartic splines are supported.

    Args:
      xs : list of `x` values
      ys : list of `y` values
      target : the target value
      order : the order of the interpolation (see :meth:`Raster.interpolate_limit`)
      name : a name for the task, to use in error reporting
    Returns:
      list of minima
    """
    if len(xs) < 2 :
      print('Cannot interpolate using %d point(s) while computing %s, giving up.' % (len(xs), name))
      return None
    if len(xs) < order + 1 :
      order = len(xs) - 1
      print('Reducing interpolation order to %d to match the number of available points while computing %s.' % (order, name))
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xs, ys, k=order)
    if order == 4 :
      derivative = spline.derivative()
      roots = derivative.roots()
    else :
      print('Min-finding not supported yet for non-quartic splines, failing in computation %s.' % name)
      return []
    return roots



class UpperLimitScan (Scan1D):
  """Base class for Upper limit scans over a raster of points

  Defines the basic interface for dealing with 1D curves,
  in particular computing extrema and crossings with 
  specified levels.

  Attributes:
    name (str) : the name of the scan
    raster (Raster) : the raster containing the information
    key (str) : the key identifying the quantity of interest in the raster
                (usually a CL or a PV)
    poi (str) : the POI object
    cl_name (str) : name of the CL value to use.
    cl (float) : the CL at which to compute the limit (default: 0.95)
  """

  def __init__(self, raster : Raster, pv_key : str, poi_name : str = None,
               calculator : TestStatisticCalculator = None, name = 'Upper limit', cl = 0.95, cl_name = None) :
    """Initialize the object

      Args:
        raster : the raster containing the information
        key : the key identifying the quantity of interest in the raster
        poi_name : the name of the POI
        calculator : a calculator to be initialized
        name : the scan name (if '', use the raster name)
        cl_name : name of the CL value to use.
        cl : the CL at which to compute the limit (default: 0.95)
    """
    super().__init__(raster, pv_key, poi_name, calculator, name)
    self.cl = 0.95
    self.cl_name = cl_name if cl_name is not None else pv_key

  def limit(self, order : int = 3, log_scale : bool = True, with_errors : bool = False, print_result : bool = False) -> float :
    """Perform a one-dimensional interpolation to compute a limit

    The limit is computed as the crossing point with the specified
    CL value. If multiple values are found, the first one is returned.
    If no value is found, returns `None`.
    
    Args:
      order : the order of the interpolation (see :meth:`Raster.interpolate_limit`)
      log_scale : if `True`, interpolate in the log of the p-values. If `False`
         (default), interpolate the p-values directly (see :meth:`Raster.interpolate_limit`)
      with_errors : if `True`, also returns the crossing points with the +/-1sigma bands.
      print_result : if `True`, print out the results.

    Returns:
      the interpolated limit (+ optional bands) 
    """
    found_crossings = self.crossings(1 - self.cl, order, log_scale, with_errors)
    if len(found_crossings) == 0 :
      print("No crossings found for %s = %g vs. %s." % (self.cl_name, self.cl, self.poi.name))
      return (None, None, None)
    if len(found_crossings) > 1 :
      print('Multiple crossings found at the %s = %g level vs. %s, returning the first one' % (self.cl_name, self.cl, self.poi.name))
    if print_result : print(self.description(found_crossings[0]))
    return found_crossings[0]

  def description(self, limit : float) -> str :
    """Build a description string for printout
    
    Args:
      limit : the limit value for which to print out the description

    Returns:
      the string description 
    """
    value = limit if not isinstance(limit, tuple) else limit[0]
    value_str = ('%g' % limit) if not isinstance(limit, tuple) else '%g +%g -%g' % (limit[0], limit[1] - limit[0], limit[0] - limit[2])
    return self.name + ' : UL(%g%%) = %s' % (100*self.cl, value_str) \
      + ('  (N = %s)' % str(self.raster.model.n_exp(self.raster.model.expected_pars(value)).sum(axis=1))) if self.raster.model is not None else ''

  def plot(self, canvas : tuple = (None, None), marker : str = 'b', with_errors : bool = False, label : str = None) :
    """Plot the CL curve and the intersection with the target CL
    
    Args:
      canvas : a (fig, axes) pair on which to plot the result. If not specified, a new
               figure is created.
      marker : the marker type to use.
      with_errors : if `True`, also returns the crossing points with the +/-1sigma bands.
      label : the curve label to use for the legend.
    """
    if canvas == (None, None) : 
      fig, axs = plt.subplots(figsize=figsize, dpi=100, constrained_layout=True)
    elif isinstance(canvas, plt.Figure) :
      fig = canvas
      axs = fig.axes[0]
    else :
      fig, axs = canvas
    fig.suptitle('$%s$' % self.cl_name)
    axs.set_xlabel('%s' % self.poi.name)
    axs.set_ylabel('$%s$' % self.cl_name)
    pts = self.points(with_errors)
    if with_errors :
      axs.fill_between(pts[0], [ up - nom for (up,nom) in zip(pts[1][1], pts[1][0]) ], [ nom - dn for (nom, dn) in zip(pts[1][0], pts[1][2]) ], facecolor='b', alpha=0.5)
      axs.plot(pts[0], pts[1][0], marker, label=label if label is not None else self.key)
    else :
      axs.plot(pts[0], pts[1], marker, label=label if label is not None else self.key)


class PLRScan1D (Scan1D) :
  """Base class for 1D PLR scans over a raster of points

  Defines the basic interface for dealing with 1D curves,
  in particular computing extrema and crossings with 
  specified levels.

  Attributes:
    name (str) : the name of the scan
    raster (Raster) : the raster containing the information
    key (str) : the key identifying the quantity of interest in the raster
                (usually a CL or a PV)
    poi (str) : the POI object
    ts_name (str) : the name of the test statistic
    ts_level (float) : the test statistic level at which to report the
       confidence interval.
  """

  def __init__(self, raster : Raster, ts_key : str = None, poi_name : str = None,
               calculator : TestStatisticCalculator = None, name = 'Profile likelihood',
               ts_name = None, cl = None, nsigmas = 1) :
    """Initialize the object

      The level at which the compute the confidence interval
      can be provided either as a test statistic value (`ts_level`)
      or a number of sigmas (`nsigmas`).

      Args:
        raster : the raster containing the information
        key : the key identifying the quantity of interest in the raster
        poi_name : the name of the POI
        calculator : a calculator to be initialized
        name : the scan name (if '', use the raster name)
        ts_name : the name of the test statistic
        ts_level : the test statistic level at which to report the
                   confidence interval.
        nsigmas : test statistic level, provided as number of sigmas
    """
    super().__init__(raster, ts_key, poi_name, calculator, name)
    if cl is None and nsigmas is None : raise ValueError('Must provide either a CL value or a number of sigmas to specify the interval size')
    self.ts_level = scipy.stats.chi2.isf(1 - cl, 1) if cl is not None else nsigmas**2
    self.ts_name = ts_name if ts_name is not None else ts_key

  def cl(self) -> float :
    """Return the CL value for the interval
    
    Returns:
      the CL value
    """
    return 1 - scipy.stats.chi2.sf(self.ts_level, 1)


  def interval(self, order : int = 3, log_scale : bool = False, print_result : bool = False) -> float :
    """Perform a one-dimensional interpolation to compute a likelihood interval

    Args:
      order : the order of the interpolation (see :meth:`Raster.interpolate_limit`)
      log_scale : if `True`, interpolate in the log of the p-values. If `False`
         (default), interpolate the p-values directly (see :meth:`Raster.interpolate_limit`)
      print_result : if `True`, print out the results.

    Returns:
      the interval as a (central value, +error, -error) triplet
    """
    found_crossings = self.crossings(self.ts_level, order, log_scale, with_errors=False)
    if len(found_crossings) == 0 :
      print("No crossings found for %s = %g vs. %s." % (self.ts_name, self.ts_level, self.poi.name))
      found_crossings = np.array([ None, None])
    if len(found_crossings) == 1 :
      print('Only one %s = %g crossing found vs. %s.' % (self.ts_name, self.ts_level, self.poi.name))
      found_crossings = np.array([ found_crossings[0], None])
    if len(found_crossings) > 2 :
      print('More than 2 crossings found at the %s = %g level vs. %s, returning the first two' % (self.ts_name, self.ts_level, self.poi.name))
    value_lo = found_crossings[0]
    value_hi = found_crossings[1]

    found_minima = self.minima(order+1)
    if len(found_minima) == 0 :
      print("No minima found for %s vs. %s." % (self.ts_name, self.poi.name))
      return None
    if len(found_minima) > 1 :
      spl = self.spline(order)
      minimum = min([ (found_minimum, float(spl(found_minimum))) for found_minimum in found_minima ], key=lambda x: x[1])[0]
    else :
      minimum = found_minima[0]
    if value_hi is None and value_lo is not None and value_lo > minimum :
      value_hi = value_lo
      value_lo = None
    error_hi = value_hi - minimum if value_hi is not None else None
    error_lo = minimum - value_lo if value_lo is not None else None
    if print_result : print(self.description(minimum, error_hi, error_lo))
    return minimum, error_hi, error_lo

  def description(self, central_value : float, err_hi : float, err_lo : float) :
    """Build a description string for printout
    
    Args:
      central_value : the central value of the interval
      err_hi : the positive uncertainty
      err_lo : the negative uncertainty

    Returns:
      the string description 
    """
    return '%s = %g' % (self.poi.name, central_value) + ((' +%g' % err_hi) if err_hi is not None else '') + ((' -%g' % err_lo) if err_lo is not None else '') + ' @ %4.1f%% CL' % (100*self.cl())

  def plot(self, canvas : tuple = (None, None), linestyle : str = '-', marker = 'b', label : str = None, smooth : int = None) :
    """Plot the CL curve and the intersection with the target CL
    
    Args:
      canvas : a (fig, axes) pair on which to plot the result. If not specified, a new
               figure is created.
      linestyle : the line style to use.
      marker : the marker type to use.
      label : the curve label to use for the legend.
      smooth : if not `None`, resample the specified number of points
               to get a smoother curve
    """
    if canvas == (None, None) : 
      fig, axs = plt.subplots(figsize=figsize, dpi=100, constrained_layout=True)
    elif isinstance(canvas, plt.Figure) :
      fig = canvas
      axs = fig.axes[0]
    else :
      fig, axs = canvas
    fig.suptitle('$%s$' % self.ts_name)
    axs.set_xlabel('%s' % self.poi.name)
    axs.set_ylabel('$%s$' % self.ts_name)
    if smooth is not None :
      rsp = self.resample(smooth)
      axs.plot(rsp[0], rsp[1], marker, linestyle=linestyle, label=label if label is not None else self.key)
    else :
      pts = self.points(with_errors=False)
      axs.plot(pts[0], pts[1], marker, linestyle=linestyle, label=label if label is not None else self.key)


class PLRScan2D (Scan) :
  """Base class for 2D PLR scans over a raster of points

  Defines the basic interface for dealing with 1D curves,
  in particular computing extrema and crossings with 
  specified levels.

  Attributes:
    name (str) : the name of the scan
    raster (Raster) : the raster containing the information
    key (str) : the key identifying the quantity of interest in the raster
                (usually a CL or a PV)
    poi1 (str) : the first POI object
    poi2 (str) : the second POI object
    ts_name (str) : the name of the test statistic
    ts_level (float) : the test statistic level at which to report the
       confidence interval.
  """

  def __init__(self, raster : Raster, ts_key : str = None, poi1_name : str = None, poi2_name : str = None,
               calculator : TestStatisticCalculator = None, name = 'Profile likelihood', ts_name = None, cl = None, nsigmas = 1) :
    """Initialize the object

      The level at which the compute the confidence interval
      can be provided either as a test statistic value (`ts_level`)
      or a number of sigmas (`nsigmas`).

      Args:
        raster : the raster containing the information
        key : the key identifying the quantity of interest in the raster
        poi_name : the name of the POI
        calculator : a calculator to be initialized
        name : the scan name (if '', use the raster name)
        ts_name : the name of the test statistic
        ts_level : the test statistic level at which to report the
                   confidence interval.
        nsigmas : test statistic level, provided as number of sigmas
    """
    super().__init__(raster, ts_key, calculator, name)
    self.ts_name = ts_name if ts_name is not None else ts_key
    self.poi1 = self.find_poi(poi1_name, 0)
    self.poi2 = self.find_poi(poi2_name, 1)
    if cl is None and nsigmas is None : raise ValueError('Must provide either a CL value or a number of sigmas to specify the contour size')
    cl_level = cl if cl is not None else 1 - 2*scipy.stats.norm.sf(nsigmas)
    self.ts_level = scipy.stats.chi2.isf(1 - cl_level, 2)

  def cl(self) :
    """Return the CL value for the contour
    
    Returns:
      the CL value
    """
    return 1 - scipy.stats.chi2.sf(self.ts_level, 2)

  def points(self) -> tuple :
    """Collect the raster information into a set of points

    Returns a triplet of lists containing the
    POI1 values (X), POI2 values (Y) and result values (Z)
    in this order.

    Returns:
      triplet of point coordinates
    """
    poi1_values = []
    poi2_values = []
    result_values = []
    for hypo, plr_data in self.raster.plr_data.items() :
      poi1_values.append(hypo[self.poi1.name])
      poi2_values.append(hypo[self.poi2.name])
      result_values.append(self.value(plr_data))
    return (poi1_values, poi2_values, result_values)

  def spline(self, order : int = 3) :
    """Compute a spline over the raster points

    The spline is computed from the points returned
    by :meth:`Scan1D.points` above.

    Args:
      order : the spline order

    Returns:
      the spline curve
    """
    pts = self.points()
    return scipy.interpolate.SmoothBivariateSpline(pts[0], pts[1], pts[2], kx=order, ky=order)

  def best_fit(self, print_result=False) :
    """Compute the best-fit point

    Args:
      print_result : if `True`, print out the result

    Returns:
      the best-fit position as a list of [x,y] coordinates.
    """
    spl = self.spline()
    best_point = self.minimum()
    init1 = best_point[0][self.poi1.name]
    init2 = best_point[0][self.poi2.name]
    result = scipy.optimize.minimize(lambda pois : spl(pois[0], pois[1])[0][0], x0=[init1, init2], method='L-BFGS-B', options={'gtol': 1e-5, 'ftol':1e-5 })
    if not result.success :
      print('Minimization failed while computing best-fit value in 2D scan starting from %s=%g, %s=%g returning these values.' % (self.poi1.name, init1, self.poi2.name, init2))
      print(result.message)
      return init1, init2
    if print_result :
      print('best-fit value @ %s=%g, %s=%g' % (self.poi1.name, result.x[0], self.poi2.name, result.x[1]))
    return result.x

  def plot(self, canvas : tuple = (None, None), best_fit : bool = False, points : bool = False,
           color : str = 'g', linestyle : str = 'solid', marker : str = '+',
           smoothing : int = None, label : str = None) :
    """Plot the CL curve and the intersection with the target CL
    
    Args:
      canvas : a (fig, axes) pair on which to plot the result. If not specified, a new
               figure is created.
      best_fit : if `True`, plot the best-fit point
      points : if `True`, plot the raster points
      color : the line color to use
      linestyle : the line style to use.
      marker : the marker type to use.
      smooth : if not `None`, resample the specified number of points
               to get a smoother curve
      label : the curve label to use for the legend.
    """
    if canvas == (None, None) : 
      fig, axs = plt.subplots(figsize=figsize, dpi=100, constrained_layout=True)
    elif isinstance(canvas, plt.Figure) :
      fig = canvas
      axs = fig.axes[0]
    else :
      fig, axs = canvas
    fig.suptitle('$%s$' % self.ts_name)
    axs.set_xlabel('%s' % self.poi1.name)
    axs.set_ylabel('%s' % self.poi2.name)
    #axs.set_zlabel('$%s$' % self.ts_name)
    if smoothing is None :
      pts = self.points()
      cs = axs.tricontour(pts[0], pts[1], pts[2], levels=[self.ts_level], colors=[color], linestyles=[linestyle])
      if label is not None : cs.collections[0].set_label(label)
    else :
      pts = self.points()
      min1 = pts[0][0]
      min2 = pts[1][0]
      max1 = pts[0][-1]
      max2 = pts[1][-1]
      spl = self.spline()
      x1 = np.linspace(min1, max1, smoothing)
      x2 = np.linspace(min2, max2, smoothing)
      mesh1, mesh2 = np.meshgrid(x1, x2)
      z = spl(mesh1, mesh2, grid=False)
      cs = axs.contour(mesh1, mesh2, z, levels=[self.ts_level], colors=[color], linestyles=[linestyle], label=label)
      if label is not None : cs.collections[0].set_label(label)
    if best_fit :
      best1, best2 = self.best_fit()
      axs.scatter(best1, best2, marker=marker, color='k', label='Best fit')
    if points : 
      min1 = pts[0][0]
      min2 = pts[1][0]
      max1 = pts[0][-1]
      max2 = pts[1][-1]
      for x1,x2,z in zip(*pts) :
        if x1 == min1 or x1 == max1 or x2 == min2 or x2 == min1 : continue # remove edge points that overlap with axes
        axs.annotate('%.1f' % z, (x1,x2))

      
    
