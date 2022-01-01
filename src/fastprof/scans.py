"""
  Utility classes to plot and interpret parameter scans
  The classes are

  * :class:`UpperLimitScan` : 

"""

from abc import abstractmethod
import math
import scipy
import numpy as np

from .core import Model, Data, Parameters
from .fit_data import PLRData, Raster
from .calculators import TestStatisticCalculator


class Scan :
  """Utility class for 1D scans over PLR data

  Attributes:
  """
  def __init__(self, raster : Raster, key : str, calculator : TestStatisticCalculator = None, name : str = '') :
    """Initialize the `Scan` object"""
    self.raster = raster
    self.name = name
    if calculator : calculator.fill_all_pv(raster)
    if not raster.is_filled(key) :
      raise KeyError("No p-value information with key '%s' found in raster '%s'." % (key, raster.name)) 
    self.key = key

  def find_poi(self, poi_name : str, index : int = 0) :
    raster_pois = self.raster.pois()
    if poi_name is not None :
      if poi_name in raster_pois :
        return raster_pois[poi_name]
      else :
        raise KeyError("POI '%s' is not defined in raster '%s'." % (poi_name, raster.name))
    else :
      return raster_pois[list(raster_pois.keys())[index]]

  def value(self, plr_data, with_variation : int = 0) :
    raw_value = plr_data.pvs[self.key] if self.key in plr_data.pvs else plr_data.test_statistics[self.key]
    if not isinstance(raw_value, tuple) or len(raw_value) < 2 :
      if with_variation == 0 :
        return raw_value
      else :
        raise('Cannot return %+g sigma variation on %s since no error information is provided.' % (with_variation, self.key))
    else :
      return raw_value[0] + with_variation*raw_value[1]

  def minimum(self) :
    return min([ (hypo, self.value(plr_data)) for hypo, plr_data in self.raster.plr_data.items() ], key = lambda x : x[1])

  def maximum(self) :
    return max([ (hypo, self.value(plr_data)) for hypo, plr_data in self.raster.plr_data.items() ], key = lambda x : x[1])


class Scan1D (Scan) :
  """Utility class for 1D scans over PLR data

  Attributes:
  """
  def __init__(self, raster : Raster, key : str, poi_name : str = None, calculator : TestStatisticCalculator = None, name : str = '') :
    """Initialize the `Scan1D` object"""
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
      self.key : the key of the selected p-values in :class:`PLRData`
      pv_level : the target p-value
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
      return self.interpolate_crossings(hypos, values, pv_level, order, log_scale)
    hypos, (values_nom, values_up, values_dn) = self.points(with_errors)
    crossings_nom = self.interpolate_crossings(hypos, values_nom, pv_level, order, log_scale)
    crossings_up  = self.interpolate_crossings(hypos, values_up , pv_level, order, log_scale)
    crossings_dn  = self.interpolate_crossings(hypos, values_dn , pv_level, order, log_scale)
    if len(crossings_up) != len(crossings_nom) : raise ValueError('Number of +1sigma crossings (%d) does not match the number of nominal crossings (%d).' % (len(crossings_up), len(crossings_nom)))
    if len(crossings_dn) != len(crossings_nom) : raise ValueError('Number of -1sigma crossings (%d) does not match the number of nominal crossings (%d).' % (len(crossings_dn), len(crossings_nom)))
    return [ (crossing_nom, crossing_up, crossing_dn) for crossing_nom, crossing_up, crossing_dn in zip(crossings_nom, crossings_up, crossings_dn) ]

  def minima(self, order : int = 3) -> float :
    """Compute the minimum value of a test statistic

    Args:
      order : the order of the interpolation (see :meth:`Raster.interpolate_limit`)
      log_scale : if `True`, interpolate in the log of the p-values. If `False`
         (default), interpolate the p-values directly (see :meth:`Raster.interpolate_limit`)

    Returns:
    """
    hypos, values = self.points()
    return self.interpolate_minima(hypos, values, order)
    
  def points(self, with_errors = False) -> tuple :
    """Collect the raster information into a set of points

    Returns:
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
    pts = self.points()
    return scipy.interpolate.InterpolatedUnivariateSpline(pts[0], pts[1], k=order)

  def resample(self, n : int = 100, order : int = 3) :
    pts = self.points()
    grid = np.linspace(pts[0][0], pts[0][-1], n, endpoint=True)
    spl = self.spline(order)
    return (grid, spl(grid))

  def interpolate_crossings(self, xs : list, ys : list, target : float, order : int = 3, log_scale : bool = True) -> list :
    """Perform a one-dimensional interpolation between points to find crossing positions

    Takes 2 lists of same size, corresponding to the `x` and `y`
    dimensions, and interpolates to find the value giving y=target.

    Returns the list of all solutions.
    
    Uses the `InterpolatedUnivariateSpline` method from `scipy`, with
    spline order specified by the `order` parameter. If `log_scale` is
    `True`, the interpolation is performed in the log of the p-values.

    Args:
      x : list of `x` values
      y : list of `y` values
      target : the target value
      order : the order of the interpolation (see :meth:`Raster.interpolate_limit`)
      log_scale : if `True`, interpolate in the log of the p-values. If `False`
         (default), interpolate the p-values directly (see :meth:`Raster.interpolate_limit`)
    Returns:
      The list of interpolated solutions
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
      print('Cannot interpolate using %d point(s) while computing %s, giving up.' % (len(interp_xs), self.name))
      return None
    if len(interp_xs) < order + 1 :
      order = len(interp_xs) - 1
      print('Reducing interpolation order to %d to match the number of available points while computing %s.' % (order, self.name))
    spline = scipy.interpolate.InterpolatedUnivariateSpline(interp_xs, interp_ys, k=order)
    if order == 3 :
      roots = spline.roots()
    else :
      print('Root-finding not supported yet for non-cubic splines, failing in computation %s.' % self.name)
      return []
    return roots

  def interpolate_minima(self, xs : list, ys : list, order : int = 4) -> list :
    """Perform a one-dimensional interpolation between points to find crossing positions

    Takes 2 lists of same size, corresponding to the `x` and `y`
    dimensions, and interpolates to find the value giving y=target.

    Returns the list of all solutions.
    
    Uses the `InterpolatedUnivariateSpline` method from `scipy`, with
    spline order specified by the `order` parameter. If `log_scale` is
    `True`, the interpolation is performed in the log of the p-values.

    Args:
      x : list of `x` values
      y : list of `y` values
      target : the target value
      order : the order of the interpolation (see :meth:`Raster.interpolate_limit`)
      log_scale : if `True`, interpolate in the log of the p-values. If `False`
         (default), interpolate the p-values directly (see :meth:`Raster.interpolate_limit`)
    Returns:
      The list of interpolated solutions
    """
    if len(xs) < 2 :
      print('Cannot interpolate using %d point(s) while computing %s, giving up.' % (len(xs), self.name))
      return None
    if len(xs) < order + 1 :
      order = len(xs) - 1
      print('Reducing interpolation order to %d to match the number of available points while computing %s.' % (order, self.name))
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xs, ys, k=order)
    if order == 4 :
      derivative = spline.derivative()
      roots = derivative.roots()
    else :
      print('Min-finding not supported yet for non-quartic splines, failing in computation %s.' % self.name)
      return []
    return roots



class UpperLimitScan (Scan1D):
  """Utility class to compute upper limits from PLR scan information

  Attributes:
  """

  def __init__(self, raster : Raster, pv_key : str = None, poi_name : str = None, calculator : TestStatisticCalculator = None, name = 'Upper limit', cl = 0.95, cl_name = None) :
    """Initialize the `UpperLimitScan` object"""
    super().__init__(raster, pv_key, poi_name, calculator, name)
    self.cl = 0.95
    self.cl_name = cl_name if cl_name is not None else pv_key

  def limit(self, order : int = 3, log_scale : bool = True, with_errors : bool = False, print_result : bool = False) -> float :
    """Perform a one-dimensional interpolation to compute a limit

    If multiple values are found, the first one is returned. If no
    value is found, returns `None`.

    Args:
      hypos : list of POI values
      pvs   : list of p-values
      order : the order of the interpolation (see :meth:`Raster.interpolate_limit`)
      log_scale : if `True`, interpolate in the log of the p-values. If `False`
         (default), interpolate the p-values directly (see :meth:`Raster.interpolate_limit`)

    Returns:
      The interpolated limit
    """
    found_crossings = self.crossings(1 - self.cl, order, log_scale, with_errors)
    if len(found_crossings) == 0 :
      print("No crossings found for %s = %g vs. %s." % (self.cl_name, self.cl, self.poi.name))
      return None
    if len(found_crossings) > 1 :
      print('Multiple crossings found at the %s = %g level vs. %s, returning the first one' % (self.ts_name, self.ts_level, self.poi.name))
    if print_result : print(self.description(found_crossings[0]))
    return found_crossings[0]

  def description(self, limit) :
    value = limit if not isinstance(limit, tuple) else limit[0]
    value_str = ('%g' % limit) if not isinstance(limit, tuple) else '%g +%g -%g' % (limit[0], limit[1] - limit[0], limit[0] - limit[2])
    return self.name + ' : UL(%g%%) = %s' % (100*self.cl, value_str) \
      + ('  (N = %s)' % str(self.raster.model.n_exp(self.raster.model.expected_pars(value)).sum(axis=1))) if self.raster.model is not None else ''

  def plot(self, plt, marker = 'b', with_errors : bool = False, label : str = None) :
    plt.suptitle('$%s$' % self.cl_name)
    plt.xlabel('%s' % self.poi.name)
    plt.ylabel('$%s$' % self.cl_name)
    pts = self.points(with_errors)
    if with_errors :
      plt.fill_between(pts[0], [ up - nom for (up,nom) in zip(pts[1][1], pts[1][0]) ], [ nom - dn for (nom, dn) in zip(pts[1][0], pts[1][2]) ], facecolor='b', alpha=0.5)
      plt.plot(pts[0], pts[1][0], marker, label=label if label is not None else self.key)
    else :
      plt.plot(pts[0], pts[1], marker, label=label if label is not None else self.key)


class PLRScan1D (Scan1D) :
  """Utility class to compute 1D confidence intervals from PLR scan information

  Attributes:
  """

  def __init__(self, raster : Raster, ts_key : str = None, poi_name : str = None, calculator : TestStatisticCalculator = None, name = 'Profile likelihood', ts_name = None, cl = None, nsigmas = 1) :
    """Initialize the `PLRScan` object"""
    super().__init__(raster, ts_key, poi_name, calculator, name)
    if cl is None and nsigmas is None : raise ValueError('Must provide either a CL value or a number of sigmas to specify the interval size')
    self.ts_level = scipy.stats.chi2.isf(1 - cl, 1) if cl is not None else nsigmas**2
    self.ts_name = ts_name if ts_name is not None else ts_key

  def cl(self) :
    return 1 - scipy.stats.chi2.sf(self.ts_level, 1)


  def interval(self, order : int = 3, log_scale : bool = False, print_result : bool = False) -> float :
    """Perform a one-dimensional interpolation to compute a likelihood interval

    Args:
      hypos : list of POI values
      pvs   : list of p-values
      order : the order of the interpolation (see :meth:`Raster.interpolate_limit`)
      log_scale : if `True`, interpolate in the log of the p-values. If `False`
         (default), interpolate the p-values directly (see :meth:`Raster.interpolate_limit`)

    Returns:
      The interpolated limit
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

  def description(self, minimum, err_hi, err_lo) :
    return '%s = %g' % (self.poi.name, minimum) + ((' +%g' % err_hi) if err_hi is not None else '') + ((' -%g' % err_lo) if err_lo is not None else '') + ' @ %4.1f%% CL' % (100*self.cl())

  def plot(self, plt, linestyle : str = '-', marker = 'b', label : str = None, smooth : int = None) :
    plt.suptitle('$%s$' % self.ts_name)
    plt.xlabel('%s' % self.poi.name)
    plt.ylabel('$%s$' % self.ts_name)
    if smooth is not None :
      rsp = self.resample(smooth)
      plt.plot(rsp[0], rsp[1], marker, linestyle=linestyle, label=label if label is not None else self.key)
    else :
      pts = self.points(with_errors=False)
      plt.plot(pts[0], pts[1], marker, linestyle=linestyle, label=label if label is not None else self.key)


class PLRScan2D (Scan) :
  """Utility class to compute 2D confidence intervals from PLR scan information

  Attributes:
  """

  def __init__(self, raster : Raster, ts_key : str = None, poi1_name : str = None, poi2_name : str = None,
               calculator : TestStatisticCalculator = None, name = 'Profile likelihood', ts_name = None, cl = None, nsigmas = 1) :
    """Initialize the `PLRScan` object"""
    super().__init__(raster, ts_key, calculator, name)
    self.ts_name = ts_name if ts_name is not None else ts_key
    self.poi1 = self.find_poi(poi1_name, 0)
    self.poi2 = self.find_poi(poi2_name, 1)
    if cl is None and nsigmas is None : raise ValueError('Must provide either a CL value or a number of sigmas to specify the contour size')
    cl_level = cl if cl is not None else 1 - 2*scipy.stats.norm.sf(nsigmas)
    self.ts_level = scipy.stats.chi2.isf(1 - cl_level, 2)

  def cl(self) :
    return 1 - scipy.stats.chi2.sf(self.ts_level, 2)

  def points(self) -> tuple :
    """Collect the raster information into a set of points

    Returns:
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
    pts = self.points()
    return scipy.interpolate.SmoothBivariateSpline(pts[0], pts[1], pts[2], kx=order, ky=order)

  def best_fit(self, print_result=False) :
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
      print('best-fit value @ %s=%g, %s=%g' % (self.poi1.name, result.x[0], self.poi2.name, result.x[0]))
    return result.x

  def plot(self, plt, best_fit : bool = False, points : bool = False, color : str = 'g', linestyle : str = 'solid', marker : str = '+', smoothing : int = 0, label : str = None) :
    plt.suptitle('$%s$' % self.ts_name)
    plt.xlabel('%s' % self.poi1.name)
    plt.ylabel('%s' % self.poi2.name)
    #plt.zlabel('$%s$' % self.ts_name)
    if smoothing == 0 :
      pts = self.points()
      cs = plt.tricontour(pts[0], pts[1], pts[2], levels=[self.ts_level], colors=[color], linestyles=[linestyle])
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
      cs = plt.contour(mesh1, mesh2, z, levels=[self.ts_level], colors=[color], linestyles=[linestyle], label=label)
      if label is not None : cs.collections[0].set_label(label)
    if best_fit :
      best1, best2 = self.best_fit()
      plt.scatter(best1, best2, marker=marker, color='k', label='Best fit')
    if points : 
      min1 = pts[0][0]
      min2 = pts[1][0]
      max1 = pts[0][-1]
      max2 = pts[1][-1]
      for x1,x2,z in zip(*pts) :
        if x1 == min1 or x1 == max1 or x2 == min2 or x2 == min1 : continue # remove edge points that overlap with axes
        plt.annotate('%.1f' % z, (x1,x2))

      
    
