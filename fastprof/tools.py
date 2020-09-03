import json
import math
import scipy
import numpy as np
import copy

from .core import Model, Data, Parameters, JSONSerializable
from .minimizers import OptiMinimizer
from .test_statistics import QMu, QMuTilda


class FitPar(JSONSerializable) :
  def __init__(self, name = '', value, error = None, min_value = None, max_value = None, init_value = None) :
    self.name = name
    self.value = value
    self.error = error
    self.min_value = min_value
    self.max_value = max_value
    self.init_value = init_value
  def __str__(self) :
    s = "Fit parameter '%s' : %g +/- %g (min = %g, max = %g, init = %g)" % (self.name, self.value, self.error, self.min_value, self.max_value, self.init_value)
    return s
  def load_jdict(self, jdict) : 
    self.name       = self.load_field('name'      , jdict, '', str)
    self.value      = self.load_field('value'     , jdict, '', [int, float])
    self.error      = self.load_field('error'     , jdict, '', [int, float])
    self.min_value  = self.load_field('min_value' , jdict, '', [int, float])
    self.max_value  = self.load_field('max_value' , jdict, '', [int, float])
    self.init_value = self.load_field('init_value', jdict, '', [int, float])
    return self
  def fill_jdict(self, jdict) :
    jdict['name']       = self.name
    jdict['value']      = self.value
    jdict['error']      = self.error
    jdict['min_value']  = self.min_value
    jdict['max_value']  = self.max_value
    jdict['init_value'] = self.init_value


class FitResult(JSONSerializable) :
  def __init__(self, name = '', fitpars = {}, nll = None) :
    super().__init__()
    self.name = name
    self.fitpars = fitpars
    self.nll = nll
  def pars(self, model = None) :
    return Parameters(model).set_from_dict({ par.name : par.value for par in self.fitpars })
  def load_jdict(self, jdict) :
    for par_dict in jdict['pars'] :
      fitpar = FitPar().load_jdict(par_dict)
      self.fitpars[fitpar.name] = fitpar
    self.nll = jdict['nll']
    return self
  def save_jdict(self, jdict) :
    jdict['pars'] = { par.name : par.dump_jdict() for par in self.fitpars }
    jdict['nll'] = self.nll
  def __str__(self) :
    s =  "  Fit '%s' : nll = %g, pars : %s" % (self.name, self.nll, str(self.pars))


class PLRData(JSONSerializable) :
  def __init__(self, name = '', hypo = None, free_fit = None, hypo_fit = None, test_statistics = {}, pvs = {}, asimov = None) :
    super().__init__()
    self.name = ''
    self.hypo = hypo
    self.free_fit = free_fit
    self.hypo_fit = hypo_fit
    self.test_statistics = test_statistics
    self.pvs = pvs
    self.asimov = asimov
    if not 'tmu' in self.test_statistics : self.compute_tmu()
  def poi_names(self) :
    return list(self.hypo.keys())
  def hypo_pars(self, model = None) :
    return Parameters(model).set_from_dict(self.hypo)
  def compute_tmu(self) :
    self.test_statistics[key] = -2*(hypo_fit.nll - free_fit.nll)
  def set_asimov(asimov_plr_data, local_key = 'tmu_0') :
    self.test_statistics[local_key] = asimov_plr_data.test_statistics['tmu']
    self.asimov = asimov_plr_data
  def load_jdict(self, jdict) :
    self.free_fit = FitResult('free_fit').load_jdict(jdict['free_fit'])
    self.hypo_fit = FitResult('hypo_fit').load_jdict(jdict['hypo_fit'])
    self.test_statistics = jdict['test_statistics'] if 'test_statistics' in jdict else {}
    self.pvs = jdict['pvs'] if 'pvs' in jdict else {}
    if not 'tmu' in self.test_statistics : self.compute_tmu()
    return self
  def save_jdict(self, jdict) :
    jdict['free_fit'] = self.free_fit.dump_jdict()
    jdict['hypo_fit'] = self.hypo_fit.dump_jdict()
    jdict['test_statistics'] = self.test_statistics
    jdict['pvs'] = self.pvs
  def __str__(self) :
    s = 'Profile-likelihood ratio data for hypothesis:' + str(self.hypo)
    s += '\n  test statistics : %s' % str(self.test_statistics)
    s += '\n  p-values : %s' % str(self.pvs)
    s += '\n  Unconditional fit:' + str(self.free_fit)
    s += '\n  Conditional fit:' + str(self.hypo_fit)


class ScanData(JSONSerializable) :
  def __init__(self, name, plr_data = {}, filename = '', use_global_best_fit = True) :
    super().__init__()
    self.name = name
    self.plr_data = plr_data
    self.use_global_best_fit = use_global_best_fit
    if filename != '' : self.load_with_asimov(filename, 'asimov')
    if self.use_global_best_fit : self.set_global_best_fit()

  def poi_names(self) :
    return list(plr_data.values())[0].poi_names() if len(self.plr_data) > 0 else None

  def set_global_best_fit(self) :
    nlls = { hypo : plr_data.free_fit.nll for hypo, plr_data in self.plr_data.items()
    best_hypo = min(nlls, key=nlls.get)
    for plr_data in self.plr_data :
      plr_data.best_fit = self.plr_data[best_hypo].free_fit

  def set_asimov(asimov_scan_data, asimov_key = 'tmu', local_key = 'tmu_0') :
    for plr_data in self.plr_data.keys() :
      if not plr_data.hypo in asimov_scan_data.plr_data :
        raise KeyError("Hypo %s not found in Asimov scan data '%s'." % (str(plr_data.hypo), asimov_scan_data.name))
      plr_data.set_asimov(asimov_scan_data[plr_data.hypo], asimov_key, local_key)

  def compute_tmu(self) :
    for plr_data in self.plr_data : plr_data.compute_tmu()

  def compute_limit(self, pv_key = 'cls', cl = 0.05, order = 3, log_scale = True) :
    if len(self.poi_names()) > 1 : raise ValueError('Cannot interpolate limit in more than 1 dimension.')
    poi = self.poi_names()[0]
    hypos = [ hypo[poi] for hypo in self.plr_data.keys() ]
    values = []
    for hypo, plr_data in self.plr_data.items() :
      if not pv_key in plr_data.pvs : raise KeyError("P-value '%s' not found at hypo %s." % (pv_key, str(hypo))
      if log_scale :
        value = math.log(plr_data.pvs[pv_key]/cl) if plr_data.pvs[pv_key] > 0 else -999
      else :
        value = plr_data.pvs[pv_key] - cl for result in self.fit_results ]
      values.append(value)
    finder = scipy.interpolate.InterpolatedUnivariateSpline(hypos, values, k=order)
    roots = finder.roots() 
    if len(roots) == 0 :
      print("No solution found for %s[%s] = %g. Interpolation set:" % (pv_key, poi, cl))
      print([a for a in zip(hypos, values)])
      return None
    if len(roots) > 1 :
      print('Multiple solutions found for %s[%s] = %g, returning the first one' % (pv_key, poi, cl))
    return roots[0]

  def load_jdict(self, jdict) :
    for plr_dict in jdict[self.name] :
      plr_data = PLRData().load_jdict(plr_dict)
      self.plr_data[plr_data.hypo] = plr_data
    if self.use_global_best_fit : self.set_global_best_fit()
    return self

  def fill_jdict(self, jdict) :
    jdict['points'] = []
    for plr_data in self.plr_data.values() :
      jdict['points'].append(plr_data.dump_jdict())

  def load_with_asimov(self, filename, asimov_key = 'asimov') :
    self.load(filename)
    ScanData(asimov_key).load(filename)
    self.set_asimov(asimov)
    return self

  def __str__(self) :
    s = ''
    s += 'POIs : ' + str(self.poi_names()) + '\n'
    s += 'PLR data : '
    for hypo, plr_data in self.plr_data.items() :
      s += '\nHypo :' + str(hypo)
      s += '\n' + str(pv_result)
    return s

  def key_value(self, key, hypo) :
    if not hypo in self.results : raise KeyError('While trying to access key %s in result %s, hypo %s was not found' % (key, name, str(hypo)))
    for poi in self.poi_names() :
      if key == poi : return hypo[poi]
      if key == 'best_' + poi : return self.plr_data[hypo].free_fit.pars[poi]
    if key in self.plr_data[hypo].pvs : return self.plr_data[hypo].pvs[key]
    if key in self.plr_data[hypo].test_statistics : return self.plr_data[hypo].test_statistics[key]
    raise KeyError('No data found for key %s in result %s of hypo %s' % (key, name, str(hypo)))
  
  # TODO : update this
  def print(self, print_keys = [], verbosity = 0, print_limits=True) :
    if len(self.fit_results) == 0 : return ''
    if print_keys == [] :
      print_keys = self.poi_names() + [ 'pv' ]
      if verbosity > 0 :
        print_keys.extend([ 'cls', 'clb' ])
      if verbosity > 1 :
        print_keys.extend([ 'tmu' ] + [ 'best_' + k for k in self.poi_names() ]
    s = ''
    for k in print_keys : s += '| %-15s ' % k
    for hypo, results in self.results.items() : 
      for key in print_keys :
        if key in self.poi_names() : 
          s += '| %-15g ' % self.key_value(key, hypo)
        for name in results :
          for key in print_keys :
            s += '| %-15g ' % self.key_value(key, hypo, name)
    if print_limits and len(self.poi_names()) == 1 :
      for result_name in self.result_names() :
        limit = self.solve(result_name, 'cls', 0.05)
        limit_str = '%g' % limit if limit != None else 'not computable'
        s += '\n' + "Asymptotic 95%% CLs limit for '%s' computation = %s" % (result_name, limit_str)
    return s


class TestStatisticCalculator :
  def __init__(self, minimizer) :
    self.minimizer = minimizer

  def poi_name(self, plr_data) :
    if len(plr_data.poi_names()) != 1 : raise ValueError('Cannot only compute upper limits for a single POI.')
    return plr_data.poi_names()[0]

  @abstractmethod
  def fill_pv(self, plr_data) :
    pass

  def compute_fast_plr(self, hypo, data, name = 'fast') :
    plr_data =  PLRData(name, hypo)
    tmu = self.minimizer.tmu(hypo, data, hypo)
    plr_data.test_statistics['tmu'] = tmu
    plr_data.free_fit = FitResult('free_fit', self.minimizer.free_pars, self.minimizer.free_nll)
    plr_data.hypo_fit = FitResult('hypo_fit', self.minimizer.hypo_pars, self.minimizer.hypo_nll)
    return plr_data

  def compute_fast_q(self, hypo, data) :
    fast_plr_data = compute_fast_plr(hypo, data, 'fast')
    asimov = data.model.generate_expected(poi, self)
    asimov_plr_data = compute_fast_plr(hypo, data, 'fast_asimov')
    fast_plr_data.set_asimov_ts(asimov_plr_data)
    self.fill_pv(plr_data)
    return plr_data

  def fill_all_pv(self, scan_data) :
    for plr_data in scan_data.plr_data.values() : plr_data.fill_pv()

  def compute_all_fast_q(self, scan_data, data, name = 'fast') :
    fast_plr_data = {}
    for hypo in scan_data.plr_data :
      fast_plr_data[hypo] = self.compute_fast_q(hypo, data)
    return ScanData(name, fast_plr_data)


class QMuCalculator :
  def __init__(self, minimizer) :
    super().__init__(minimizer)

  def fill_pv(self, plr_data) :
    try :
      # since we use tmu_A to compute CLb, we need tmu_A = tmu_0 (computed from an Asimov with mu'=0)
      q = QMu(test_poi = plr_data.hypo[self.poi_name()], tmu = plr_data.test_statistics['tmu'],
              best_poi = plr_data.best_fit.pars[self.poi_name()], tmu_A = plr_data.test_statistics['tmu_0'])
      plr_data.test_statistics['q_mu'] = q.value()
      plr_data.pvs['pv' ] = q.asymptotic_pv()
      plr_data.pvs['cls'] = q.asymptotic_cls()
      plr_data.pvs['clb'] = q.asymptotic_clb()
    except Exception as inst:
      print("q_mu computation failed for PLR '%s', hypothesis %s, with exception below:" % (plr_data.name, plr_data.hypo))
      print(inst)
      return None
    return self


class QMuTildaCalculator(LimitCalculator) :
  def __init__(self, minimizer) :
    super().__init__(minimizer)
    self.qs = []

  def fill_pv(self, plr_data) :
      try :
        # since we use tmu_A to compute CLb, we need tmu_A = tmu_0 (computed from an Asimov with mu'=0)
        q = QMuTilda(test_poi = plr_data.hypo[self.poi_name()], tmu = plr_data.test_statistics['tmu'],
              best_poi = plr_data.best_fit.pars[self.poi_name()], tmu_A = plr_data.test_statistics['tmu_0'],
              tmu_0 = plr_data.test_statistics['tmu_0'])
        plr_data.test_statistics['qm~u'] = q.value()
        plr_data.pvs['pv' ] = q.asymptotic_pv()
        plr_data.pvs['cls'] = q.asymptotic_cls()
        plr_data.pvs['clb'] = q.asymptotic_clb()
      except Exception as inst:
        print("q~mu computation failed for computation '%s', hypothesis %s, with exception below:" % (result_name, hypo))
        print(inst)
        return None
    return self


class ParBound :
  def __init__(self, par, minval = None, maxval = None) :
    self.par = par
    self.minval = minval
    self.maxval = maxval
  def test(self, pars) :
    try :
      return (pars[self.par] >= self.minval if self.minval != None else True) and (pars[self.par] <= self.maxval if self.maxval != None else True)
    except KeyError :
      return True
  def __str__(self) :
    smin = '%s >= %g' % (self.par, self.minval) if self.minval != None else ''
    smax = '%s <= %g' % (self.par, self.maxval) if self.maxval != None else ''
    if smin == '' : return smax
    if smax == '' : return smin
    return smin + ' and ' + smax
