import json
import math
import scipy
import numpy as np
import copy

from .core import Model, Data, Parameters, JSONSerializable
from .minimizers import OptiMinimizer
from .test_statistics import QMu, QMuTilda


class FitResult(JSONSerializable) :
  def __init__(self, name = '', pars = None, nll = None) :
    super().__init__()
    self.pars = pars
    self.nll = nll
  def load_jdict(self, jdict) :
    self.pars = Parameters.set_from_dict(jdict['pars'])
    self.nll = jdict['nll']
    return self
  def save_jdict(self, jdict) :
    jdict['pars'] = self.pars.dict()
    jdict['nll'] = self.nll
  def __str__(self) :
    s =  "  FitResult '%s' : " % self.name
    s += "    - nll = %g" % self.nll
    s += "    - best-fit values = %s" % str(self.pars)

class PVResult(JSONSerializable) :
  def __init__(self, name = '', hypo = None, free_fit = None, hypo_fit = None, test_statistics = {}, pvs = {}) :
    super().__init__()
    self.free_fit = free_fit
    self.hypo_fit = hypo_fit
    self.hypo = None
    self.test_statistics = test_statistics
    self.pvs = pvs
  def load_jdict(self, jdict) :
    self.free_fit = FitResult('free_fit'.load_jdict(jdict['free_fit'])
    self.hypo_fit = FitResult('hypo_fit').load_jdict(jdict['hypo_fit'])
    self.test_statistics = jdict['test_statistics']
    self.pvs = jdict['pvs']
    return self
  def save_jdict(self, jdict) :
    jdict['free_fit'] = self.free_fit.dump_jdict()
    jdict['hypo_fit'] = self.hypo_fit.dump_jdict()
    jdict['test_statistics'] = self.test_statistics
    jdict['pvs'] = self.pvs
  def __str__(self) :
    s = "PVResult '%s' : " % self.name
    s += '\n  Computed for hypothesis:' + str(self.hypo)
    s += '\n  test statistics : %s' % str(self.test_statistics)
    s += '\n  p-values : %s' % str(self.pvs)
    s += '\n  Unconditional fit:' + str(self.free_fit)
    s += '\n  Conditional fit:' + str(self.hypo_fit)

class FitResults(JSONSerializable) :

  def __init__(self, model, hypos = [], filename = '') :
    super().__init__()
    self.model = model
    self.pois = model.pois()
    self.results = {}
    for hypo in hypos : self.results[hypo] = {}
    if filename != '' : self.load(filename)

  def result_names(self) :
    if len(self.results) == 0 : return None
    return list(list(self.results.values())[0].keys())

  def solve(self, result_name, pv_key = 'cls', target = 0.05, order = 3, log_scale = True) :
    if len(self.pois) > 1 : raise ValueError('Cannot interpolate limit in more than 1 dimension.')
    poi = list(self.pois.keys())[0]
    hypos = [ hypo[poi] for hypo in self.results.keys() ]
    values = []
    for hypo, hypo_result in self.results.items() :
      if not result_name in hypo_result : raise KeyError("Result '%s' not found at hypo %s." % (result_name, str(hypo))
      pv_result = hypo_result[result_name]
      if pv_key in pv_result.data : raise KeyError("P-value '%s' not found in result '%s' at hypo %s." % (pv_key, result_name, str(hypo))
      if log_scale :
        value = math.log(pv_results.data[pv_key]/target) if pv_results.data[pv_key] > 0 else -999
      else :
        value = pv_results.data[pv_key] - target for result in self.fit_results ]
      data.append(append(value)
    finder = scipy.interpolate.InterpolatedUnivariateSpline(hypos, values, k=order)
    roots = finder.roots() 
    if len(roots) == 0 :
      print("No solution found for %s[%s] = %g in computation '%s'. Interpolation set:" % (pv_key, poi, target, result_name))
      print([a for a in zip(hypos, values)])
      return None
    if len(roots) > 1 :
      print('Multiple solutions found for %s[%s] = %g in computation '%s', returning the first one' % (pv_key, poi, target, result_name))
    return roots[0]

  def load_jdict(self, jdict) :
    for poi_dict in jdict['POIs'] :
      poi = ModelPOI().load_jdict(poi_dict)
      self.pois[poi.name] = poi
    for fit_dict in jdict['results'] :
      hypo = Parameters().set_from_dict(fit_dict['hypo'])
      hypo_results = {}
      for name, pv_result in fit_dict['pv_results'].items() :
        hypo_results[name] = PVResult(name, hypo).load_jdict(result)
      self.results[hypo] = hypo_results
    return self

  def fill_jdict(self, jdict) :
    jdict['POIs'] = []
    for poi in self.pois.values() : jdict['POIs'].append(poi.dump_jdict())
    jdict['results'] = []
    for hypo, results in self.fits.items() :
      fit_dict = {}
      fit_dict['hypo'] = hypo.dict(pois_only = True)
      fit_dict['pv_results'] = {}
      for name, pv_result in results.items() :
        fit_dict['pv_results'][name] = pv_result.dump_jdict()

  def __str__(self) :
    s = ''
    s += 'POIs : ' + str(self.pois) + '\n'
    s += 'Results : '
    for hypo, pv_results in self.results.items() :
      s += '\nHypo :' + str(hypo)
      for pv_result in pv_results : s += '\n' + str(pv_result)
    return s

  def key_value(self, key, hypo, name = None) :
    if not hypo in self.results : raise KeyError('While trying to access key %s in result %s, hypo %s was not found' % (key, name, str(hypo)))
    if name != None and not name in self.results[hypo] : raise KeyError('While trying to access key %s in hypo %s, result %s was not found' % (key, str(hypo), name))
    for poi in self.pois.values() :
      if key == poi.name : return hypo[poi.name]
      if key == 'best_' + poi.name : return self.results[hypo][name].free_fit[poi.name]
    if key in self.results[hypo][name].pvs : return self.results[hypo][name].pvs[key]
    if key in self.results[hypo][name].test_statistics : return self.results[hypo][name].test_statistics[key]
    raise KeyError('No data found for key %s in result %s of hypo %s' % (key, name, str(hypo)))
  
  def print(self, print_keys = [], verbosity = 0, print_limits=True) :
    if len(self.fit_results) == 0 : return ''
    if print_keys == [] :
      print_keys = self.pois.keys() + [ 'pv' ]
      if verbosity > 0 :
        print_keys.extend([ 'cls', 'clb' ])
      if verbosity > 1 :
        print_keys.extend([ 'tmu' ] + [ 'best_' + k for k in self.pois.keys() ]
    s = ''
    for k in print_keys : s += '| %-15s ' % k
    for hypo, results in self.results.items() : 
      for key in print_keys :
        if key in self.pois : 
          s += '| %-15g ' % self.key_value(key, hypo)
        for name in results :
          for key in print_keys :
            s += '| %-15g ' % self.key_value(key, hypo, name)
    if print_limits and len(self.pois) == 1 :
      for result_name in self.result_names() :
        limit = self.solve(result_name, 'cls', 0.05)
        limit_str = '%g' % limit if limit != None else 'not computable'
        s += '\n' + "Asymptotic 95%% CLs limit for '%s' computation = %s" % (result_name, limit_str)
    return s


class LimitCalculator :
  def __init__(self, fit_results) :
    self.fit_results = fit_results
    if len(fit_results.pois) != 1 : raise ValueError('Cannot compute upper limits for more than 1 POI.')
    self.poi_name = list(fit_results.pois.keys())[0]
  def limit(self, result_name, pv_key, cl = 0.95) :
    return self.results.solve(result_name, pv_key , 1 - cl, log_scale = True)


class QMuCalculator(LimitCalculator) :
  def __init__(self, minimizer, fit_results) :
    super().__init__(fit_results)
    self.minimizer = minimizer
  def fill_pv(self, result_name) :
    for hypo, pv_results in self.fit_results.results.items() :
      if not result_name in pv_results : raise ValueError("Result '%s' not found at hypothesis %s." % (result_name, str(hypo)))
      pv_result = pv_results[result_name]
      try :
        # since we use tmu_A to compute CLb, we need tmu_A = tmu_0 (computed from an Asimov with mu'=0)
        q = QMu(test_poi = hypo[self.poi_name], tmu = pv_result.test_statistics['tmu'], best_poi = pv_result.free_fit.pars[self.poi_name], tmu_A = pv_result.test_statistics['tmu_0'])
        pv_result.test_statistics['q_mu'] = q.value()
        pv_result.pvs['pv' ] = q.asymptotic_pv()
        pv_result.pvs['cls'] = q.asymptotic_cls()
        pv_result.pvs['clb'] = q.asymptotic_clb()
      except Exception as inst:
        print("q_mu computation failed for computation '%s', hypothesis %s, with exception below:" % (result_name, hypo))
        print(inst)
        return None
    return self

  def compute_fast_results(self, result_name, data = None) :
    if data != None : self.minimizer.data = data
    for hypo, pv_results in self.fit_results.results.items() :
      tmu = self.minimizer.tmu(hypo, hypo)
      pv_result = PVResult(result_name, hypo)
      pv_result.test_statistics['tmu'] = tmu
      pv_result.free_fit = FitResult('free_fit', self.minimizer.free_pars, self.minimizer.free_nll)
      pv_result.hypo_fit = FitResult('hypo_fit', self.minimizer.hypo_pars, self.minimizer.hypo_nll)
      asimov = data.model.generate_expected(poi, self)
      tmu_0 = self.minimizer.asimov_clone(0).tmu(hypo, hypo)
      pv_result.test_statistics['tmu_0'] = tmu_0
      q = QMu(test_poi = hypo[self.poi_name], tmu = tmu, best_poi = self.minimizer.min_poi)
      pv_result.test_statistics['q_mu'] = tmu = q.value()
    self.fill_qpv(result_name, data)
    return self


class QMuTildaCalculator(LimitCalculator) :
  def __init__(self, minimizer, results) :
    super().__init__(results)
    self.minimizer = minimizer
    self.qs = []
  def fill_pv(self, result_name) :
    for hypo, pv_results in self.fit_results.results.items() :
      if not result_name in pv_results : raise ValueError("Result '%s' not found at hypothesis %s." % (result_name, str(hypo)))
      pv_result = pv_results[result_name]
      try :
        # since we use tmu_A to compute CLb, we need tmu_A = tmu_0 (computed from an Asimov with mu'=0)
        q = QMuTilda(test_poi = hypo[self.poi_name], tmu = pv_result.test_statistics['tmu'], best_poi = pv_result.free_fit.pars[self.poi_name],
                   tmu_A = pv_result.test_statistics['tmu_0'], tmu_0 = pv_result.test_statistics['tmu_0'])
        pv_result.test_statistics['qm~u'] = q.value()
        pv_result.pvs['pv' ] = q.asymptotic_pv()
        pv_result.pvs['cls'] = q.asymptotic_cls()
        pv_result.pvs['clb'] = q.asymptotic_clb()
      except Exception as inst:
        print("q~mu computation failed for computation '%s', hypothesis %s, with exception below:" % (result_name, hypo))
        print(inst)
        return None
    return self

  def compute_fast_results(self, result_name, data = None) :
    if data != None : self.minimizer.data = data
    for hypo, pv_results in self.fit_results.results.items() :
      tmu = self.minimizer.tmu(hypo, hypo)
      pv_result = PVResult(result_name, hypo)
      pv_result.test_statistics['tmu'] = tmu
      pv_result.free_fit = FitResult('free_fit', self.minimizer.free_pars, self.minimizer.free_nll)
      pv_result.hypo_fit = FitResult('hypo_fit', self.minimizer.hypo_pars, self.minimizer.hypo_nll)
      asimov = data.model.generate_expected(poi, self)
      tmu_0 = self.minimizer.asimov_clone(0).tmu(hypo, hypo)
      pv_result.test_statistics['tmu_0'] = tmu_0
      q = QMuTilda(test_poi = hypo[self.poi_name], tmu = tmu, best_poi = self.minimizer.min_poi)
      pv_result.test_statistics['q_mu'] = tmu = q.value()
    self.fill_qpv(result_name, data)
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
