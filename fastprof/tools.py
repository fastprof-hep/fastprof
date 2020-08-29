import json
import math
import scipy
import numpy as np
import copy

from .core import Model, Data, Parameters, JSONSerializable
from .minimizers import OptiMinimizer
from .test_statistics import QMu, QMuTilda


class FitResults (JSONSerializable) :
  def __init__(self, model, filename = '') :
    super().__init__()
    self.model = model
    if filename != '' :
      self.load(filename)

  def load_jdict(self, jdict) :
    self.poi_name              = jdict['poi_name']
    self.poi_initial_value     = jdict['poi_initial_value']
    self.poi_min, self.poi_max = jdict['poi_range']
    self.fit_results           = jdict['fit_results']
    for fit_result in self.fit_results :
      fit_result['free_pars'] = self.make_pars(fit_result, 'free_', 'best_fit_val')
      fit_result['hypo_pars'] = self.make_pars(fit_result, 'hypo_', self.poi_name)
    self.hypos= np.array([ fit_result[self.poi_name] for fit_result in self.fit_results ])
    return self

  def fill_jdict(self) :
    jdict['poi_name'] = self.poi_name
    jdict['poi_initial_value'] = self.poi_init_val
    jdict['poi_range'] = self.poi_min, self.poi_max
    jdict['fit_results'] = self.fit_results

  def make_pars(self, fr, prefix, poi_key) :
    pars = Parameters(fr[poi_key], model=self.model)
    for par in fr :
      if par.startswith(prefix) : pars.set(par[len(prefix):], fr[par], unscaled=True)
    return pars

  def str_rep(self, print_keys = [], verbosity = 0) :
    if len(self.fit_results) == 0 : return ''
    if print_keys == [] :
      print_keys = [ self.poi_name, 'pv', 'fast_pv', 'sampling_pv' ]
      if verbosity > 0 :
        print_keys.extend([ 'cls', 'fast_cls', 'sampling_cls', 'clb', 'fast_clb', 'sampling_clb' ])
      if verbosity > 1 :
        print_keys.extend([ 'tmu', 'fast_tmu', 'best_fit_val', 'fast_best_fit_val' ])
    s = ''
    for k in print_keys :
      if k in self.fit_results[0] : s += '| %-15s ' % k
    for fit_result in self.fit_results :
      s += '\n'
      for k in print_keys :
        if k in fit_result : s += '| %15.4g ' % fit_result[k]
      if verbosity > 2 :
        s += '\n' + 'Unconditional minimum (fast):'
        s += '\n' + str(fit_result['fast_free_pars'])
        if verbosity > 3 :
          s += '\n' + 'Unconditional minimum (full):'
          s += '\n' + str(fit_result['free_pars'])
        s += '\n' + 'Conditional minimum (fast):'
        s += '\n' + str(fit_result['fast_hypo_pars'])
        if verbosity > 3 :
          s += '\n' + 'Conditional minimum (full):'
          s += '\n' + str(fit_result['hypo_pars'])
        s += '\n' + 'tmu (full) : %g' % fit_result['tmu']
        s += '\n' + 'tmu (fast) : %g' % fit_result['fast_tmu']
        if verbosity > 3 :
          s += '\n' + 'tmu (fast@full) : %g' % fit_result['fast_tmu@full']
    limit_full = self.solve('cls'     , 0.05)
    limit_fast = self.solve('fast_cls', 0.05)
    limit_full = '%g' % limit_full if limit_full != None else 'not computable'
    limit_fast = '%g' % limit_fast if limit_fast != None else 'not computable'
    s += '\n' + 'Asymptotic 95%% CLs limits: reference = %s, fast = %s.' % (limit_full, limit_fast)
    return s

  def __str__(self) :
    return self.str_rep(verbosity = 0)

  def print(self, print_keys = [], verbosity = 0) :
    print(self.str_rep(print_keys, verbosity))

  def solve(self, result_key, target = 0.05, order = 3, log_scale = True) :
    if not result_key in self.fit_results[0] : return None
    if log_scale :
      values = np.array([ math.log(result[result_key]/target) if result[result_key] > 0 else -999 for result in self.fit_results ])
    else :
      values = [ result_key - target for result in self.fit_results ]
    finder = scipy.interpolate.InterpolatedUnivariateSpline(self.hypos, values, k=order)
    roots = finder.roots() 
    if len(roots) == 0 :
      print('No solution found for fit_results[%s] = %g. Interpolation set:' % (result_key, target))
      print([a for a in zip(self.hypos, values)])
      return None
    if len(roots) > 1 :
      print('Multiple solutions found for fit_results[%s] = %g, returning the first one' % (result_key, target))
    return roots[0]


class LimitCalculator :
  def __init__(self, results) :
    self.results = results

  def limit(self, pv_key, cl = 0.95) :
    return self.results.solve(pv_key , 1 - cl, log_scale = True)


class QMuCalculator(LimitCalculator) :
  def __init__(self, minimizer, results) :
    super().__init__(results)
    self.minimizer = minimizer

  def fill_qpv(self, q_key = 'q_mu', pv_key = 'pv', cls_key = 'cls', clb_key = 'clb', tmu_key = 'tmu', best_poi_key = 'best_fit_val', tmu_0_key = 'tmu_0', data = None) :
    if data != None : self.minimizer.data = data
    for fit_result in self.results.fit_results :
      # since we use tmu_A to compute CLb, we need tmu_A = tmu_0 (computed from an Asimov with mu'=0)
      q = QMu(test_poi = fit_result[self.results.poi_name], tmu = fit_result[tmu_key], best_poi = fit_result[best_poi_key], tmu_A = fit_result[tmu_0_key])
      #print(fit_result[self.results.poi_name], fit_result[tmu_key], fit_result[best_poi_key], fit_result[tmu_0_key])
      if not   q_key in fit_result : fit_result[  q_key] = q.value()
      if not  pv_key in fit_result : fit_result[ pv_key] = q.asymptotic_pv()
      if not cls_key in fit_result : fit_result[cls_key] = q.asymptotic_cls()
      if not clb_key in fit_result : fit_result[clb_key] = q.asymptotic_clb()
    return self

  def fill_fast_results(self, hypo_key = 'hypo_pars', free_key = 'free_pars', q_key = 'fast_q', pv_key = 'fast_pv', cls_key = 'fast_cls', clb_key = 'fast_clb',
                        tmu_key = 'fast_tmu', best_poi_key = 'fast_best_fit_val', tmu_0_key = 'fast_tmu_0',
                        free_pars_key = 'fast_free_pars', hypo_pars_key = 'fast_hypo_pars', fast_tmu_full = 'fast_tmu@full', data = None) :
    if data != None : self.minimizer.data = data
    for fit_result in self.results.fit_results :
      hypo = fit_result[hypo_key] if hypo_key else self.minimizer.model.expected_pars(fit_result[self.results.poi_name])
      tmu = self.minimizer.tmu(hypo, hypo)
      fit_result[tmu_key] = tmu
      fit_result[best_poi_key] = self.minimizer.min_poi
      fit_result[free_pars_key] = self.minimizer.min_pars
      fit_result[hypo_pars_key] = self.minimizer.hypo_pars
      tmu_0 = self.minimizer.asimov_clone(0).tmu(hypo, hypo)
      fit_result[tmu_0_key] = tmu_0
      q = QMu(test_poi = hypo.pois[0], tmu = tmu, best_poi = self.minimizer.min_poi)
      fit_result[q_key] = q.value()
      if hypo_key != '' and free_key != '' :
        fit_result[fast_tmu_full] = 2*(self.minimizer.data.model.nll(fit_result[hypo_key], self.minimizer.data) - self.minimizer.data.model.nll(fit_result[free_key], self.minimizer.data))
    self.fill_qpv(q_key = 'fast_q_mu', pv_key = 'fast_pv', cls_key = 'fast_cls', clb_key = 'fast_clb', tmu_key = 'fast_tmu', best_poi_key = 'fast_best_fit_val', tmu_0_key = 'fast_tmu_0')
    return self


class QMuTildaCalculator(LimitCalculator) :
  def __init__(self, minimizer, results) :
    super().__init__(results)
    self.minimizer = minimizer
    self.qs = []
    self.fast_qs = []
  def fill_qpv(self, q_key = 'q~mu', pv_key = 'pv', cls_key = 'cls', clb_key = 'clb', tmu_key = 'tmu', best_poi_key = 'best_fit_val', tmu_0_key = 'tmu_0', data = None) :
    if data != None : self.minimizer.data = data
    for fit_result in self.results.fit_results :
      # since we use tmu_A to compute CLb, we need tmu_A = tmu_0 (computed from an Asimov with mu'=0)
      q = QMuTilda(test_poi = fit_result[self.results.poi_name], tmu = fit_result[tmu_key], best_poi = fit_result[best_poi_key], tmu_A = fit_result[tmu_0_key], tmu_0 = fit_result[tmu_0_key])
      #print(fit_result[self.results.poi_name], fit_result[tmu_key], fit_result[best_poi_key], fit_result[tmu_0_key])
      if not   q_key in fit_result : fit_result[  q_key] = q.value()
      if not  pv_key in fit_result : fit_result[ pv_key] = q.asymptotic_pv()
      if not cls_key in fit_result : fit_result[cls_key] = q.asymptotic_cls()
      if not clb_key in fit_result : fit_result[clb_key] = q.asymptotic_clb()
      self.qs.append(q)
    return self

  def fill_fast_results(self, hypo_key = 'hypo_pars', free_key = 'free_pars', q_key = 'fast_q~mu', pv_key = 'fast_pv', cls_key = 'fast_cls', clb_key = 'fast_clb', 
                        tmu_key = 'fast_tmu', best_poi_key = 'fast_best_fit_val', tmu_0_key = 'fast_tmu_0',
                        free_pars_key = 'fast_free_pars', hypo_pars_key = 'fast_hypo_pars', fast_tmu_full = 'fast_tmu@full', data = None) :
    if data != None : self.minimizer.data = data
    for fit_result in self.results.fit_results :
      hypo = fit_result[hypo_key] if hypo_key else self.minimizer.model.expected_pars(fit_result[self.results.poi_name])
      tmu = self.minimizer.tmu(hypo, hypo)
      fit_result[tmu_key] = tmu
      fit_result[best_poi_key] = self.minimizer.min_poi
      fit_result[free_pars_key] = self.minimizer.min_pars
      fit_result[hypo_pars_key] = self.minimizer.hypo_pars
      tmu_0 = self.minimizer.asimov_clone(0).tmu(hypo, hypo)
      fit_result[tmu_0_key] = tmu_0
      q = QMuTilda(test_poi = hypo.pois[0], tmu = tmu, best_poi = self.minimizer.min_poi)
      fit_result[q_key] = q.value()
      if hypo_key != '' and free_key != '' :
        fit_result[fast_tmu_full] = 2*(self.minimizer.data.model.nll(fit_result[hypo_key], self.minimizer.data) - self.minimizer.data.model.nll(fit_result[free_key], self.minimizer.data))
      self.fast_qs.append(q)
    self.fill_qpv(q_key = 'fast_q~mu', pv_key = 'fast_pv', cls_key = 'fast_cls', clb_key = 'fast_clb',
                  tmu_key = 'fast_tmu', best_poi_key = 'fast_best_fit_val', tmu_0_key = 'fast_tmu_0')
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
