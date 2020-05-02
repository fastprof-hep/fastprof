import json
import math
import scipy
import numpy as np
import copy

from .core import Model, Data, JSONSerializable
from .minimizers import OptiMinimizer
from .test_statistics import QMu


class FitResults (JSONSerializable) :
  def __init__(self, filename = '') :
    super().__init__()
    if filename != '' :
      self.load(filename)

  def load_jdict(self, jdict) :
    self.poi_name              = jdict['POI_name']
    self.poi_initial_value     = jdict['POI_initial_value']
    self.poi_min, self.poi_max = jdict['POI_range']
    self.fit_results           = jdict['fit_results']
    self.test_statistic        = jdict['test_statistic']
    self.hypos= np.array([ fit_result[self.poi_name] for fit_result in self.fit_results ])
    return self

  def dump_jdict(self) :
    jdict = {}
    jdict['POI_name'] = self.poi_name
    jdict['POI_initial_value'] = self.poi_init_val
    jdict['POI_range'] = self.poi_min, self.poi_max
    jdict['test_statistic'] = self.test_statistic
    jdict['fit_results'] = self.fit_results
    return jdict
  
  def str_rep(self, print_keys = [], verbosity = 0) :
    if len(self.fit_results) == 0 : return ''
    if print_keys == [] :
      print_keys = [ self.poi_name, 'cl', 'fast_cl', 'sampling_cl' ]
      if verbosity > 0 :
        print_keys.extend([ 'cls', 'fast_cls', 'sampling_cls', 'clb', 'fast_clb', 'sampling_clb' ])
      if verbosity > 1 :
        print_keys.extend([ 'qmu', 'fast_qmu', 'best_fit_val', 'fast_best_fit_val' ])
    s = ''
    for k in print_keys :
      if k in self.fit_results[0] : s += '| %-15s ' % k
    for fit_result in self.fit_results :
      s += '\n'
      for k in print_keys :
        if k in fit_result : s += '| %15.4g ' % fit_result[k]
      if verbosity > 2 :
        s += '\n' + 'Unconditional minimum:'
        s += '\n' + str(fit_result['free_pars'])
        s += '\n' + 'Conditional minimum:'
        s += '\n' + str(fit_result['hypo_pars'])
    limit_full = self.solve('cl'     , 0.05)
    limit_fast = self.solve('fast_cl', 0.05)
    s += '\n' + 'Asymptotic 95%% CLs limits: reference = %g, fast = %g.' % (limit_full, limit_fast)
    return s

  def __str__(self) :
    return self.str_rep(verbosity = 0)

  def print(self, print_keys = [], verbosity = 0) :
    print(self.str_rep(print_keys, verbosity))

  def solve(self, result_key, target = 0.05, order = 3, log_scale = True) :
    if log_scale :
      values = np.array([ math.log(result[result_key]/target) if result[result_key] > 0 else -999 for result in self.fit_results ])
    else :
      values = [ result_key - target for result in self.fit_results ]
    finder = scipy.interpolate.InterpolatedUnivariateSpline(self.hypos, values, k=order)
    roots = finder.roots() 
    if len(roots) == 0 :
      print('No solution found for fit_results[%s] = %g' % (result_key, target))
      return None
    if len(roots) > 1 :
      print('Multiple solutions found for fit_results[%s] = %g, returning the first one' % (result_key, target))
    return roots[0]


class QMuCalculator :
  def __init__(self, minimizer, results) :
    self.minimizer = minimizer
    self.results = results
    if self.results.test_statistic != 'qmu' :
      raise ValueError('Cannot process input results produce for test statistic %s -- expecting qmu instead' % self.results.test_statistic)

  def fill_qcl(self, qmu_key = 'qmu', cl_key = 'cl', cls_key = 'cls', clb_key = 'clb', tmu_key = 'tmu', best_mu_key = 'best_fit_val', tmu_A_key = 'tmu_A') :
    for fit_result in self.results.fit_results :
      q = QMu(test_mu = fit_result[self.results.poi_name], tmu = fit_result[tmu_key], best_mu = fit_result[best_mu_key], tmu_A = fit_result[tmu_A_key])
      #print(fit_result[self.results.poi_name], fit_result[tmu_key], fit_result[best_mu_key], fit_result[tmu_A_key])
      if not qmu_key in fit_result : fit_result[qmu_key] = q.value()
      if not cl_key  in fit_result : fit_result[ cl_key] = q.asymptotic_cl()
      if not cls_key in fit_result : fit_result[cls_key] = q.asymptotic_cls()
      if not clb_key in fit_result : fit_result[clb_key] = q.asymptotic_clb()
    return self

  def fill_fast(self, qmu_key = 'fast_qmu', cl_key = 'fast_cl', cls_key = 'fast_cls', clb_key = 'fast_clb', 
                tmu_key = 'fast_tmu', best_mu_key = 'fast_best_fit_val', tmu_A_key = 'fast_tmu_A',
                free_pars_key = 'free_pars', hypo_pars_key = 'hypo_pars') :
    for fit_result in self.results.fit_results :
      mu = fit_result[self.results.poi_name]
      tmu, min_mu = self.minimizer.tmu(mu)
      fit_result[tmu_key] = tmu
      fit_result[best_mu_key] = min_mu
      fit_result[free_pars_key] = self.minimizer.min_pars
      fit_result[hypo_pars_key] = self.minimizer.hypo_pars
      tmu_A, min_mu_A = self.minimizer.asimov_clone(0).tmu(mu)
      fit_result[tmu_A_key] = tmu_A
      q = QMu(test_mu = mu, tmu = tmu, best_mu = min_mu)
      fit_result[qmu_key] = q.value()
    self.fill_qcl(qmu_key = 'fast_qmu', cl_key = 'fast_cl', cls_key = 'fast_cls', clb_key = 'fast_clb', tmu_key = 'fast_tmu', best_mu_key = 'fast_best_fit_val', tmu_A_key = 'fast_tmu_A')
    return self
