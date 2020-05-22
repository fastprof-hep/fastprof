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
    self.poi_name              = jdict['POI_name']
    self.poi_initial_value     = jdict['POI_initial_value']
    self.poi_min, self.poi_max = jdict['POI_range']
    self.fit_results           = jdict['fit_results']
    for fit_result in self.fit_results :
      fit_result['free_pars'] = self.make_pars(fit_result, 'free_', 'best_fit_val')
      fit_result['hypo_pars'] = self.make_pars(fit_result, 'hypo_', self.poi_name)
    self.hypos= np.array([ fit_result[self.poi_name] for fit_result in self.fit_results ])
    return self

  def dump_jdict(self) :
    jdict = {}
    jdict['POI_name'] = self.poi_name
    jdict['POI_initial_value'] = self.poi_init_val
    jdict['POI_range'] = self.poi_min, self.poi_max
    jdict['fit_results'] = self.fit_results
    return jdict

  def make_pars(self, fr, prefix, poi_key) :
    pars = Parameters(model=self.model)
    pars.poi = fr[poi_key]
    for par in fr :
      if par.startswith(prefix) : pars.set_np(par[len(prefix):], fr[par], unscaled=True)
    return pars

  def str_rep(self, print_keys = [], verbosity = 0) :
    if len(self.fit_results) == 0 : return ''
    if print_keys == [] :
      print_keys = [ self.poi_name, 'cl', 'fast_cl', 'sampling_cl' ]
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
        s += '\n' + 'Unconditional minimum:'
        s += '\n' + str(fit_result['fast_free_pars'])
        s += '\n' + 'Conditional minimum:'
        s += '\n' + str(fit_result['fast_hypo_pars'])
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


class QMuCalculator :
  def __init__(self, minimizer, results) :
    self.minimizer = minimizer
    self.results = results

  def fill_qcl(self, q_key = 'qmu', cl_key = 'cl', cls_key = 'cls', clb_key = 'clb', tmu_key = 'tmu', best_poi_key = 'best_fit_val', tmu_A_key = 'tmu_A') :
    for fit_result in self.results.fit_results :
      q = QMu(test_poi = fit_result[self.results.poi_name], tmu = fit_result[tmu_key], best_poi = fit_result[best_poi_key], tmu_A = fit_result[tmu_A_key])
      #print(fit_result[self.results.poi_name], fit_result[tmu_key], fit_result[best_poi_key], fit_result[tmu_A_key])
      if not   q_key in fit_result : fit_result[  q_key] = q.value()
      if not  cl_key in fit_result : fit_result[ cl_key] = q.asymptotic_cl()
      if not cls_key in fit_result : fit_result[cls_key] = q.asymptotic_cls()
      if not clb_key in fit_result : fit_result[clb_key] = q.asymptotic_clb()
    return self

  def fill_fast_results(self, q_key = 'fast_q', cl_key = 'fast_cl', cls_key = 'fast_cls', clb_key = 'fast_clb',
                        tmu_key = 'fast_tmu', best_poi_key = 'fast_best_fit_val', tmu_A_key = 'fast_tmu_A',
                        free_pars_key = 'fast_free_pars', hypo_pars_key = 'fast_hypo_pars') :
    for fit_result in self.results.fit_results :
      poi = fit_result[self.results.poi_name]
      tmu = self.minimizer.tmu(poi)
      fit_result[tmu_key] = tmu
      fit_result[best_poi_key] = self.minimizer.min_poi
      fit_result[free_pars_key] = self.minimizer.min_pars
      fit_result[hypo_pars_key] = self.minimizer.hypo_pars
      tmu_A = self.minimizer.asimov_clone(0).tmu(poi)
      fit_result[tmu_A_key] = tmu_A
      q = QMu(test_poi = poi, tmu = tmu, best_poi = self.minimizer.min_poi)
      fit_result[q_key] = q.value()
    self.fill_qcl(q_key = 'fast_qmu', cl_key = 'fast_cl', cls_key = 'fast_cls', clb_key = 'fast_clb', tmu_key = 'fast_tmu', best_poi_key = 'fast_best_fit_val', tmu_A_key = 'fast_tmu_A')
    return self


class QMuTildaCalculator :
  def __init__(self, minimizer, results) :
    self.minimizer = minimizer
    self.results = results
    self.qs = []
    self.fast_qs = []
  def fill_qcl(self, q_key = 'qmu_tilda', cl_key = 'cl', cls_key = 'cls', clb_key = 'clb', tmu_key = 'tmu', best_poi_key = 'best_fit_val', tmu_A_key = 'tmu_A', tmu_0_key = 'tmu_0') :
    for fit_result in self.results.fit_results :
      q = QMuTilda(test_poi = fit_result[self.results.poi_name], tmu = fit_result[tmu_key], best_poi = fit_result[best_poi_key], tmu_A = fit_result[tmu_A_key], tmu_0 = fit_result[tmu_0_key])
      #print(fit_result[self.results.poi_name], fit_result[tmu_key], fit_result[best_poi_key], fit_result[tmu_A_key])
      if not   q_key in fit_result : fit_result[  q_key] = q.value()
      if not  cl_key in fit_result : fit_result[ cl_key] = q.asymptotic_cl()
      if not cls_key in fit_result : fit_result[cls_key] = q.asymptotic_cls()
      if not clb_key in fit_result : fit_result[clb_key] = q.asymptotic_clb()
      self.qs.append(q)
    return self

  def fill_fast_results(self, q_key = 'fast_qmu_tilda', cl_key = 'fast_cl', cls_key = 'fast_cls', clb_key = 'fast_clb', 
                        tmu_key = 'fast_tmu', best_poi_key = 'fast_best_fit_val', tmu_A_key = 'fast_tmu_A', tmu_0_key = 'fast_tmu_0',
                        free_pars_key = 'fast_free_pars', hypo_pars_key = 'fast_hypo_pars') :
    for fit_result in self.results.fit_results :
      poi = fit_result[self.results.poi_name]
      tmu = self.minimizer.tmu(poi)
      fit_result[tmu_key] = tmu
      fit_result[best_poi_key] = self.minimizer.min_poi
      fit_result[free_pars_key] = self.minimizer.min_pars
      fit_result[hypo_pars_key] = self.minimizer.hypo_pars
      tmu_A = self.minimizer.asimov_clone(0).tmu(poi)
      fit_result[tmu_A_key] = tmu_A
      fit_result[tmu_0_key] = tmu_A # since we use a 0-Asimov for tmu_A already
      q = QMuTilda(test_poi = poi, tmu = tmu, best_poi = self.minimizer.min_poi)
      fit_result[q_key] = q.value()
      self.fast_qs.append(q)
    self.fill_qcl(q_key = 'fast_qmu_tilda', cl_key = 'fast_cl', cls_key = 'fast_cls', clb_key = 'fast_clb',
                  tmu_key = 'fast_tmu', best_poi_key = 'fast_best_fit_val', tmu_A_key = 'fast_tmu_A', tmu_0_key = 'fast_tmu_0')
    return self
