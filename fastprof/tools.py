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

  def load(self, filename) :
    with open(filename, 'r') as fd :
      jdict = json.load(fd)
      return self.load_jdict(jdict)

  def save(self, filename) :
    with open(filename, 'w') as fd :
      jdict = self.dump_jdict()
      return json.dump(jdict, fd, ensure_ascii=True, indent=3)

  def load_json(self, js) :
    jdict = json.loads(js)
    return self.load_jdict(jdict)

  def dump_json(self) :
    jdict = self.dump_jdict()
    return json.dumps(jdict)

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

  def fill(self) : # Fill in the missing pieces
    for fit_result in self.fit_results :
      if not 'cl' in fit_result :
        if self.test_statistic == 'qmu' :
          fit_result['cl'] = QMu(fit_result['qmu'], fit_result[self.poi_name], fit_result['best_fit_val']).asymptotic_cl()
      if not 'cls' in fit_result :
        if self.test_statistic == 'qmu' :
          fit_result['cls'] = QMu(fit_result['qmu'], fit_result[self.poi_name], fit_result['best_fit_val']).asymptotic_cls(qA=fit_result['qmu_A'])
    return self
  
  def check(self, data, verbosity = 0) :
    self.fill()
    results_fast = copy.deepcopy(self)
    for result_full, result_fast in zip(self.fit_results, results_fast.fit_results) :
      mini = OptiMinimizer(data, result_full[self.poi_name], (self.poi_min, self.poi_max))
      tmu, min_mu = mini.tmu(result_full[self.poi_name])
      q = QMu(tmu, result_full[self.poi_name], min_mu)
      result_fast['cl'] = q.asymptotic_cl()
      print('%s = %4g : Asymptotic reference CL = %6.4f , fast CL = %6.4f.' % (self.poi_name, result_full[self.poi_name], result_full['cl'], q.asymptotic_cl()))
      if verbosity > 0 : print('     (tmu = %g, min_mu = %g)' % (tmu, min_mu))
      if verbosity > 1 :
        print('Unconditional minimum:')
        print(mini.min_pars)
        print('Conditional minimum:')
        print(mini.hypo_pars)
        print('\n')
    limit_full = self.solve('cl', 0.05)
    limit_fast = results_fast.solve('cl', 0.05)
    print('Asymptotic 95%% CLs limits: reference = %g, fast = %g.' % (limit_full, limit_fast))
      
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
