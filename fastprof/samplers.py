import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import copy
from abc import abstractmethod
from scipy.interpolate import InterpolatedUnivariateSpline

from .core import Parameters
from .test_statistics import QMu, QMuTilda
from .sampling import SamplingDistribution
from .minimizers import OptiMinimizer, ScanMinimizer

# -------------------------------------------------------------------------
class Sampler :
  def __init__(self, model, gen_hypo = None, print_freq = 1000) :
    self.model = model
    self.gen_hypo = model.expected_pars(gen_hypo) if isinstance(gen_hypo, (int, float)) else gen_hypo
    self.freq = print_freq

  def progress(self, k, ntoys) :
    if k % self.freq == 0 : print('-- Processing iteration %d of %d' % (k, ntoys))

  def generate(self, ntoys) :
    print('Generating POI hypothesis %s. Full gen hypo = ' % str(self.gen_hypo.pois))
    print(str(self.gen_hypo))
    self.dist = SamplingDistribution(ntoys)
    for k in range(0, ntoys) :
      self.progress(k, ntoys)
      success = False
      while not success :
        if self.debug : print('DEBUG: iteration %d generating data for hypo %s.' % (k, str(self.gen_hypo.pois)))
        data = self.model.generate_data(self.gen_hypo)
        result = self.compute(data, k)
        if result != None :
          success = True
        else :
          print('Processing toy iteration %d failed, repeating it.' % k)
      self.dist.samples[k] = result
    return self.dist

  @abstractmethod
  def compute(self, data, toy_iter) :
     pass


# -------------------------------------------------------------------------
class ScanSampler (Sampler) :
  def __init__(self, model, test_hypo, scan_mus, gen_hypo = None, print_freq = 1000, tmu_A = None, tmu_0 = None) :
    super().__init__(model, gen_hypo, print_freq)
    self.test_hypo = model.expected_pars(test_hypo) if isinstance(test_hypo, (int, float)) else test_hypo
    if self.gen_hypo == None : self.gen_hypo = copy.deepcopy(self.test_hypo)
    self.scan_mus = scan_mus
    self.tmu_A = tmu_A
    self.tmu_0 = tmu_0
    self.use_qtilda = True if tmu_A != None and tmu_0 != None else False
    
  def compute(self, data, toy_iter) :
    opti = ScanMinimizer(self.scan_mus)
    tmu = opti.tmu(self.test_hypo, data, self.test_hypo)
    if self.use_qtilda :
      q = QMuTilda(test_poi = self.test_hypo.pois[0], tmu = tmu, best_poi = opti.min_poi, tmu_A = self.tmu_A, tmu_0 = self.tmu_0)
    else :
      q = QMu(test_poi = self.test_hypo.pois[0], tmu = tmu, best_poi = opti.min_poi)
    return q.asymptotic_pv()


# -------------------------------------------------------------------------
class OptiSampler (Sampler) :
  def __init__(self, model, test_hypo, mu0 = 0, poi_bounds = None, bounds = [], method = 'scalar', gen_hypo = None, print_freq = 1000, niter=1, tmu_A = None, tmu_0 = None, floor=1E-7, debug=False) :
    super().__init__(model, gen_hypo, print_freq)
    self.test_hypo = model.expected_pars(test_hypo) if isinstance(test_hypo, (int, float)) else test_hypo
    if self.gen_hypo == None : self.gen_hypo = copy.deepcopy(self.test_hypo)
    self.mu0 = mu0
    self.poi_bounds = poi_bounds
    self.bounds = bounds
    self.method = method
    self.debug = debug
    self.niter = niter
    self.floor = floor
    self.tmu_A = tmu_A
    self.tmu_0 = tmu_0
    self.use_qtilda = True if tmu_A != None and tmu_0 != None else False
    self.debug_data = pd.DataFrame()
    
  def compute(self, data, toy_iter) :
    opti = OptiMinimizer(self.mu0, self.poi_bounds, self.method, self.niter, self.floor)
    if self.debug : opti.debug = 2
    tmu = opti.tmu(self.test_hypo, data, self.test_hypo)
    if tmu < 1E-7 :
      print('Warning: tmu <= 0 at toy iteration %d' % toy_iter)
      if self.debug and opti.tmu_debug < -10 : data.save('data/debug_data_neg_tmu_%d.json' % toy_iter)
      return None
    for bound in self.bounds :
      if not bound.test(opti.free_deltas) :
        print('Warning: free fit parameters below fail bound %s' % str(bound))
        print(opti.free_pars)
        return None
      if not bound.test(opti.hypo_deltas) :
        print('Warning: hypothesis fit parameters below fail bound %s' % str(bound))
        print(opti.hypo_pars)
        return None
    if self.debug :
      print('DEBUG: fitting data with mu0 = %g and range = %g, %g -> t = %g, mu_hat = %g.' % (self.mu0, *self.poi_bounds, tmu, opti.min_poi))
      print(opti.free_pars)
      print(opti.hypo_pars)
    if self.use_qtilda :
      q = QMuTilda(test_poi = self.test_hypo.pois[0], tmu = tmu, best_poi = opti.min_poi, tmu_A = self.tmu_A, tmu_0 = self.tmu_0)
    else :
      q = QMu(test_poi = self.test_hypo.pois[0], tmu = tmu, best_poi = opti.min_poi)
    if self.debug :
      if self.debug_data.shape[0] == 0 :
        columns = [ 'pv', 'tmu', 'mu_hat', 'free_nll', 'hypo_nll', 'nfev' ]
        columns.extend( [ 'free_' + p for p in self.model.nps ] )
        columns.extend( [ 'hypo_' + p for p in self.model.nps ] )
        self.debug_data = pd.DataFrame(columns=columns)
      self.debug_data.at[toy_iter, 'pv'      ] = q.asymptotic_pv()
      self.debug_data.at[toy_iter, 'tmu'     ] = tmu
      self.debug_data.at[toy_iter, 'mu_hat'  ] = opti.min_poi
      self.debug_data.at[toy_iter, 'free_nll'] = opti.free_nll
      self.debug_data.at[toy_iter, 'hypo_nll'] = opti.hypo_nll
      self.debug_data.at[toy_iter, 'nfev'    ] = opti.nfev
      for i, p in enumerate(self.model.nps) :
        self.debug_data.at[toy_iter, 'free_' + p] = opti.free_pars[p]
        self.debug_data.at[toy_iter, 'hypo_' + p] = opti.hypo_pars[p]
        self.debug_data.at[toy_iter, 'aux_'  + p] = data.aux_obs[i]
      data.save('data/debug_data_%d.json' % toy_iter)
    return q.asymptotic_pv()


# -------------------------------------------------------------------------
class LimitSampler (Sampler) :
  def __init__(self, model, gen_hypo, limit_calc, cl = 0.95, print_freq = 1000) :
    super().__init__(model, gen_hypo, print_freq)
    self.limit_calc = limit_calc
    self.cl = cl

  def compute(self, data, toy_iter) :
    self.limit_calc.fill_fast_results(data = data, pv_key = 'fast_pv')
    return self.limit_calc.limit(pv_key = 'fast_pv', cl=self.cl)
