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
  def __init__(self, model, test_hypo, gen_hypo = None, print_freq = 1000) :
    self.model = model
    if isinstance(test_hypo, (int, float)) :
      self.test_hypo = model.expected_pars(test_hypo)
    else :
      self.test_hypo = test_hypo
    if gen_hypo == None :
      self.gen_hypo = copy.deepcopy(self.test_hypo)
    elif isinstance(gen_hypo, (int, float)) :
      self.gen_hypo = model.expected_pars(gen_hypo)
    else :
      self.gen_hypo = gen_hypo
    self.freq = print_freq

  def progress(self, k, ntoys) :
    if k % self.freq == 0 : print('-- Processing iteration %d of %d' % (k, ntoys))

  def generate(self, ntoys) :
    print('Generating POI hypothesis %g, and will compute at %g. Full gen hypo = ' % (self.gen_hypo.poi, self.test_hypo.poi))
    print(str(self.gen_hypo))
    self.dist = SamplingDistribution(ntoys)
    for k in range(0, ntoys) :
      self.progress(k, ntoys)
      success = False
      while not success :
        if self.debug : print('DEBUG: iteration %d generating data for hypo %g.' % (k, self.gen_hypo.poi))
        data = self.model.generate_data(self.gen_hypo)
        cl = self.compute_cl(data, k)
        if cl != None :
          success = True
        else :
          print('Processing toy iteration %d failed, repeating it.' % k)
      self.dist.samples[k] = cl
    return self.dist

  @abstractmethod
  def compute_cl(self, data, toy_iter) :
     pass


# -------------------------------------------------------------------------
class ScanSampler (Sampler) :
  def __init__(self, model, test_hypo, scan_mus, gen_hypo = False, print_freq = 1000, tmu_A = None, tmu_0 = None) :
    super().__init__(model, test_hypo, gen_hypo, print_freq)
    self.scan_mus = scan_mus
    self.tmu_A = tmu_A
    self.tmu_0 = tmu_0
    self.use_qtilda = True if tmu_A != None and tmu_0 != None else False
    
  def compute_cl(self, data, toy_iter) :
    opti = ScanMinimizer(data, self.scan_mus)
    tmu = opti.tmu(self.test_hypo, self.test_hypo)
    if self.use_qtilda :
      q = QMuTilda(test_poi = self.test_hypo.poi, tmu = tmu, best_poi = opti.min_poi, tmu_A = self.tmu_A, tmu_0 = self.tmu_0)
    else :
      q = QMu(test_poi = self.test_hypo.poi, tmu = tmu, best_poi = opti.min_poi)
    return q.asymptotic_cl()


# -------------------------------------------------------------------------
class OptiSampler (Sampler) :
  def __init__(self, model, test_hypo, mu0 = 0, bounds = None, method = 'scalar', gen_hypo = None, print_freq = 1000, niter=1, tmu_A = None, tmu_0 = None, floor=1E-7, debug=False) :
    super().__init__(model, test_hypo, gen_hypo, print_freq)
    self.mu0 = mu0
    self.bounds = bounds
    self.method = method
    self.debug = debug
    self.niter = niter
    self.floor = floor
    self.tmu_A = tmu_A
    self.tmu_0 = tmu_0
    self.use_qtilda = True if tmu_A != None and tmu_0 != None else False
    self.debug_data = None
    
  def compute_cl(self, data, toy_iter) :
    opti = OptiMinimizer(data, self.mu0, self.bounds, self.method, self.niter, self.floor)
    tmu = opti.tmu(self.test_hypo, self.test_hypo)
    if tmu == 0 :
      print('Warning: tmu <= 0 at toy iteration %d' % toy_iter)
      if self.debug and opti.tmu_debug < -10 : data.save('data_%d.json' % toy_iter)
      return None
    if self.debug :
      print('DEBUG: fitting data with mu0 = %g and range = %g, %g -> t = %g, mu_hat = %g.' %(self.mu0, *self.bounds, tmu, opti.min_poi))
      print(opti.min_pars)
      print(opti.hypo_pars)
    if self.use_qtilda :
      q = QMuTilda(test_poi = self.test_hypo.poi, tmu = tmu, best_poi = opti.min_poi, tmu_A = self.tmu_A, tmu_0 = self.tmu_0)
    else :
      q = QMu(test_poi = self.test_hypo.poi, tmu = tmu, best_poi = opti.min_poi)
    if self.debug :
      if self.debug_data == None :
        columns = [ 'cl', 'tmu', 'mu_hat', 'free_nll', 'hypo_nll', 'nfev' ]
        columns.extend( [ 'free_' + a for a in self.model.alphas ] )
        columns.extend( [ 'free_' + b for b in self.model.betas  ] )
        columns.extend( [ 'free_' + c for c in self.model.gammas ] )
        columns.extend( [ 'hypo_' + a for a in self.model.alphas ] )
        columns.extend( [ 'hypo_' + b for b in self.model.betas  ] )
        columns.extend( [ 'hypo_' + c for c in self.model.gammas ] )
        columns.extend( [ 'aux_'  + a for a in self.model.alphas ] )
        columns.extend( [ 'aux_'  + b for b in self.model.betas  ] )
        self.debug_data = pd.DataFrame(columns=columns)
      self.debug_data.at[toy_iter, 'cl'      ] = q.asymptotic_cl()
      self.debug_data.at[toy_iter, 'tmu'     ] = tmu
      self.debug_data.at[toy_iter, 'mu_hat'  ] = opti.min_poi
      self.debug_data.at[toy_iter, 'free_nll'] = opti.free_nll
      self.debug_data.at[toy_iter, 'hypo_nll'] = opti.hypo_nll
      self.debug_data.at[toy_iter, 'nfev'    ] = opti.nfev
      for i, a in enumerate(self.model.alphas) :
        self.debug_data.at[toy_iter, 'free_' + a] = opti.free_pars.alphas[i]
        self.debug_data.at[toy_iter, 'hypo_' + a] = opti.hypo_pars.alphas[i]
        self.debug_data.at[toy_iter, 'aux_'  + a] = data.aux_alphas[i]
      for i, b in enumerate(self.model.betas) :
        self.debug_data.at[toy_iter, 'free_' + b] = opti.free_pars.betas [i]
        self.debug_data.at[toy_iter, 'hypo_' + b] = opti.hypo_pars.betas [i]
        self.debug_data.at[toy_iter, 'aux_'  + b] = data.aux_betas[i]
      for i, c in enumerate(self.model.gammas) :
        self.debug_data.at[toy_iter, 'free_' + c] = opti.free_pars.gammas[i]
        self.debug_data.at[toy_iter, 'hypo_' + c] = opti.hypo_pars.gammas[i]
      data.save('data/data_%d.json' % toy_iter)
    return q.asymptotic_cl()
