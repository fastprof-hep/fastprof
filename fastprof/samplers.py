import pyhf
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import copy
from abc import abstractmethod
from scipy.interpolate import InterpolatedUnivariateSpline

from .core import Parameters
from .test_statistics import QMu
from .sampling import SamplingDistribution
from .minimizers import NPMinimizer, OptiMinimizer, ScanMinimizer

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
  @abstractmethod
  def generate(self, ntoys) :
     pass


# -------------------------------------------------------------------------
class ScanSampler (Sampler) :
  def __init__(self, model, test_hypo, scan_mus, gen_hypo = False, print_freq = 1000) :
    super().__init__(model, test_hypo, gen_hypo, print_freq)
    self.scan_mus = scan_mus
    
  def generate(self, ntoys) :
    self.dist = SamplingDistribution(ntoys)
    for k in range(0, ntoys) :
      self.progress(k, ntoys)
      data = self.model.generate_data(self.gen_hypo)
      opti = ScanMinimizer(data, self.scan_mus)
      tmu = opti.tmu(self.test_hypo, self.test_hypo)
      q = QMu(test_poi = self.test_hypo.poi, tmu = tmu, best_poi = opti.min_poi)
      self.dist.samples[k] = q.asymptotic_cl()
    return self.dist


# -------------------------------------------------------------------------
class OptiSampler (Sampler) :
  def __init__(self, model, test_hypo, mu0 = 0, bounds = None, method = 'scalar', gen_hypo = None, print_freq = 1000, debug=False, niter=1) :
    super().__init__(model, test_hypo, gen_hypo, print_freq)
    self.mu0 = mu0
    self.bounds = bounds
    self.method = method
    self.debug = debug
    self.niter = niter
    
  def generate(self, ntoys) :
    print('Generating POI hypothesis %g, and wil compute at %g. Full gen hypo = ' % (self.gen_hypo.poi, self.test_hypo.poi))
    print(str(self.gen_hypo))
    self.dist = SamplingDistribution(ntoys)
    if self.debug :
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
    for k in range(0, ntoys) :
      self.progress(k, ntoys)
      success = False
      while not success :
        if self.debug : print('DEBUG: iteration %d generating data for hypo %g.' % (k, self.gen_hypo.poi))
        data = self.model.generate_data(self.gen_hypo)
        opti = OptiMinimizer(data, self.mu0, self.bounds, self.method, self.niter)
        tmu = opti.tmu(self.test_hypo, self.test_hypo)
        if self.debug : 
          print('DEBUG: fitting data with mu0 = %g and range = %g, %g -> t = %g, mu_hat = %g.' %(self.mu0, *self.bounds, tmu, opti.min_poi))
          print(opti.min_pars)
          print(opti.hypo_pars)
        if tmu != None :
          success = True
        else :
          print('Minimization failed at toy iteration %d, repeating it.' % k)
      q = QMu(test_poi = self.test_hypo.poi, tmu = tmu, best_poi = opti.min_poi)
      #print(mu, tmu, opti.min_poi, '->', q.value(), q.asymptotic_cl())
      self.dist.samples[k] = q.asymptotic_cl()
      if self.debug :
        self.debug_data.at[k, 'cl'      ] = q.asymptotic_cl()
        self.debug_data.at[k, 'tmu'     ] = tmu
        self.debug_data.at[k, 'mu_hat'  ] = opti.min_poi
        self.debug_data.at[k, 'free_nll'] = opti.free_nll
        self.debug_data.at[k, 'hypo_nll'] = opti.hypo_nll
        self.debug_data.at[k, 'nfev'    ] = opti.nfev
        for i, a in enumerate(self.model.alphas) :
          self.debug_data.at[k, 'free_' + a] = opti.free_pars.alphas[i]
          self.debug_data.at[k, 'hypo_' + a] = opti.hypo_pars.alphas[i]
          self.debug_data.at[k, 'aux_'  + a] = data.aux_alphas[i]
        for i, b in enumerate(self.model.betas) :
          self.debug_data.at[k, 'free_' + b] = opti.free_pars.betas [i]
          self.debug_data.at[k, 'hypo_' + b] = opti.hypo_pars.betas [i]
          self.debug_data.at[k, 'aux_'  + b] = data.aux_betas[i]
        for i, c in enumerate(self.model.gammas) :
          self.debug_data.at[k, 'free_' + c] = opti.free_pars.gammas[i]
          self.debug_data.at[k, 'hypo_' + c] = opti.hypo_pars.gammas[i]
        #if tmu > 1000 : data.save('data_%d.json' % k)
        data.save('data/data_%d.json' % k)
    return self.dist


# -------------------------------------------------------------------------
class PyhfSampler (Sampler) :
  def __init__(self, model, test_hypo, nbins, n_np, gen_hypo = None, print_freq = 1000) :
    super().__init__(model, test_hypo, gen_hypo, print_freq)
    self.nbins = nbins
    self.n_np = n_np
    
  def generate_data(self) :
    data = np.zeros(self.nbins + self.n_np)
    expected = self.model.expected_data(self.gen_hypo)
    for i in range(0, self.nbins) :
      data[i] = np.random.poisson(expected.n[i])
    for i in range(0, self.n_np) :
      data[self.nbins + i] = np.random.normal(0, 1)
    return data
  
  def clsb(self, poi, data) :
    return pyhf.infer.hypotest(poi, data, self.model, return_tail_probs = True)[1][0]

  def generate(self, ntoys) :
    self.dist = SamplingDistribution(ntoys)
    for k in range(0, ntoys) :
      self.progress(k, ntoys)
      data = self.generate_data()
      self.dist.samples[k] = self.clsb(self.test_hypo.poi, data)
    return self.dist
