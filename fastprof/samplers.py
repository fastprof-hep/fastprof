import pyhf
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from abc import abstractmethod
from scipy.interpolate import InterpolatedUnivariateSpline

from .core import Parameters
from .test_statistics import QMu
from .sampling import SamplingDistribution
from .minimizers import NPMinimizer, OptiMinimizer, ScanMinimizer

# -------------------------------------------------------------------------
class Sampler :
  def __init__(self, model, gen_mu = None, print_freq = 1000) :
    self.model = model
    self.gen_mu = gen_mu
    self.freq = print_freq
  def progress(self, k, ntoys) :
    if k % self.freq == 0 : print('-- Processing iteration %d of %d' % (k, ntoys))
  @abstractmethod
  def generate(self, mu, ntoys) :
     pass


# -------------------------------------------------------------------------
class ScanSampler (Sampler) :
  def __init__(self, model, scan_mus, gen_mu = False, print_freq = 1000) :
    super().__init__(model, gen_mu, print_freq)
    self.scan_mus = scan_mus
    
  def generate(self, mu, ntoys) :
    gen_hypo = Parameters(self.gen_mu if self.gen_mu != None else mu, 0, 0)
    self.dist = SamplingDistribution(ntoys)
    for k in range(0, ntoys) :
      self.progress(k, ntoys)
      data = self.model.generate_data(gen_hypo)
      tmu, min_pos = ScanMinimizer(data, self.scan_mus).tmu(mu)
      q = QMu(test_mu = mu, tmu = tmu, best_mu = min_pos)
      self.dist.samples[k] = q.asymptotic_cl()
    return self.dist


# -------------------------------------------------------------------------
class OptiSampler (Sampler) :
  def __init__(self, model, mu0, bounds, method = 'scalar', gen_mu = None, print_freq = 1000, debug=False, niter=1) :
    super().__init__(model, gen_mu, print_freq)
    self.mu0 = mu0
    self.bounds = bounds
    self.method = method
    self.debug = debug
    self.niter = niter
    
  def generate(self, mu, ntoys) :
    gen_hypo = self.model.expected_pars(self.gen_mu if self.gen_mu != None else mu)
    self.dist = SamplingDistribution(ntoys)
    if self.debug :
      columns = [ 'cl', 'tmu', 'mu_hat', 'free_nll', 'hypo_nll', 'nfev' ]
      columns.extend( [ 'free_' + a for a in self.model.alphas ] )
      columns.extend( [ 'free_' + b for b in self.model.betas  ] )
      columns.extend( [ 'free_' + c for c in self.model.gammas ] )
      columns.extend( [ 'hypo_' + a for a in self.model.alphas ] )
      columns.extend( [ 'hypo_' + b for b in self.model.betas  ] )
      columns.extend( [ 'hypo_' + c for c in self.model.gammas ] )
      self.debug_data = pd.DataFrame(columns=columns)
    for k in range(0, ntoys) :
      self.progress(k, ntoys)
      success = False
      while not success :
        if self.debug : print('DEBUG: iteration %d generating data for hypo %s.' % (k, str(gen_hypo)))
        data = self.model.generate_data(gen_hypo)
        mini = OptiMinimizer(data, self.mu0, self.bounds, self.method, self.niter)
        tmu, min_pos = mini.tmu(mu)
        if self.debug : 
          print('DEBUG: fitting data with mu0 = %g and range = %g, %g -> t = %g, mu_hat = %g.' %(self.mu0, *self.bounds, tmu, min_pos))
          print(mini.min_pars)
          print(mini.hypo_pars)
        if tmu != None :
          success = True
        else :
          print('Minimization failed at toy iteration %d, repeating it.' % k)
      q = QMu(test_mu = mu, tmu = tmu, best_mu = min_pos)
      #print(mu, tmu, min_pos, '->', q.value(), q.asymptotic_cl())
      self.dist.samples[k] = q.asymptotic_cl()
      if self.debug :
        self.debug_data.at[k, 'cl'      ] = q.asymptotic_cl()
        self.debug_data.at[k, 'tmu'     ] = tmu
        self.debug_data.at[k, 'mu_hat'  ] = min_pos
        self.debug_data.at[k, 'free_nll'] = mini.free_nll
        self.debug_data.at[k, 'hypo_nll'] = mini.hypo_nll
        self.debug_data.at[k, 'nfev'    ] = mini.nfev
        for i, a in enumerate(self.model.alphas) :
          self.debug_data.at[k, 'free_' + a] = mini.free_pars.alphas[i]
          self.debug_data.at[k, 'hypo_' + a] = mini.hypo_pars.alphas[i]
        for i, b in enumerate(self.model.betas) :
          self.debug_data.at[k, 'free_' + b] = mini.free_pars.betas [i]
          self.debug_data.at[k, 'hypo_' + b] = mini.hypo_pars.betas [i]
        for i, c in enumerate(self.model.gammas) :
          self.debug_data.at[k, 'free_' + c] = mini.free_pars.gammas[i]
          self.debug_data.at[k, 'hypo_' + c] = mini.hypo_pars.gammas[i]
        if tmu > 1000 : data.save('data_%d.json' % k)
    return self.dist


# -------------------------------------------------------------------------
class PyhfSampler (Sampler) :
  def __init__(self, model, nbins, n_np, gen_mu = None, print_freq = 1000) :
    super().__init__(model, gen_mu, print_freq)
    self.nbins = nbins
    self.n_np = n_np
    
  def generate_data(self, mu) :
    data = np.zeros(self.nbins + self.n_np)
    params = [mu] + self.n_np*[0]
    expected = self.model.expected_data(params)
    for i in range(0, self.nbins) :
      data[i] = np.random.poisson(expected[i])
    for i in range(0, self.n_np) :
      data[self.nbins + i] = np.random.normal(0, 1)
    return data
  
  def clsb(self, mu, data) :
    return pyhf.infer.hypotest(mu, data, self.model, return_tail_probs = True)[1][0]

  def generate(self, mu, ntoys) :
    gen_mu = self.gen_mu if self.gen_mu != None else mu
    self.dist = SamplingDistribution(ntoys)
    for k in range(0, ntoys) :
      self.progress(k, ntoys)
      data = self.generate_data(gen_mu)
      self.dist.samples[k] = self.clsb(mu, data)
    return self.dist
