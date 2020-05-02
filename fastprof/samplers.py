import pyhf
import json
import matplotlib.pyplot as plt
import numpy as np
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
  def __init__(self, model, mu0, bounds, method = 'scalar', gen_mu = None, print_freq = 1000) :
    super().__init__(model, gen_mu, print_freq)
    self.mu0 = mu0
    self.bounds = bounds
    self.method = method
    
  def generate(self, mu, ntoys) :
    gen_hypo = self.model.expected_pars(self.gen_mu if self.gen_mu != None else mu)
    self.dist = SamplingDistribution(ntoys)
    for k in range(0, ntoys) :
      self.progress(k, ntoys)
      success = False
      while not success :
        data = self.model.generate_data(gen_hypo)
        tmu, min_pos = OptiMinimizer(data, self.mu0, self.bounds, self.method).tmu(mu)
        if tmu != None :
          success = True
        else :
          print('Minimization failed at toy iteration %d, repeating it.' % k)
      q = QMu(test_mu = mu, tmu = tmu, best_mu = min_pos)
      #print(mu, tmu, min_pos, '->', q.value(), q.asymptotic_cl())
      self.dist.samples[k] = q.asymptotic_cl()
    return self.dist


# -------------------------------------------------------------------------
class DebuggingScanSampler (Sampler) :
  def __init__(self, model, scan_mus, pyhf_model, gen_mu = None, print_freq = 1000) :
    super().__init__(model, gen_mu, print_freq)
    self.scan_mus = scan_mus
    self.pyhf_model = pyhf_model

  def generate(self, mu, ntoys) :
    # debug : each toy stores data_bin1, .., data_binN, aux_NP1, ... aux_NPN, fitval_mu, fitval_NP1 ... fitval_NPN, profA, profB, cl
    gen_hypo = Parameters(self.gen_mu if self.gen_mu != None else mu, 0, 0)
    n_dat = self.model.nbins + self.model.nsyst
    n_np = self.model.n_nps
    self.debug_info = SamplingDistribution(ntoys, n_dat + 3*n_np + 6)
    for k in range(0, ntoys) :
      self.progress(k, ntoys)
      data = self.model.generate_data(gen_hypo)
      minimizer = ScanMinimizer(data, self.scan_mus)
      nll_min, min_pos = minimizer.minimize(True)
      nll_hypo = NPMinimizer(mu, data).profile_nll()
      q = QMu(test_mu = mu, tmu = 2*(nll_hypo - nll_min), best_mu = min_pos)
      pyhf_data = data.export_pyhf_data(self.pyhf_model)
      pars, val = pyhf.infer.mle.fit(pyhf_data, self.pyhf_model, return_fitted_val=True) # return [mhat, ahat, bhat], nll_min
      pyhf_clsb = pyhf.infer.hypotest(mu, pyhf_data, self.pyhf_model, return_tail_probs = True)[1][0]
      pars, val = pyhf.infer.mle.fixed_poi_fit(self.scan_mus[minimizer.min_idx], pyhf_data, self.pyhf_model, return_fitted_val=True)
      self.debug_info.samples[k, :n_dat] = pyhf_data # data
      self.debug_info.samples[k, n_dat:n_dat + 1] = min_pos  # fast best-fit mu
      self.debug_info.samples[k, n_dat + 1:n_dat + n_np + 1] = np.concatenate((minimizer.min_pars.alphas, minimizer.min_pars.betas, minimizer.min_pars.gammas)) # fast best-fit NPs
      self.debug_info.samples[k, n_dat + n_np + 1:n_dat + n_np + 2] = q.asymptotic_cl() # fast best-fit CL
      self.debug_info.samples[k, n_dat + n_np + 2:n_dat + 2*n_np + 3] = pars # pyhf best-fit pars 
      self.debug_info.samples[k, n_dat + 2*n_np + 3:n_dat + 2*n_np + 4] = pyhf_clsb # pyhf CLs+b @ mu=min
      self.debug_info.samples[k, n_dat + 2*n_np + 4:n_dat + 3*n_np + 5] = pars # pyhf best-fit pars @ mu=min sample
    return self.debug_info


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
