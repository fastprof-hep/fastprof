import pyhf
import json
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import fastprof
from scipy.interpolate import InterpolatedUnivariateSpline

class SamplingDistribution :
  def __init__(self, nentries, ncols = 0) :
    self.samples = np.zeros((nentries, ncols)) if ncols > 0 else np.zeros(nentries)

  def sort(self) :
    self.samples = np.sort(self.samples)
  
  def load(self, filename) :
    try:
      self.samples = np.load(filename)
    except:
      raise IOError('Could not load samples from file %s.' % filename)

  def save(self, filename, sort_before_saving = True) :
    if sort_before_saving : self.sort()
    np.save(filename, self.samples)


class FastSampler :
  def __init__(self, model, scan_mus, do_CLb = False) :
    self.model = model
    self.scan_mus = scan_mus
    self.do_CLb = do_CLb
    
  def generate(self, mu, ntoys) :
    gen_hypo = fastprof.Parameters(0 if self.do_CLb else mu, 0, 0)
    self.dist = SamplingDistribution(ntoys)
    for k in range(0, ntoys) :
      if k % 1000 == 0 : print(k)
      data = self.model.generate_data(gen_hypo)
      minimizer = fastprof.ScanMinimizer(data, self.scan_mus)
      nll_min, min_pos = minimizer.minimize()
      nll_hypo = fastprof.NPMinimizer(mu, data).profile_nll()
      q = fastprof.QMu(2*(nll_hypo - nll_min), mu, min_pos)
      self.dist.samples[k] = q.asymptotic_cl()
    return self.dist


class DebuggingFastSampler :
  def __init__(self, model, scan_mus, pyhf_model, do_CLb = False) :
    self.model = model
    self.scan_mus = scan_mus
    self.pyhf_model = pyhf_model
    self.do_CLb = do_CLb

  def generate(self, mu, ntoys) :
    # debug : each toy stores data_bin1, .., data_binN, aux_NP1, ... aux_NPN, fitval_mu, fitval_NP1 ... fitval_NPN, profA, profB, cl
    gen_hypo = fastprof.Parameters(0 if self.do_CLb else mu, 0, 0)
    n_dat = self.model.n_bins() + self.model.n_syst()
    n_np = self.model.n_syst()
    self.debug_info = SamplingDistribution(ntoys, n_dat + 3*n_np + 6)
    for k in range(0, ntoys) :
      if k % 1000 == 0 : print(k)
      data = self.model.generate_data(gen_hypo)
      minimizer = fastprof.ScanMinimizer(data, self.scan_mus)
      nll_min, min_pos = minimizer.minimize(True)
      nll_hypo = fastprof.NPMinimizer(mu, data).profile_nll()
      q = fastprof.QMu(2*(nll_hypo - nll_min), mu, min_pos)
      pyhf_data = data.export_pyhf_data(self.pyhf_model)
      self.debug_info.samples[k, :n_dat] = pyhf_data # 0,1,2,3 : data
      self.debug_info.samples[k, n_dat:n_dat + n_np + 2] = [ min_pos, minimizer.min_pars.alpha, minimizer.min_pars.beta, q.asymptotic_cl() ] # 4,5,6,7 : fast best-fit pars & cls
      pars, val = pyhf.infer.mle.fit(pyhf_data, self.pyhf_model, return_fitted_val=True) # return [mhat, ahat, bhat], nll_min
      self.debug_info.samples[k, n_dat + n_np + 2:n_dat + 2*n_np + 3] = pars # 8,9,10 : best-fit pars 
      pyhf_clsb = pyhf.infer.hypotest(mu, pyhf_data, self.pyhf_model, return_tail_probs = True)[1][0]
      self.debug_info.samples[k, n_dat + 2*n_np + 3:n_dat + 2*n_np + 4] = pyhf_clsb # 11: CLs+b @ mu=min
      pars, val = pyhf.infer.mle.fixed_poi_fit(scan_mus[minimizer.min_idx], pyhf_data, self.pyhf_model, return_fitted_val=True)
      self.debug_info.samples[k, n_dat + 2*n_np + 4:n_dat + 3*n_np + 5] = pars # 11,12,13 : full best-fit pars @ mu=min sample
    return self.debug_info


class PyhfSampler :
  def __init__(self, model, n_bins, n_np, do_CLb = False) :
    self.model = model
    self.n_bins = n_bins
    self.n_np = n_np
    self.do_CLb = do_CLb
    
  def generate_data(self, mu) :
    data = np.zeros(self.n_bins + self.n_np)
    params = [mu] + self.n_np*[0]
    expected = self.model.expected_data(params)
    for i in range(0, self.n_bins) :
      data[i] = np.random.poisson(expected[i])
    for i in range(0, self.n_np) :
      data[self.n_bins + i] = np.random.normal(0, 1)
    return data
  
  def clsb(self, mu, data) :
    return pyhf.infer.hypotest(mu, data, self.model, return_tail_probs = True)[1][0]

  def generate(self, mu, ntoys) :
    gen_mu = 0 if self.do_CLb else mu
    self.dist = SamplingDistribution(ntoys)
    for k in range(0, ntoys) :
      if k % 1000 == 0 : print(k)
      data = self.generate_data(gen_mu)
      self.dist.samples[k] = self.clsb(mu, data)
    return self.dist


class Samples :
  def __init__(self, sampler, file_root) :
    self.sampler = sampler
    self.file_root = file_root
    self.samples = {}
    
  def file_name(self, mu, ext = '') :
    return self.file_root + '_%g' % mu + ext
  
  def generate_and_save(self, mus, ntoys, break_lock = False) :
    for mu in mus :
      if os.path.exists(self.file_name(mu, '.lock')) and not break_lock : 
        print('Samples for mu = %g already being produced, skipping' % mu)
        continue
      if os.path.exists(self.file_name(mu, '.npy')) and not break_lock :
        print('Samples for mu = %g already produced, just loading' % mu)
        self.samples[mu] = SamplingDistribution(ntoys)
        self.samples[mu].load(self.file_name(mu, '.npy'))
        continue
      print('Creating sampling distribution for %g' % mu)
      with open(self.file_name(mu, '.lock'), 'w') as f :
        f.write(str(os.getpid()))
      self.samples[mu] = self.sampler.generate(mu, ntoys)
      self.samples[mu].save(self.file_name(mu))
      print('Done')
      os.remove(self.file_name(mu, '.lock'))
    return self
  
  def load(self, mus) :
    fname = self.file_name(mu, '.npy')
    for mu in mus :
      try:
        samples = np.load(fname)
      except:
        print('File %s not found, for samples at mu = %g' % (fname, mu))
        return {}
      self.samples[mu] = samples
    return self

  def generate(self, mus, ntoys) : # on the fly generation -- for fast cases only!
    for mu in mus :
      print('Creating sampling distribution for %g' % mu)
      self.samples[mu] = self.sampler.generate(mu, ntoys)
      self.samples[mu].sort()
      print('Done')
    return self

  def cl(self, acl, mu) :
    try:
      samples = self.samples[mu].samples
    except:
      raise KeyError('No sample available for mu = %g', mu)
    return np.searchsorted(samples, acl)/len(samples)


class CLsSamples :
  def __init__(self, clsb_samples, cl_b_samples) :
    self.clsb = clsb_samples
    self.cl_b = cl_b_samples
    
  def generate_and_save(self, mus, ntoys) :
    self.clsb.generate_and_save(mus, ntoys)
    self.cl_b.generate_and_save(mus, ntoys)
    return self
  
  def load(self, mus) :
    self.clsb.load(mus)
    self.cl_b.load(mus)
    return self
  
  def generate(self, mus, ntoys) : # on the fly generation -- for fast cases only!
    self.clsb.generate(mus, ntoys)
    self.cl_b.generate(mus, ntoys)
    return self

  def cl(self, acl, mu) :
    clsb = self.clsb.cl(acl, mu)
    cl_b = self.cl_b.cl(acl, mu)
    return clsb/cl_b
