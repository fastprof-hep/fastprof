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
    nbefore = self.samples.shape[0]
    try:
      self.samples = np.load(filename)
    except:
      raise IOError('Could not load samples from file %s.' % filename)
    nafter =  self.samples.shape[0]
    if nbefore > 0 and nafter < nbefore :
      raise IOError('File %s did not contain enough samples (expected %d, got %d).' % (filename, nbefore, nafter))
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
      if k % 1000 == 0 : print('-- Processing iteration %d of %d' % (k, ntoys))
      data = self.model.generate_data(gen_hypo)
      nll_min, min_pos = fastprof.ScanMinimizer(data, self.scan_mus).minimize()
      nll_hypo = fastprof.NPMinimizer(mu, data).profile_nll()
      q = fastprof.QMu(2*(nll_hypo - nll_min), mu, min_pos)
      self.dist.samples[k] = q.asymptotic_cl()
    return self.dist

class OptiSampler :
  def __init__(self, model, x0, bounds, method = 'scalar', do_CLb = False) :
    self.model = model
    self.do_CLb = do_CLb
    self.x0 = x0
    self.bounds = bounds
    self.method = method
    
  def generate(self, mu, ntoys) :
    gen_hypo = self.model.expected_pars(0 if self.do_CLb else mu)
    self.dist = SamplingDistribution(ntoys)
    for k in range(0, ntoys) :
      if k % 1000 == 0 : print('-- Processing iteration %d of %d' % (k, ntoys))
      success = False
      while not success :
        data = self.model.generate_data(gen_hypo)
        nll_min, min_pos = fastprof.OptiMinimizer(data, self.x0, self.bounds, self.method).minimize()
        if nll_min != None : 
          success = True
        else :
          print('Minimization failed at toy iteration %d, repeating it.' % k)
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
    n_dat = self.model.nbins + self.model.nsyst
    n_np = self.model.n_nps
    self.debug_info = SamplingDistribution(ntoys, n_dat + 3*n_np + 6)
    for k in range(0, ntoys) :
      if k % 1000 == 0 : print('Processing iteration %d of %d' % (k+1, ntoys))
      data = self.model.generate_data(gen_hypo)
      minimizer = fastprof.ScanMinimizer(data, self.scan_mus)
      nll_min, min_pos = minimizer.minimize(True)
      nll_hypo = fastprof.NPMinimizer(mu, data).profile_nll()
      q = fastprof.QMu(2*(nll_hypo - nll_min), mu, min_pos)
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


class PyhfSampler :
  def __init__(self, model, nbins, n_np, do_CLb = False) :
    self.model = model
    self.nbins = nbins
    self.n_np = n_np
    self.do_CLb = do_CLb
    
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
  
  def generate_and_save(self, mus, ntoys, break_lock = False, sort_before_saving = True) :
    for mu in mus :
      if os.path.exists(self.file_name(mu, '.lock')) and not break_lock : 
        print('Samples for mu = %g already being produced, skipping' % mu)
        continue
      if os.path.exists(self.file_name(mu, '.npy')) and not break_lock :
        print('Samples for mu = %g already produced, just loading (%d samples from %s)' % (mu, ntoys, self.file_name(mu, '.npy')))
        self.samples[mu] = SamplingDistribution(ntoys)
        self.samples[mu].load(self.file_name(mu, '.npy'))
        continue
      print('Processing sampling distribution for POI = %g' % mu)
      with open(self.file_name(mu, '.lock'), 'w') as f :
        f.write(str(os.getpid()))
      self.samples[mu] = self.sampler.generate(mu, ntoys)
      self.samples[mu].save(self.file_name(mu), sort_before_saving=sort_before_saving)
      print('Done')
      os.remove(self.file_name(mu, '.lock'))
    return self
  
  def load(self, mus) :
    fname = self.file_name(mu, '.npy')
    for mu in mus :
      try:
        samples = np.load(fname)
      except:
        raise FileNotFoundError('File %s not found, for samples at mu = %g' % (fname, mu))
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
      raise KeyError('No sample available for mu = %g' % mu)
    return np.searchsorted(samples, acl)/len(samples)


class CLsSamples :
  def __init__(self, clsb_samples, cl_b_samples) :
    self.clsb = clsb_samples
    self.cl_b = cl_b_samples
    
  def generate_and_save(self, mus, ntoys) :
    print('Processing CL_{s+b} sampling distributions for POI values %s' % str(mus))
    self.clsb.generate_and_save(mus, ntoys)
    print('Processing CL_b sampling distributions for POI values %s' % str(mus))
    self.cl_b.generate_and_save(mus, ntoys)
    return self
  
  def load(self, mus) :
    self.clsb.load(mus)
    self.cl_b.load(mus)
    return self
  
  def generate(self, mus, ntoys) : # on the fly generation -- for fast cases only!
    print('Creating CL_{s+b} sampling distributions for POI values %s' % str(mus))
    self.clsb.generate(mus, ntoys)
    print('Creating CL_b sampling distributions for POI values %s' % str(mus))
    self.cl_b.generate(mus, ntoys)
    return self

  def cl(self, acl, mu) :
    clsb = self.clsb.cl(acl, mu)
    cl_b = self.cl_b.cl(acl, mu)
    return clsb/cl_b
