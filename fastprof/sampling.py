import pyhf
import json
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import os
from scipy.interpolate import InterpolatedUnivariateSpline


# -------------------------------------------------------------------------
class SamplingDistribution :
  def __init__(self, nentries = 0, ncols = 0) :
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

  def cl(self, acl) :
    return np.searchsorted(self.samples, acl)/len(self.samples)

  def quantile(self, fraction=None, sigma=None) :
    if fraction == None :
      if sigma != None :
        fraction = scipy.stats.norm.cdf(sigma)
      else :
        raise ValueError("Should provide exactly one of 'fraction' or 'sigma'.")
    if fraction < 0 or fraction > 1 :
      raise ValueError('Invalid fraction value %g, should be between 0 and 1.' % fraction)
    index = int(len(self.samples)*fraction)
    #print('Quantile: frac = %g -> index = %d -> cl = %g' % (fraction, index, self.samples[index]))
    return self.samples[index]


# -------------------------------------------------------------------------
class SamplesBase :
  def __init__(self, mus) :
    self.mus = mus

  def bands(self, max_sigma) :
    bds = {}
    for i in range (-max_sigma, max_sigma+1) :
      bds[i]  = np.array([ self.quantile(hypo, sigma=i) for hypo in self.mus ])
    return bds

  def plot_bands(self, max_sigma = 2, canvas=plt) :
    colors = [ 'k', 'g', 'y', 'c', 'b' ]
    bands = self.bands(max_sigma)
    for i in reversed(range(1, max_sigma + 1)) :
      canvas.fill_between(self.mus, bands[+i], bands[-i], color=colors[i])
    canvas.plot(self.mus, bands[0], 'k--')


# -------------------------------------------------------------------------
class Samples (SamplesBase) :
  def __init__(self, mus, sampler = None, file_root = '') :
    super().__init__(mus)
    self.sampler = sampler
    self.file_root = file_root
    self.dists = {}
    
  def file_name(self, mu, ext = '') :
    return self.file_root + '_%g' % mu + ext
  
  def generate_and_save(self, ntoys, break_lock = False, sort_before_saving = True) :
    for mu in self.mus :
      if os.path.exists(self.file_name(mu, '.lock')) and not break_lock : 
        print('Samples for mu = %g already being produced, skipping' % mu)
        continue
      if os.path.exists(self.file_name(mu, '.npy')) and not break_lock :
        print('Samples for mu = %g already produced, just loading (%d samples from %s)' % (mu, ntoys, self.file_name(mu, '.npy')))
        self.dists[mu] = SamplingDistribution(ntoys)
        self.dists[mu].load(self.file_name(mu, '.npy'))
        continue
      print('Processing sampling distribution for POI = %g' % mu)
      with open(self.file_name(mu, '.lock'), 'w') as f :
        f.write(str(os.getpid()))
      self.dists[mu] = self.sampler.generate(mu, ntoys)
      self.dists[mu].save(self.file_name(mu), sort_before_saving=sort_before_saving)
      print('Done')
      os.remove(self.file_name(mu, '.lock'))
    return self
  
  def load(self) :
    fname = self.file_name(mu, '.npy')
    for mu in self.mus :
      try:
        samples = np.load(fname)
      except:
        raise FileNotFoundError('File %s not found, for samples at mu = %g' % (fname, mu))
      self.dists[mu] = samples
    return self

  def generate(self, ntoys) : # on the fly generation -- for fast cases only!
    for mu in self.mus :
      print('Creating sampling distribution for %g' % mu)
      self.dists[mu] = self.sampler.generate(mu, ntoys)
      self.dists[mu].sort()
      print('Done')
    return self

  def cl(self, mu, acl) :
    try:
      samples = self.dists[mu]
    except:
      raise KeyError('No sample available for mu = %g' % mu)
    return samples.cl(acl)

  def quantile(self, mu, fraction=None, sigma=None) :
    try:
      samples = self.dists[mu]
    except:
      raise KeyError('No sample available for mu = %g' % mu)
    return samples.quantile(fraction, sigma)


# -------------------------------------------------------------------------
class CLsSamples (SamplesBase) :
  def __init__(self, clsb_samples, cl_b_samples) :
    super().__init__(clsb_samples.mus)
    self.clsb = clsb_samples
    self.cl_b = cl_b_samples
    
  def generate_and_save(self, ntoys) :
    print('Processing CL_{s+b} sampling distributions for POI values %s' % str(self.mus))
    self.clsb.generate_and_save(ntoys)
    print('Processing CL_b sampling distributions for POI values %s' % str(self.mus))
    self.cl_b.generate_and_save(ntoys)
    return self
  
  def load(self) :
    self.clsb.load()
    self.cl_b.load()
    return self
  
  def generate(self, ntoys) : # on the fly generation -- for fast cases only!
    print('Creating CL_{s+b} sampling distributions for POI values %s' % str(self.mus))
    self.clsb.generate(ntoys)
    print('Creating CL_b sampling distributions for POI values %s' % str(self.mus))
    self.cl_b.generate(ntoys)
    return self

  def cl(self, mu, acl) :
    clsb = self.clsb.cl(mu, acl)
    cl_b = self.cl_b.cl(mu, acl)
    #print('Sampling CLs = %g/%g = %g' % (clsb, cl_b, clsb/cl_b))
    return clsb/cl_b if cl_b > 0 else 1

  def quantile(self, mu, fraction=None, sigma=None, cl_b = 0.5) :
    return self.clsb.quantile(mu, fraction, sigma)/cl_b

  def bands(self, max_sigmas) :
    cls_samples = Samples(self.mus)
    for mu in self.mus :
      ns = len(self.cl_b.dists[mu].samples)
      sd = SamplingDistribution(ns)
      for i, acl in enumerate(self.cl_b.dists[mu].samples) :
        sd.samples[i] = self.cl(mu, acl)
      sd.sort()
      cls_samples.dists[mu] = sd
    return cls_samples.bands(max_sigmas)
