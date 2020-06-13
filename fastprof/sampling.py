import json
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import os
from scipy.interpolate import InterpolatedUnivariateSpline


# -------------------------------------------------------------------------
class SamplingDistribution :
  def __init__(self, nentries = 0) :
    self.samples = np.zeros(nentries)

  def sort(self) :
    self.samples = np.sort(self.samples)

  def load(self, filename) :
    nbefore = self.samples.shape[0]
    try:
      self.samples = np.load(filename)
    except Exception as inst :
      print(inst)
      raise IOError('Could not load samples from file %s.' % filename)
    nafter =  self.samples.shape[0]
    if nbefore > 0 and nafter < nbefore :
      raise IOError('File %s did not contain enough samples (expected %d, got %d).' % (filename, nbefore, nafter))
    return self

  def save(self, filename, sort_before_saving = True) :
    if sort_before_saving : self.sort()
    np.save(filename, self.samples)

  def pv(self, apv) :
    return np.searchsorted(self.samples, apv)/len(self.samples)

  def quantile(self, fraction=None, sigma=None) :
    if fraction == None :
      if sigma != None :
        fraction = scipy.stats.norm.cdf(sigma)
      else :
        raise ValueError("Should provide exactly one of 'fraction' or 'sigma'.")
    if fraction < 0 or fraction > 1 :
      raise ValueError('Invalid fraction value %g, should be between 0 and 1.' % fraction)
    index = int(len(self.samples)*fraction)
    #print('Quantile: frac = %g -> index = %d -> pv = %g' % (fraction, index, self.samples[index]))
    return self.samples[index]


# -------------------------------------------------------------------------
class SamplesBase :
  def __init__(self, pois) :
    self.pois = pois

  def bands(self, max_sigma) :
    bds = {}
    for i in range (-max_sigma, max_sigma+1) :
      bds[i]  = np.array([ self.quantile(poi, sigma=i) for poi in self.pois ])
    return bds

  def plot_bands(self, max_sigma = 2, canvas=plt) :
    colors = [ 'k', 'g', 'y', 'c', 'b' ]
    bands = self.bands(max_sigma)
    for i in reversed(range(1, max_sigma + 1)) :
      canvas.fill_between(self.pois, bands[+i], bands[-i], color=colors[i])
    canvas.plot(self.pois, bands[0], 'k--')


# -------------------------------------------------------------------------
class Samples (SamplesBase) :
  def __init__(self, samplers = [], file_root = '', pois = []) :
    if len(samplers) > 0 and len(pois) > 0 :
      raise ValueError('Should specify either samplers or hypotheses, but not both.')
    if len(samplers) == 0 and len(pois) == 0 :
      raise ValueError('Should specify either samplers or hypotheses.')
    if len(samplers) > 0  and len(pois) == 0 : pois = [ sampler.test_hypo.poi for sampler in samplers ]
    super().__init__(pois)
    self.samplers = samplers
    self.file_root = file_root
    self.dists = {}
    
  def file_name(self, poi, ext = '') :
    return self.file_root + '_%g' % poi + ext
  
  def generate_and_save(self, ntoys, break_locks = False, sort_before_saving = True) :
    if not self.samplers :
      raise ValueError('Cannot generate as no samplers were specified.')
    for poi, sampler in zip(self.pois, self.samplers) :
      if os.path.exists(self.file_name(poi, '.npy')) :
        print('Samples for POI = %g already produced, just loading (%d samples from %s)' % (poi, ntoys, self.file_name(poi, '.npy')))
        self.dists[poi] = SamplingDistribution(ntoys)
        self.dists[poi].load(self.file_name(poi, '.npy'))
        continue
      if os.path.exists(self.file_name(poi, '.lock')) and not break_locks :
        print('Samples for POI = %g already being produced, skipping' % poi)
        continue
      print('Processing sampling distribution for POI = %g' % poi)
      with open(self.file_name(poi, '.lock'), 'w') as f :
        f.write(str(os.getpid()))
      self.dists[poi] = sampler.generate(ntoys)
      self.dists[poi].save(self.file_name(poi), sort_before_saving=sort_before_saving)
      if hasattr(sampler, 'debug_data') and sampler.debug_data.shape[0] != 0 : sampler.debug_data.to_csv(self.file_name(poi, '_debug.csv'))
      print('Done')
      os.remove(self.file_name(poi, '.lock'))
    return self
  
  def load(self) :
    for poi in self.pois :
      try:
        self.dists[poi] = SamplingDistribution()
        self.dists[poi].load(self.file_name(poi, '.npy'))
      except Exception as inst :
        print(inst)
        raise FileNotFoundError('File %s not found, for samples at POI = %g' % (fname, poi))
    return self

  def generate(self, ntoys) : # on the fly generation -- for fast cases only!
    if not self.samplers :
      raise ValueError('Cannot generate as no samplers were specified.')
    for poi, sampler in zip(self.pois, self.samplers) :
      print('Creating sampling distribution for %g' % poi)
      self.dists[poi] = self.sampler.generate(ntoys)
      self.dists[poi].sort()
      print('Done')
    return self

  def pv(self, poi, apv) :
    try:
      samples = self.dists[poi]
    except Exception as inst :
      print(inst)
      print('Available samples are', self.dists.keys())
      raise KeyError('No sample available for POI = %g' % poi)
    return samples.pv(apv)

  def quantile(self, poi, fraction=None, sigma=None) :
    try:
      samples = self.dists[poi]
    except Exception as inst :
      print(inst)
      raise KeyError('No sample available for POI = %g' % poi)
    return samples.quantile(fraction, sigma)

# -------------------------------------------------------------------------
class CLsSamples (SamplesBase) :
  def __init__(self, clsb_samples, cl_b_samples) :
    super().__init__(clsb_samples.pois)
    self.clsb = clsb_samples
    self.cl_b = cl_b_samples
    
  def generate_and_save(self, ntoys, break_locks = False, sort_before_saving = True) :
    print('Processing CL_{s+b} sampling distributions for POI values %s' % str(self.pois))
    self.clsb.generate_and_save(ntoys, break_locks, sort_before_saving)
    print('Processing CL_b sampling distributions for POI values %s' % str(self.pois))
    self.cl_b.generate_and_save(ntoys, break_locks, sort_before_saving)
    return self
  
  def load(self) :
    self.clsb.load()
    self.cl_b.load()
    return self
  
  def generate(self, ntoys) : # on the fly generation -- for fast cases only!
    print('Creating CL_{s+b} sampling distributions for POI values %s' % str(self.pois))
    self.clsb.generate(ntoys)
    print('Creating CL_b sampling distributions for POI values %s' % str(self.pois))
    self.cl_b.generate(ntoys)
    return self

  def pv(self, poi, apv) :
    clsb = self.clsb.pv(poi, apv)
    cl_b = self.cl_b.pv(poi, apv)
    #print('Sampling CLs = %g/%g = %g' % (clsb, cl_b, clsb/cl_b))
    return clsb/cl_b if cl_b > 0 else 1

  def quantile(self, poi, fraction=None, sigma=None, cl_b=0.5) :
    return self.clsb.quantile(poi, fraction, sigma)/cl_b

  def bands(self, max_sigmas) :
    cls_samples = Samples(pois=self.pois)
    for poi in self.pois :
      ns = len(self.cl_b.dists[poi].samples)
      sd = SamplingDistribution(ns)
      for i, apv in enumerate(self.cl_b.dists[poi].samples) :
        sd.samples[i] = self.pv(poi, apv)
      sd.sort()
      cls_samples.dists[poi] = sd
    return cls_samples.bands(max_sigmas)
