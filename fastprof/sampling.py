import json
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
from scipy.interpolate import InterpolatedUnivariateSpline
import os
from abc import abstractmethod


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
      print('Could not load samples from file %s, exception below:' % filename)
      raise(inst)
      raise IOError
    nafter =  self.samples.shape[0]
    if nbefore > 0 and nafter < nbefore :
      raise IOError('File %s did not contain enough samples (expected %d, got %d).' % (filename, nbefore, nafter))
    if nbefore > 0 and nafter > nbefore :
      print('Info: File %s contained more samples than expected (expected %d, got %d), only using the first %d.' % (filename, nbefore, nafter, nbefore))
      self.samples = self.samples[:nbefore]
    return self

  def save(self, filename, sort_before_saving = True) :
    if sort_before_saving : self.sort()
    np.save(filename, self.samples)

  def pv(self, apv, with_error = False) :
    nbelow = np.searchsorted(self.samples, apv)
    return (nbelow/len(self.samples), math.sqrt(nbelow+1)/len(self.samples)) if with_error else nbelow/len(self.samples)

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

  def cut(self, min_val = None, max_val = None) :
    if min_val : self.samples = self.samples[self.samples >= min_val]
    if max_val : self.samples = self.samples[self.samples <= max_val]
    return self


# -------------------------------------------------------------------------
class SamplesBase :
  def __init__(self, pois) :
    self.pois = pois

  @abstractmethod
  def bands(self, max_sigma) :
    pass

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
    if len(samplers) > 0  and len(pois) == 0 : pois = [ sampler.test_hypo.pois[0] for sampler in samplers ]
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
        print('Cannot load from file %s, for samples at POI = %g, exception below:' % (fname, poi))
        raise(inst)
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

  def pv(self, poi, apv, with_error = False) :
    try:
      samples = self.dists[poi]
    except Exception as inst :
      print('No sample available for POI = %g, available samples are %s' % (poi, self.dists.keys()))
      raise(inst)
    return samples.pv(apv, with_error)

  def quantile(self, poi, fraction=None, sigma=None) :
    try:
      samples = self.dists[poi]
    except Exception as inst :
      print('No sample available for POI = %g' % poi)
      raise(inst)
    return samples.quantile(fraction, sigma)

  def bands(self, max_sigma) :
    bds = {}
    for i in range (-max_sigma, max_sigma+1) :
      bds[i]  = np.array([ self.quantile(poi, sigma=i) for poi in self.pois ])
    return bds

  def cut(self, min_val = None, max_val = None) :
    for poi in self.pois : self.dists[poi].cut(min_val, max_val)
    return self


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

  def pv(self, poi, apv, with_error = False) :
    clsb = self.clsb.pv(poi, apv, with_error)
    cl_b = self.cl_b.pv(poi, apv, with_error)
    #print('Sampling CLs = %g/%g = %g' % (clsb, cl_b, clsb/cl_b))
    if not with_error : return clsb/cl_b if cl_b > 0 else 1
    if cl_b[0] <= 0 : return (1,1)
    if clsb[0] <= 0 : return (0,clsb[1]/cl_b[0])
    return (clsb[0]/cl_b[0], clsb[0]/cl_b[0]*math.sqrt((clsb[1]/clsb[0])**2 + (cl_b[1]/cl_b[0])**2))

  def quantile(self, poi, fraction=None, sigma=None, cl_b=0.5) :
    return self.clsb.quantile(poi, fraction, sigma)/cl_b

  def bands(self, max_sigmas) :
    cls_samples = Samples(pois=self.pois)
    for poi in self.pois :
      sd = SamplingDistribution(len(self.cl_b.dists[poi].samples))
      for i, apv in enumerate(self.cl_b.dists[poi].samples) : sd.samples[i] = self.pv(poi, apv)
      sd.sort()
      cls_samples.dists[poi] = sd
    return cls_samples.bands(max_sigmas)

  def cut(self, min_val = None, max_val = None) :
    self.clsb.cut(min_val, max_val)
    self.cl_b.cut(min_val, max_val)
    return self
