import pyhf
import json
import matplotlib.pyplot as plt
import numpy as np
import math
import os
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
    #print('Sampling CLs = %g/%g = %g' % (clsb, cl_b, clsb/cl_b))
    return clsb/cl_b
