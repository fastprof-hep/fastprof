"""Interface classes for distribution sampling

The module defines the interface classes for producing
sampling distributions. The actual generation is delegated
to `Sampler` classes (see the :mod:`samplers.py` module), so
the purpose of the classes defined here is to deal with the
logistics of generation: file save/load operations,
generating missing distributions, etc.

The module also defines the :class:`SamplingDistribution` class
which stores a sampling distribution.
"""

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
  """Class describing a sampling distribution

  The class contains a numpy array with the samples,
  usually asymptptic p-values from hypothesis tests.
  It defines functions for file save/load operations,
  value lookup, and the computation of quantiles.

  Attributes:
    samples  (np.arrray) : the sampling distribution
    filename (str)       : the filename from which the
                           distribution was loaded
  """
  def __init__(self, nentries : int = 0) :
    """Initialize the `SamplingDistribution` object

    Args:
      nentries : the number of samples to reserve in the array
    """
    self.samples = np.zeros(nentries)
    self.filename = None

  def sort(self) :
    """Sort the samples in increasing order

    Sorting is mandatory to allow fast lookup of a value in the distribution,
    and is usually performed immediately after sample generation.
    """
    self.samples = np.sort(self.samples)

  def load(self, filename : str) -> 'SamplingDistribution' :
    """Load the distribution from file

    Args:
      filename : name of a numpy file to load from
    Returns:
      self
    """
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
    self.filename = filename
    return self

  def save(self, filename : str, sort_before_saving : bool = True) -> 'SamplingDistribution' :
    """Save the distribution to file

    If `sort_before_saving` is set to `True`, the distribution
    is sorted before saving.

    Args:
      filename : name of a numpy file to save to
      sort_before_saving : if `True`, sort before saving
    Returns:
      self
    """
    if sort_before_saving : self.sort()
    np.save(filename, self.samples)

  def pv(self, apv : float, with_error : bool = False) -> float :
    """Return the quantile of a given value

    In the nominal case where the samples are asymptotic p-values,
    and `apv` is a given asymptotic p-value, the result is the
    corresponding sampling p-value.

    The p-value is computed using the *mid-p-value* method to 
    properly handle the Poisson case without systematics: if the
    distribution is made up of delta functions, and `apv` is right
    on a peak, then exactly half the peak is counted.

    If `with_error` is `True`, return also the sampling error,
    computed as a binomial uncertainty.

    Args:
      apv : an asymptotic p-value
      with_error : if `True`, return also the sampling error (default: False).
    Returns:
      the quantile (i.e. usually the sampling p-value)
    """
    nbelow_lft = np.searchsorted(self.samples, apv, 'left')
    nbelow_rgt = np.searchsorted(self.samples, apv, 'right')
    nbelow = (nbelow_lft + nbelow_rgt)/2
    ntot = len(self.samples)
    # Add +1 terms to avoid uncertainty going to 0 at the edges of the distribution
    return (nbelow/ntot, math.sqrt((nbelow+1)*(ntot - nbelow + 1)/ntot)/ntot) if with_error else nbelow/ntot

  def quantile(self, fraction : float = None, nsigmas : float = None) -> float :
    """Return the position of a given quantile of the distribution

    The quantile can be specified either directly as `fraction`,
    or as a number of Gaussian sigmas as `nsigmas`.

    Args:
      fraction : the quantile fraction
      nsigmas  : the quantile fraction in number of sigmas
    Returns:
      the p-value position of the quantile
    """
    if fraction == None :
      if nsigmas != None :
        fraction = scipy.stats.norm.cdf(nsigmas)
      else :
        raise ValueError("Should provide exactly one of 'fraction' or 'nsigmas'.")
    if fraction < 0 or fraction > 1 :
      raise ValueError('Invalid fraction value %g, should be between 0 and 1.' % fraction)
    index = int(len(self.samples)*fraction)
    #print('Quantile: frac = %g -> index = %d -> pv = %g' % (fraction, index, self.samples[index]))
    return self.samples[index]

  def cut(self, min_val : float = None, max_val : float = None) -> 'SamplingDistribution' :
    """Apply a selection to the distribution

    Removes the lower and/or upper edge of the distribution, which
    can be sensitive to numerical artefacts.

    Args:
      min_val : the value below which to truncate the distribution
      max_val : the value above which to truncate the distribution
    Returns:
      self
    """
    if min_val : self.samples = self.samples[self.samples >= min_val]
    if max_val : self.samples = self.samples[self.samples <= max_val]
    return self


# -------------------------------------------------------------------------
class SamplesBase :
  """Base class for Samples classes

  A *Samples* class is designed to manage multiple sampling distributions
  at various hypothesis points -- used for instance to set limits or
  scan over a parameter space.

  The base class defined here implement common operations across the different
  classes. At the moment this reduces to the :meth:`bands` method which plots
  expected variation bands for 1D results.

  Attributes:
    hypos (list) : a list of POI hypotheses for which to generate samples
  """

  def __init__(self, hypos) :
    """Initialize the `SamplesBase` object

    Args:
      hypos : a list of hypothesis values at which to generate samples
    """
    self.hypos = hypos

  @abstractmethod
  def bands(self, max_sigma : int) -> dict :
    """Compute expected band positions

    Args:
      max_sigma : the highest-order band to compute. The bands or order
                  -max_sigma ... -max_sigma will be computed
    Returns :
      A dict mapping band order to a list of band positions for each hypothesis
    """
    pass

  def plot_bands(self, max_sigma = 2, canvas=plt) :
    """Plot expected bands

    Args:
      max_sigma : the highest-order band to show. The bands or order
                  -max_sigma ... -max_sigma will be drawn
      canvas : the matplotlib figure on which to draw (default: plt)
    """
    colors = [ 'k', 'g', 'y', 'c', 'b' ]
    hypos = [ hypo.pars[list(hypo.pars.keys())[0]] for hypo in self.hypos ]
    bands = self.bands(max_sigma)
    for i in reversed(range(1, max_sigma + 1)) :
      canvas.fill_between(hypos, bands[+i], bands[-i], color=colors[i])
    canvas.plot(hypos, bands[0], 'k--')


# -------------------------------------------------------------------------
class Samples (SamplesBase) :
  """Class handling the generation of sampling distributions

  Provides an implementation of :class:`SampleBase`, which handles
  the generation of a sampling distribution for each hypothesis.

  The main workflow is to use the class to generate sampling distribution
  and save them to disk, so that they can be reused in later sessions instead
  of regenerating them. The file locations are built from a root file name,
  with suffixes corresponding to each hypothesis. Files are in the numpy (.npy)
  format, each containing a numpy array of samples.
  The sampling is delegated to :class:`sampler` algorithms: one is stored for
  each hypothesis and is called upon when samples need to be generated.

  The generated samples are asymptotic p-values for the hypothesis test in
  each pseudo-dataset. This is equivalent to other sampling quantities such as
  profile-likelihood values, and has the advantage of being bounded to the
  interval :math:`[0,1]`, with a flat distribution if the asymptotic
  approximation is valid.

  Attributes:
    samplers (Sampler) : a list of sampler objects, matched to the list of
       hypotheses and used to generate the sampling distribution at each point.
    file_root (str) : the root file name to use for writing out sampling distributions
    dists (dict) : dictionary mapping hypotheses to sampling distributions.
  """
  def __init__(self, samplers : list = [], file_root : str = '', hypos : list = []) :
    """Initialize the `Samples` object

    Args:
      samplers : a list of sampler objects used for generating samples,
                 matching the hypothesis list provided in `hypos`
      file_root : the root file name to use for writing out sampling distributions
      hypos : a list of hypothesis values at which to generate samples
    """
    if len(samplers) > 0 and len(hypos) > 0 :
      raise ValueError('Should specify either samplers or hypotheses, but not both.')
    if len(samplers) == 0 and len(hypos) == 0 :
      raise ValueError('Should specify either samplers or hypotheses.')
    if len(samplers) > 0  and len(hypos) == 0 : hypos = [ sampler.test_hypo for sampler in samplers ]
    super().__init__(hypos)
    self.samplers = samplers
    self.file_root = file_root
    self.dists = {}

  def file_name(self, hypo : float, suffix : str = '') -> str :
    """Build the normalized file name for a hypothesis value

    Args:
      hypo : a hypothesis value
      suffix : an optional suffix (default: '')
    Returns :
      a normalized file name, without extension
    """
    poi_vals = hypo.pars
    fields = [ self.file_root ] + [ '%g' % val for val in poi_vals.values() ]
    return '_'.join(fields) + suffix

  def generate_and_save(self, ntoys : int, break_locks : bool = False, sort_before_saving : bool = True) -> 'Samples' :
    """Generate and save the sampling distribution for all hypothesis values

    Generate a sampling distribution at each hypothesis value, and store it in a
    numpy file. A lock file is created during the generation, so that multiple
    jobs can run in parallel and generate samples at different hypotheses.
    The `break_locks` option ignores those locks, in case they were left by
    interrupted jobs.

    Args:
      ntoys : number of samples to generate at each hypothesis
      break_locks : if True, break existing lock files and generate
                    the corresponding distributions (default: False)
      sort_before_saving : if True, sort the sampling distributions
          before saving, for faster reuse later.
    Returns :
      self
    """
    if not self.samplers :
      raise ValueError('Cannot generate as no samplers were specified.')
    for i, hypo, sampler in zip(range(0, len(self.hypos)), self.hypos, self.samplers) :
      if os.path.exists(self.file_name(hypo, '.npy')) :
        print('Samples for hypo = %s already produced, just loading (%d samples from %s)' % (str(hypo), ntoys, self.file_name(hypo, '.npy')))
        self.dists[hypo] = SamplingDistribution(ntoys)
        self.dists[hypo].load(self.file_name(hypo, '.npy'))
        continue
      if os.path.exists(self.file_name(hypo, '.lock')) and not break_locks :
        print('Samples for hypo = %s already being produced, skipping' % str(hypo))
        continue
      print('Processing sampling distribution for hypo %s' % str(hypo))
      with open(self.file_name(hypo, '.lock'), 'w') as f :
        f.write(str(os.getpid()))
      self.dists[hypo] = sampler.generate(ntoys, '(%d of %d)' % (i+1, len(self.hypos)))
      self.dists[hypo].save(self.file_name(hypo), sort_before_saving=sort_before_saving)
      if hasattr(sampler, 'debug_data') and sampler.debug_data.shape[0] != 0 : sampler.debug_data.to_csv(self.file_name(hypo, '_debug.csv'))
      print('Done')
      os.remove(self.file_name(hypo, '.lock'))
    return self

  def load(self) -> 'Samples' :
    """Load existing sampling distribution for all hypothesis values

    Returns :
      self
    """
    for hypo in self.hypos :
      try:
        self.dists[hypo] = SamplingDistribution()
        self.dists[hypo].load(self.file_name(hypo, '.npy'))
      except Exception as inst :
        print('Cannot load from file %s, for samples at hypo = %s, exception below:' % (fname, str(hypo)))
        raise(inst)
    return self

  def generate(self, ntoys) :
    """Generate sampling distribution for all hypothesis values

    Generate a sampling distribution at each hypothesis value. Unlike
    :meth:`generate_and_save` above, this doesn't save the results
    for further use, so should be useful mainly for simple situations
    where on-the-fly generation can be performed quickly.

    Args:
      ntoys : number of samples to generate at each hypothesis
    Returns :
      self
    """
    if not self.samplers :
      raise ValueError('Cannot generate as no samplers were specified.')
    for hypo, sampler in zip(self.hypos, self.samplers) :
      print('Creating sampling distribution for %s' % str(hypo))
      self.dists[hypo] = self.sampler.generate(ntoys)
      self.dists[hypo].sort()
      print('Done')
    return self

  def pv(self, hypo : float, apv : float, with_error : bool = False) -> float :
    """Compute the sampling p-value

    Uses the sampling distribution at the specified hypothesis
    value to convert an asymptotic p-value to a calibrated
    sampling p-value.

    If `with_error` is specified, the return value is a (value, uncertainty)
    pair, with the uncertainty due to the limited size of the sampling
    distribution (see :meth:`SamplingDistribution.pv`)

    Args:
      hypo : a hypothesis value
      apv : an asymptotic p-value for this hypothesis
      with_error : if `True`, returns also the uncertainty
    Returns :
      the asymptotic p-value (with uncertainties, if `with_error` is `True`)
    """
    try:
      samples = self.dists[hypo]
    except Exception as inst :
      print('No sample available for hypo = %s, available samples are %s' % (str(hypo), self.dists.keys()))
      raise(inst)
    return samples.pv(apv, with_error)

  def quantile(self, hypo : float, fraction : float = None, nsigmas : float = None) -> float :
    """Compute the p-value position of a given quantile

    The quantile can be specified either directly as `fraction`,
    or as a number of Gaussian sigmas as `nsigmas`.
    (see :meth:`SamplingDistribution.pv`)

    Args:
      fraction : the quantile fraction
      nsigmas  : the quantile fraction in number of sigmas
    Returns:
      the p-value position of the quantile
    """
    try:
      samples = self.dists[hypo]
    except Exception as inst :
      print('No sample available for hypo = %s' % str(hypo))
      raise(inst)
    return samples.quantile(fraction, nsigmas)

  def bands(self, max_sigma : int) -> 'Samples' :
    """Compute expected band positions for all hypotheses

    Args:
      max_sigma : the highest-order band to compute. The bands or order
                  -max_sigma ... -max_sigma will be computed
    Returns :
      A dict mapping band order to a list of band positions for each hypothesis
    """
    bds = {}
    for i in range (-max_sigma, max_sigma+1) :
      bds[i]  = np.array([ self.quantile(hypo, nsigmas=i) for hypo in self.hypos ])
    return bds

  def cut(self, min_val : float = None, max_val : float = None) -> 'Samples' :
    """Apply a selection to the distributions for all hypotheses

    Removes the lower and/or upper edge of the distribution, which
    can be sensitive to numerical artefacts.

    Args:
      min_val : the value below which to truncate the distribution
      max_val : the value above which to truncate the distribution
    Returns:
      self
    """
    for hypo in self.hypos : self.dists[hypo].cut(min_val, max_val)
    return self


# -------------------------------------------------------------------------
class CLsSamples (SamplesBase) :
  """Class handling the generation of :math:`CL_s` sampling distributions

  Provides an implementation of :class:`SampleBase` which handles
  the generation of a sampling distribution for :math:`CL_s` limits.

  Since computing :math:`CL_s` involves both the usual :math:`CL_{s+b}`
  calculation and also that of :math:`CL_b`, the class contains two
  :math:`Samples` instances, one for each calculation.
  Two sets of samples are generated, one for the tested hypotheses for
  :math:`CL_{s+b}` and one under the POI=0 hypothesis for :math:`CL_b`.

  Attributes:
    clsb (Samples) : :math:`Samples` object for the :math:`CL_{s+b}` computation
    cl_b (Samples) : :math:`Samples` object for the :math:`CL_b` computation
  """
  def __init__(self, clsb_samples : Samples, cl_b_samples : Samples) :
    """Initialize the CLsSamples object

    Args:
      clsb_samples : the :class:`Samples` object handling :math:`CL_{s+b}` sampling
      cl_b_samples : the :class:`Samples` object handling :math:`CL_b` sampling
    """
    super().__init__(clsb_samples.hypos)
    self.clsb = clsb_samples
    self.cl_b = cl_b_samples

  def generate_and_save(self, ntoys : int, break_locks : bool = False, sort_before_saving : bool = True) -> 'CLsSamples' :
    """Generate and save the sampling distributions

    Generate sand save the sampling distributions for the :math:`CL_{s+b}`
    and :math:`CL_s` computations (see :meth:`Samples.generate_and_save`)

    Args:
      ntoys : number of samples to generate at each hypothesis
      break_locks : if True, break existing lock files and generate
                    the corresponding distributions (default: False)
      sort_before_saving : if True, sort the sampling distributions
          before saving, for faster reuse later.
    Returns :
      self
    """
    print('Processing CL_{s+b} sampling distributions for hypos: ')
    for hypo in self.hypos : print(str(hypo))
    self.clsb.generate_and_save(ntoys, break_locks, sort_before_saving)
    print('Processing CL_b sampling distributions for hypos: ')
    for hypo in self.hypos : print(str(hypo))
    self.cl_b.generate_and_save(ntoys, break_locks, sort_before_saving)
    return self

  def load(self) -> 'CLsSamples' :
    """Load existing sampling distributions

    Loads the distributions for the :math:`CL_{s+b}` and
    :math:`CL_s` computations (see :meth:`Samples.load`)

    Returns :
      self
    """
    self.clsb.load()
    self.cl_b.load()
    return self

  def generate(self, ntoys : int) -> 'CLsSamples' :
    """Generate sampling distribution

    Generate the sampling distributions for the :math:`CL_{s+b}` and
    :math:`CL_s` computations (see :meth:`Samples.generate`)

    Args:
      ntoys : number of samples to generate at each hypothesis
    Returns :
      self
    """
    print('Creating CL_{s+b} sampling distributions for hypo values %s' % str(self.hypos))
    self.clsb.generate(ntoys)
    print('Creating CL_b sampling distributions for hypo values %s' % str(self.hypos))
    self.cl_b.generate(ntoys)
    return self

  def pv(self, hypo : float, apv : float, with_error : bool = False) -> 'CLsSamples' :
    """Compute the :math:`CL_s` value

    Compute the :math:`CL_s` value as the ratio of :math:`CL_{s+b}` and
    :math:`CL_s`.

    If `with_error` is specified, the return value is a (value, uncertainty)
    pair, with the uncertainty propagated from the statistical uncertainties
    in the sampling distributions (see :meth:`SamplingDistribution.pv`)

    Args:
      hypo : a hypothesis value
      apv : an asymptotic p-value for this hypothesis
      with_error : if `True`, returns also the uncertainty
    Returns :
      the asymptotic p-value (with uncertainties, if `with_error` is `True`)
    """
    clsb = self.clsb.pv(hypo, apv, with_error)
    cl_b = self.cl_b.pv(hypo, apv, with_error)
    #print('Sampling CLs = %g/%g = %g' % (clsb, cl_b, clsb/cl_b))
    if not with_error : return clsb/cl_b if cl_b > 0 else 1
    if cl_b[0] <= 0 : return (1,1)
    if clsb[0] <= 0 : return (0,clsb[1]/cl_b[0])
    return (clsb[0]/cl_b[0], clsb[0]/cl_b[0]*math.sqrt((clsb[1]/clsb[0])**2 + (cl_b[1]/cl_b[0])**2))

  def quantile(self, hypo, fraction=None, nsigmas=None, cl_b=0.5) :
    """Compute the :math:`CL_s` position of a given quantile

    The quantile is computed from that of the :math:`CL_{s+b}`
    distribution, assuming a fixed value of :math:`CL_b` : this
    is appropriate for expected results with POI=0, for which
    :math:`CL_b=0.5`.

    The quantile can be specified either directly as `fraction`,
    or as a number of Gaussian sigmas as `nsigmas`.
    (see :meth:`SamplingDistribution.pv`)

    Args:
      fraction : the quantile fraction
      nsigmas  : the quantile fraction in number of sigmas
      clb      : the assumed value of :math:`CL_b` (default: 0.5)
    Returns:
      the p-value position of the quantile
    """
    return self.clsb.quantile(hypo, fraction, nsigmas)/cl_b

  def bands(self, max_sigmas) :
    """Compute expected :math:`CL_s` band positions

    The computation is perfomed on the ensemble of :math:`CL_b`
    results, and therefore corresponds to the POI=0 case.

    Args:
      max_sigma : the highest-order band to compute. The bands or order
                  -max_sigma ... -max_sigma will be computed
    Returns :
      A dict mapping band order to a list of band positions for each hypothesis
    """
    cls_samples = Samples(hypos=self.hypos)
    for hypo in self.hypos :
      sd = SamplingDistribution(len(self.cl_b.dists[hypo].samples))
      for i, apv in enumerate(self.cl_b.dists[hypo].samples) : sd.samples[i] = self.pv(hypo, apv)
      sd.sort()
      cls_samples.dists[hypo] = sd
    return cls_samples.bands(max_sigmas)

  def cut(self, min_val = None, max_val = None) :
    """Apply a selection to the distributions

    Applies the selection to both the :math:`CL_{s+b}` and :math:`CL_b` distributions.

    Args:
      min_val : the value below which to truncate the distribution
      max_val : the value above which to truncate the distribution
    Returns:
      self
    """
    self.clsb.cut(min_val, max_val)
    self.cl_b.cut(min_val, max_val)
    return self
