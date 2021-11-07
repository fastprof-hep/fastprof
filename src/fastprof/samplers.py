"""Definition of the algorithms that generate sampling distributions

*Sampler* classes generate a sampling distribution for a given
hypothesis. They are called by the :meth:`Samples` class defined in
the :mod:`sampling.py` module, which handle the logistics of
selecting hypotheses and storing the results; and they themselves
make use of the minimization algorithms in the :mod:`minimizers.py`
module.

The classes derive from the :class:`Sampler` base, which
implements the :meth:`Sampler.generate` method that performs the
generation. This includes the production of a toy dataset, which
is a generic operation.

The p-value computation can however be performed using any of the
different minimizer classes in :mod:`minimizers.py`, so this is
part is delegated to the abstract method :meth:`Sampler.compute` which is
reimplemented in derived classes corresponding to the minimization
algorithms. So far the following are defined:

* :class:`OptiSampler` class, using the :class:`OptiMinimizer`
  minimization algorithm.

* :class:`ScanSampler` class, using the :class:`SCanMinimizer`
  minimization algorithm.
"""

import json
import numpy as np
import pandas as pd
import math
from abc import abstractmethod

import os, sys
import datetime
from timeit import default_timer as timer

from .core import Model, Parameters, Data
from .test_statistics import QMu, QMuTilda
from .sampling import SamplingDistribution
from .minimizers import OptiMinimizer, ScanMinimizer

# -------------------------------------------------------------------------
class Sampler :
  """Base class for sampler algorithms

  Provides the interface for sample generation, including the
  generation of pseudo-datasets. The p-value computation is
  performed in the meth:`compute` abstract method which is
  reimplemented in derived classes.

  Attributes:
    model (Model) : the linear model used for generation
    gen_hypo (Parameters) : the model parameter values defining the
                            generation hypothesis
    print_freq (int) : the interval at which to print out progress messages
    max_tries (int) : the maximum allowed number of retries when the
                      generation fails
    ntries (int) : the current number of generation trials (reset to 0 after
                   each successful generation).
    dist (SamplingDistribution) : the sampling distribution produced by
                                  meth:`generate`.
  """
  def __init__(self, model : Model, gen_hypo : Parameters = None, print_freq : int = 1000, max_tries : int = 20) :
    """Initialize the Sampler object

    Args:
      model : the linear model used for generation
      gen_hypo : the model parameter values defining the
                 generation hypothesis
      print_freq : the interval at which to print out progress messages
      max_tries : the maximum allowed number of retries when the
                  generation fails
    """
    self.model = model
    self.gen_hypo = model.expected_pars(gen_hypo) if isinstance(gen_hypo, (int, float)) else gen_hypo
    self.print_freq = print_freq
    self.max_tries = max_tries
    self.ntries = 0

  def progress(self, k : int, ntoys : int, descr : str = '') :
    """Small utility method to print out a progress indicator

    Args:
      k     : the index of the current toy generation iteration
      ntoys : the total number of toys to generate
      descr : a comment string to append to the message
    """
    if k % self.print_freq == 0 or k == ntoys - 1 :
      #print('-- Processing iteration %d of %d %s' % (k, ntoys, descr))
      sys.stderr.write('\rProcessing iteration %d of %d %s' % (k if k < ntoys - 1 else ntoys, ntoys, descr))

  def generate(self, ntoys, hypo_descr : str = None) -> SamplingDistribution :
    """Generate a specified number of samples

    The function generates toy datasets and performs the computation
    in the :meth:`compute` abstract method. If the computation fails,
    another toy dataset is generated and another attempt is performed.
    This is repeated until either a computation succeeds, or the maximum
    number of attemps specified by `max_tries` is reached, in which case
    a null result is returned. The process is repeated until `ntoys`
    samples are generated in this way.

    Args:
      ntoys : the total number of samples to generate
      hypo_descr : a description string for the hypothesis
    Returns:
      `self.dist`, the generated sampling distribution
    """
    hypo_descr = ' ' + hypo_descr if hypo_descr is not None else ''
    print('Generating POI hypothesis %s%s, starting at %s. Full gen hypo = ' % (str(self.gen_hypo.pois), hypo_descr, str(datetime.datetime.now())))
    start_time = timer()
    print(str(self.gen_hypo))
    self.dist = SamplingDistribution(ntoys)
    ntotal = 0
    for k in range(0, ntoys) :
      if k % self.print_freq == 0 or k == ntoys - 1 :
        descr = 'in hypo %s%s [generation rate = %5.1f Hz]' % (str(self.gen_hypo.pois), hypo_descr, k/(timer() - start_time) if k > 0 else 0)
        self.progress(k, ntoys, descr)
      success = False
      self.ntries = 0
      while not success :
        if self.debug : print('DEBUG: iteration %d generating data for hypo %s.' % (k, str(self.gen_hypo.pois)))
        data = self.model.generate_data(self.gen_hypo)
        ntotal += 1
        self.ntries += 1
        result = self.compute(data, k)
        if result != None :
          success = True
        elif self.ntries < self.max_tries :
          print('Processing toy iteration %d failed, repeating it.' % k)
        else :
          print('Processing toy iteration %d failed, and max number of tries (%d) reached -- returning null result.' % (k, self.max_tries))
      self.dist.samples[k] = result
    end_time = timer()
    sys.stderr.write('\n')
    print('Done with POI hypothesis %s, end time %s. Generated %d good toys (%d total), elapsed time = %g s' % (str(self.gen_hypo.pois), datetime.datetime.now(), ntoys, ntotal, end_time - start_time))
    return self.dist

  @abstractmethod
  def compute(self, data : Data, toy_iter : int) -> float :
    """Perform the result computation

    Abstract method to perform the computation of the result
    Args:
      data : the toy dataset on which to perform the computation
      toy_iter : the index of the toy generation iteration
    Returns:
      the result, or `None` if the computation fails
    """
    pass


# -------------------------------------------------------------------------
class OptiSampler (Sampler) :
  """Sampler algorithms using :class:`OptiMinimizer` minimization

  Provides an implementation of :class:`Sampler` in which the
  p-value computation in the meth:`compute` method is performed
  using :class:`OptiMinimizer`. In turn, this uses algorithms from
  :mod`scipy.minimize` to perform the minimization.

  A list of parameter bounds defined by :class:`ParBounds` objects
  can be passed to the class. These are applied on the best-fit
  values of the parameters in both of the fits that define `tmu`,
  and the generation is repeated if the bounds are not verified.

  The generated samples are asymptotic p-values for the hypothesis test in
  each pseudo-dataset. This is equivalent to other sampling quantities such as
  profile-likelihood values, and has the advantage of being bounded to the
  interval :math:`[0,1]`, with a flat distribution if the asymptotic
  approximation is valid.

  Attributes:
    test_hypo (Parameters) : the model parameter values defining the
                 tested hypothesis. This is usually identical
                 to the generation hypothesis, but can differ
                 e.g. for :math:`CL_b` computations
    method (str) : the minimization method from :mod:`scipy.minimize` to use
    niter (int) : number of iterations to perform in the minimization (see
             :class:`OptiMinimizer` for details)
    bounds (list) : Bounds to apply on the best-fit parameters, in the form of
              a list of :class:`ParBounds` objects defining cuts on one
              parameter.
    floor (float) : minimal value of the per-bin event yields to use in
                    minimization (see :class:`OptiMinimizer` for details)
    tmu_Amu (float) : value of the generation-hypothesis Asimov value of
                    `qmu` to use in `qmutilda` computations
    tmu_A0 (float) : value of the zero-hypothesis Asimov value of
                    `qmu` to use in `qmutilda` computations
    use_qtilda (bool) : if True, use the :class:`QMuTilda` test statistic,
                        otherwise use :class:`QMu`.
    debug (bool) : if True, print out debug information
    debug_data (pd.DataFrame) : dataframe containing the debug information.
    minimizer (OptiMinimizer) : the minimizer object
  """
  def __init__(self, model, test_hypo : Parameters, gen_hypo : Parameters = None, method : str = 'scalar', niter : int = 1, bounds : list = [],
               print_freq : int = 1000, max_tries : int = 20, tmu_Amu : float = None, tmu_A0 : float = None, floor : float = 1E-7,
               debug : bool = False) :
    """Initialize the OptiSampler object

    Args:
      test_hypo : the model parameter values defining the
                   tested hypothesis. This is usually identical
                   to the generation hypothesis, but can differ
                   e.g. for :math:`CL_b` computations
      gen_hypo  : the model parameter values defining the
                  generation hypothesis (default: None, in which
                  case the test hypothesis is used.)
      method : the minimization method from :mod:`scipy.minimize` to use
      niter  : number of iterations to perform in the minimization (see
               :class:`OptiMinimizer` for details)
      bounds : Bounds to apply on the best-fit parameters, in the form of
               a list of :class:`ParBounds` objects defining cuts on one
               parameter.
      print_freq : the interval at which to print out progress messages
      max_tries  : the maximum allowed number of retries when the
                   generation fails
      floor : minimal value of the per-bin event yields to use in
              minimization (see :class:`OptiMinimizer` for details)
      tmu_Amu : value of the generation-hypothesis Asimov value of
              `qmu` to use in `qmutilda` computations
      tmu_A0 : value of the zero-hypothesis Asimov value of
              `qmu` to use in `qmutilda` computations
      debug : if True, print out debug information
    """
    super().__init__(model, gen_hypo, print_freq)
    self.test_hypo = model.expected_pars(test_hypo) if isinstance(test_hypo, (int, float)) else test_hypo
    if self.gen_hypo == None : self.gen_hypo = Parameters.clone(self.test_hypo)
    self.bounds = bounds
    self.method = method
    self.niter = niter
    self.floor = floor
    self.tmu_Amu = tmu_Amu
    self.tmu_A0 = tmu_A0
    self.use_qtilda = True if tmu_Amu != None and tmu_A0 != None else False
    self.debug = debug
    self.debug_data = pd.DataFrame()
    self.minimizer = OptiMinimizer(self.method, niter=self.niter, floor=self.floor)
    if self.debug : self.minimizer.debug = 2

  def compute(self, data, toy_iter) :
    """Compute the asymptotic p-value

    Compute the asymptotic p-value for a given toy dataset.
    The computation uses either the :class:`QMuTilda` or the
    :class:`QMu` test statistic, depending on the initialization
    parameters.
    The fits are performed using the :class:`OptiMinimizer`
    minimization algorith,.
    If the best-fit parameters fail the bounds specified at
    initialization, the computation fails (leading in general to
    another generation attempt, see :class:`Sampler`).
    Debug data is produced if the debug flag was passed at
    initialization.

    Args:
      data : the toy dataset on which to perform the computation
      toy_iter : the index of the toy generation iteration
    Returns:
      the computed p-value, or `None` if the computation fails
    """
    tmu = self.minimizer.tmu(self.test_hypo.pars, data, self.gen_hypo)
    if tmu < 0 :
      print('Warning: tmu <= 0 at toy iteration %d' % toy_iter)
      if self.debug and self.minimizer.tmu_debug < -10 :
        os.makedirs('data', exist_ok=True)
        data.save('data/debug_data_neg_tmu_%d.json' % toy_iter)
      return None
    for bound in self.bounds :
      if not bound.test(self.minimizer.free_deltas) :
        print('Warning: free fit parameters below fail bound %s' % str(bound))
        print(self.minimizer.free_pars)
        return None
      if not bound.test(self.minimizer.hypo_deltas) :
        print('Warning: hypothesis fit parameters below fail bound %s' % str(bound))
        print(self.minimizer.hypo_pars)
        return None
    if self.debug :
      print('DEBUG: fitting data with mu0 = %g and range = %g, %g -> t = %g, mu_hat = %g.' % (self.mu0, *self.poi_bounds, tmu, self.minimizer.min_poi))
      print(self.minimizer.free_pars)
      print(self.minimizer.hypo_pars)
    poi = self.test_hypo.pars[list(self.test_hypo.pars.keys())[0]]
    if self.use_qtilda :
      q = QMuTilda(test_poi = poi, tmu = tmu, best_poi = self.minimizer.min_pois[0], tmu_Amu = self.tmu_Amu, tmu_A0 = self.tmu_A0)
    else :
      q = QMu(test_poi = poi, tmu = tmu, best_poi = self.minimizer.min_pois[0])
    if self.debug :
      if self.debug_data.shape[0] == 0 :
        columns = [ 'pv', 'tmu', 'mu_hat', 'free_nll', 'hypo_nll', 'nfev', 'ntries' ]
        columns.extend( [ 'free_' + p for p in self.model.nps ] )
        columns.extend( [ 'hypo_' + p for p in self.model.nps ] )
        self.debug_data = pd.DataFrame(columns=columns)
      self.debug_data.at[toy_iter, 'pv'      ] = q.asymptotic_pv()
      self.debug_data.at[toy_iter, 'tmu'     ] = tmu
      self.debug_data.at[toy_iter, 'mu_hat'  ] = self.minimizer.min_poi
      self.debug_data.at[toy_iter, 'free_nll'] = self.minimizer.free_nll
      self.debug_data.at[toy_iter, 'hypo_nll'] = self.minimizer.hypo_nll
      self.debug_data.at[toy_iter, 'nfev'    ] = self.minimizer.nfev
      self.debug_data.at[toy_iter, 'ntries'  ] = self.ntries
      for i, p in enumerate(self.model.nps) :
        self.debug_data.at[toy_iter, 'free_' + p] = self.minimizer.free_pars[p]
        self.debug_data.at[toy_iter, 'hypo_' + p] = self.minimizer.hypo_pars[p]
        self.debug_data.at[toy_iter, 'aux_'  + p] = data.aux_obs[i]
      data.save('data/debug_data_%d.json' % toy_iter)
    return q.asymptotic_pv()



# -------------------------------------------------------------------------
class ScanSampler (Sampler) :
  """Sampler algorithms using :class:`ScanMinimizer` minimization

  Provides an implementation of :class:`Sampler` in which the
  p-value computation in the meth:`compute` method is performed
  using :class:`ScanMinimizer`, which implements a simple scan
  over the POIs.

  Attributes:
    test_hypo (Parameters) : the model parameter values defining the
                 tested hypothesis. This is usually identical
                 to the generation hypothesis, but can differ
                 e.g. for :math:`CL_b` computations
    scan_mus (list) : list of POI values over which to perform the scan
    tmu_Amu (float) : value of the generation-hypothesis Asimov value of
                    `qmu` to use in `qmutilda` computations
    tmu_A0 (float) : value of the zero-hypothesis Asimov value of
                    `qmu` to use in `qmutilda` computations
    use_qtilda (bool) : if True, use the :class:`QMuTilda` test statistic,
                        otherwise use :class:`QMu`.
  """
  def __init__(self, model, test_hypo, scan_mus, gen_hypo = None, print_freq = 1000, tmu_Amu = None, tmu_A0 = None) :
    """Initialize the ScanSampler object

    Args:
      test_hypo : the model parameter values defining the
                   tested hypothesis. This is usually identical
                   to the generation hypothesis, but can differ
                   e.g. for :math:`CL_b` computations
      scan_mus : list of POI values over which to perform the scan
      gen_hypo  : the model parameter values defining the
                  generation hypothesis (default: None, in which
                  case the test hypothesis is used.)
      print_freq : the interval at which to print out progress messages
      tmu_Amu : value of the generation-hypothesis Asimov value of
              `qmu` to use in `qmutilda` computations
      tmu_A0 : value of the zero-hypothesis Asimov value of
              `qmu` to use in `qmutilda` computations
    """
    super().__init__(model, gen_hypo, print_freq)
    self.test_hypo = model.expected_pars(test_hypo) if isinstance(test_hypo, (int, float)) else test_hypo
    if self.gen_hypo == None : self.gen_hypo = Parameters.clone(self.test_hypo)
    self.scan_mus = scan_mus
    self.tmu_Amu = tmu_Amu
    self.tmu_A0 = tmu_A0
    self.use_qtilda = True if tmu_Amu != None and tmu_A0 != None else False

  def compute(self, data, toy_iter) :
    """Compute the asymptotic p-value

    Compute the asymptotic p-value for a given toy dataset.
    The computation uses either the :class:`QMuTilda` or the
    :class:`QMu` test statistic, depending on the initialization
    parameters.
    The fits are performed using the :class:`ScanMinimizer`
    minimization algorith,.

    Args:
      data : the toy dataset on which to perform the computation
      toy_iter : the index of the toy generation iteration
    Returns:
      the computed p-value
    """
    opti = ScanMinimizer(self.scan_mus)
    tmu = opti.tmu(self.test_hypo, data, self.test_hypo)
    if self.use_qtilda :
      q = QMuTilda(test_poi = self.test_hypo.pois[0], tmu = tmu, best_poi = opti.min_poi, tmu_Amu = self.tmu_Amu, tmu_A0 = self.tmu_A0)
    else :
      q = QMu(test_poi = self.test_hypo.pois[0], tmu = tmu, best_poi = opti.min_poi)
    return q.asymptotic_pv()



# -------------------------------------------------------------------------
class LimitSampler (Sampler) :
  """A sampler algorithms that directy produces parameter limits

  Produces an ensemble of limits in a given generation hypothesis.
  This is not necessary in the default workflow for setting
  :math:`CL_s` upper limits, since the :math:`CL_b` toys can be used
  to define a background-only limit ensemble. However this functionality
  can be useful in other contexts.

  Attributes:
    limit_calc (Calculator) : A test statistic calculator class
       (see :mod:`tools.py`)
    cl (float) : the CL at which to compute the limit
"""

  def __init__(self, model : Model, gen_hypo : Parameters, limit_calc : 'Calculator', cl : float = 0.95, print_freq : int = 1000) :
    """Initialize the LimitSampler object

    Args:
      gen_hypo  : the model parameter values defining the
                  generation hypothesis
      limit_calc : a test statistic calculator object
       (see :mod:`tools.py`)
      cl : the CL at which to compute the limit
      print_freq : the interval at which to print out progress messages
    """
    super().__init__(model, gen_hypo, print_freq)
    self.limit_calc = limit_calc
    self.cl = cl

  def compute(self, data, toy_iter) :
    """Compute the limit

    Compute the limit for a given toy dataset.
    The computation uses the limit calculator class
    and the CL specified at initialization.

    Args:
      data : the toy dataset on which to perform the computation
      toy_iter : the index of the toy generation iteration
    Returns:
      the computed limit
    """
    self.limit_calc.fill_fast_results(data = data, pv_key = 'fast_pv')
    return self.limit_calc.limit(pv_key = 'fast_pv', cl=self.cl)
