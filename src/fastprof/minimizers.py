"""Module containing the algorithms for negative log-likelihood (NLL) minimization

  The minimization is performed with respect to the model parameters, both
  *parameters of interest* (POIs) and *nuisance parameters* (NPs). They are treated
  differently in the minimization since fastprof models are by definition linear
  in the NPs. Minimization wrt the NPs is therefore a linear problem, while
  minimization wrt POIs is potentially non-linear.

  The code therefore consists of 2 main categories of algorithm:

  * Algorithms that minimize wrt NPs, for a given set of POI values ("profiling").
  * This consists of a single class, :class:`NPMinimizer`, which performs
  * linear minimization.

  * Algorithms that minimize wrt POIs. Since the problem is non-linear there is no
    unique best algorithm for all situations, and several options are implemented:


    * The :class:`OptiMinimizer` class, which use the minimization algorithms in
      :mod:`scipy.optimize`.

    * The :class:`ScanMinimizer` class, which relies on scans of the POI space

    The :class:`OptiMinimizer` class should generally provide the best performance.

    In all cases, the POI minimizers only deal with POIs, and use :class:`NPMinimizer`
    to provide best-fit values of the NPs at each point in POI-space.
    The POI minimizers all derive from the :class:`POIMinimizer` base class, which
    provides a common interface. It implements  in particular the computation of
    :math:`t_{\mu}` in :meth:`POIMinimizer::tmu`.
"""

import numpy as np
import math
import copy
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize
from abc import abstractmethod

from .core import Model, Parameters, Data
from .model_tools import ParBound

# -------------------------------------------------------------------------
class NPMinimizer :
  """Minimizer algorithm for NPs

  The class implements NLL minimization for linear parameters, following
  the algorithm described in the package documentation.

  The main inputs to the computation are stored in the likelihood model
  (class :class:`.core.Model`). The algorithm performs the minimally
  required computations of the quantities that depend on the provided dataset
  and POI values.

  Attributes:
     data (Data)           : the dataset used to compute the NLL
     min_pars (Parameters) : best-fit parameter values (stored by the
                             minimization routine)
     min_deltas (np.ndarray) : difference between the best-fit NP values
        and the hypothesis NP values used as reference point (stored by the
        minimization routine).
  """
  optimize_einsum = True
  
  def __init__(self, data : Data, verbosity : int = 0) :
    """Initialize the NPMinimizer object

      Args:
         data     : the dataset used for the NLL computation
    """
    self.data = data
    self.verbosity = verbosity
    self.min_pars = None
    self.min_deltas = None

  def pq_einsum(self, hypo : Parameters) -> (np.ndarray, np.ndarray) :
    """Non-public helper function to compute minimization inputs

      Computes the matrix P and the vector Q that enter into the
      lienar equation providing the best-fit NP balues

      Args:
         hypo : the POI hypothesis at which to perform the minimization
      Returns:
         the pair (P, Q)
    """
    model = self.data.model
    n_nom = model.n_exp(hypo)
    # i : bin index
    # k,l : sample indices
    # a,b,c : NP indices
    impacts = model.linear_impacts(hypo)
    # print('njpb', model.nbins_poisson, model.nbins_gaussian, model.nbins)
    if model.nbins_poisson > 0 :
      (p_pois, q_pois) = self.pq_einsum_poisson (n_nom[:,:model.nbins_poisson], self.data.counts[:model.nbins_poisson], impacts[:,:model.nbins_poisson,:])
    else :
      (p_pois, q_pois) = (np.zeros((model.nnps, model.nnps)), np.zeros(model.nnps))
    if model.nbins_gaussian > 0 :
      (p_gaus, q_gaus) = self.pq_einsum_gaussian(n_nom[:,model.nbins_poisson:], self.data.counts[model.nbins_poisson:], impacts[:,model.nbins_poisson:,:])
    else :
      (p_gaus, q_gaus) = (np.zeros((model.nnps, model.nnps)), np.zeros(model.nnps))
    # print('njpb', p_pois, q_pois, p_gaus, q_gaus, model.constraint_hessian, model.constraint_hessian.dot(hypo.nps - self.data.aux_obs))
    p = p_pois + p_gaus + model.constraint_hessian
    q = q_pois + q_gaus + model.constraint_hessian.dot(hypo.nps - self.data.aux_obs)
    if self.verbosity > 4 :
      print('== NPMinimizer: using profile matrices')
      print('P_pois = \n', p_pois)
      print('P_gaus = \n', p_gaus)
      print('C      = \n', model.constraint_hessian)
      print('Q_pois = \n', q_pois)
      print('Q_gaus = \n', q_gaus)
      print('C.daux = \n', model.constraint_hessian.dot(hypo.nps - self.data.aux_obs))
    return (p,q)

  def pq_einsum_poisson(self, n_nom : np.ndarray, obs : np.ndarray,impacts : np.ndarray) -> (np.ndarray, np.ndarray) :
    """Non-public helper function to compute minimization inputs for Poisson channels
    
      The P and Q matrices (see previous method) take separate contributions from
      channels with Poisson and Gaussian distributions. This function
      computes the Poisson contributions.
      
      The `obs` and `impacts` arguments are passed so that they can be computed
      only once and then used twice for the Poisson and Gaussian contributions

      Args:
         hypo : the POI hypothesis at which to perform the minimization
         obs : the observed event yields
         impacts : the NP impact values
      Returns:
         the pair (P, Q) for Poisson channels
    """
    # i : bin index
    # k,l : sample indices
    # a,b,c : NP indices
    t_nom = n_nom.sum(axis=0)
    delta_obs = t_nom - obs
    ratio_nom = n_nom / t_nom
    ratio_impacts = np.einsum('ki,kia->ia', ratio_nom, impacts, optimize=self.optimize_einsum)
    if self.verbosity > 5 :
      print('== NPMinimizer: using Poisson profile matrices')
      print('t_nom = \n', t_nom)
      print('delta_obs = \n', delta_obs)
      print('ratio_nom = \n', ratio_nom)
      print('ratio_impacts = \n', ratio_impacts)
    q = np.einsum('i,ia->a', delta_obs, ratio_impacts, optimize=self.optimize_einsum)
    p = np.einsum('i,ia,ib->ab', obs, ratio_impacts, ratio_impacts, optimize=self.optimize_einsum)
    if self.data.model.use_lognormal_terms : p += np.einsum('ki,i,kia,kib->ab', ratio_nom, delta_obs, impacts, impacts, optimize=self.optimize_einsum)
    return (p,q)

  def pq_einsum_gaussian(self, n_nom : np.ndarray, obs : np.ndarray, impacts : np.ndarray) -> (np.ndarray, np.ndarray) :
    """Non-public helper function to compute minimization inputs for Gaussian channels

      The P and Q matrices (see previous method) take separate contributions from
      channels with Poisson and Gaussian distributions. This function
      computes the Gaussian contributions.

      The `obs` and `impacts` arguments are passed so that they can be computed
      only once and then used twice for the Poisson and Gaussian contributions

      Args:
         hypo : the POI hypothesis at which to perform the minimization
         obs : the observed event yields
         impacts : the NP impact values
      Returns:
         the pair (P, Q) for Gaussian channels
    """
    # i,j : bin indices
    # k,l : sample indices
    # a,b,c : NP indices
    t_nom = n_nom.sum(axis=0)
    delta_obs = t_nom - obs
    hessian = self.data.model.poi_hessian
    #print('njpb', n_nom.shape, impacts.shape)
    n_nom_impacts = np.einsum('ki,kia->ia', n_nom, impacts, optimize=self.optimize_einsum)
    #print('njpb', n_nom_impacts.shape, hessian.shape, delta_obs.shape)
    q = np.einsum('ia,ij,j->a', n_nom_impacts, hessian, delta_obs, optimize=self.optimize_einsum)
    p = np.einsum('ia,ij,jb->ab', n_nom_impacts, hessian, n_nom_impacts, optimize=self.optimize_einsum)
    if self.data.model.use_lognormal_terms : p += 2*np.einsum('ki,ij,j,kia,kib->ab', n_nom, hessian, delta_obs, impacts, impacts, optimize=self.optimize_einsum)
    return (p,q)

  def profile(self, hypo : Parameters) -> Parameters :
    """Compute the best-fit NP values for a given POI hypothesis

      The hypothesis provided as input can be a :class:`Parameters`
      object, or any input that can be used to build one (see
      :meth:`Parameters.__init__`).
      If only POIs are provided, the NPs will initially be set 
      to the aux. obs. values in the dataset.
      
      The return value is a copy of the input object with NP
      set to their profile values (best-fit values of the NP for
      the given POI values)

      Args:
         hypo : the POI hypothesis at which to perform the minimization
      Returns:
         the profiled values
    """
    if not isinstance(hypo, Parameters) :
      hypo = Parameters(hypo, model=model).set_from_aux(self.data)
    self.p, self.q = self.pq_einsum(hypo)
    if self.verbosity > 3 :
      print('== NPMinimizer: profiling using')
      print('P = \n', self.p)
      print('Q = \n', self.q)
    d = np.linalg.det(self.p)
    if abs(d) < (1E-3)**self.q.size :
      print('== Linear system has an ill-conditioned coefficient matrix (det=%g), returning null result' % d)
      print('Profiling at hypothesis', hypo)
      print('P = \n', self.p)
      deltas = np.zeros(self.data.model.nnps)
    else :
      # all of the lines below should give the same results
      deltas = scipy.linalg.solve(self.p, self.q, assume_a='sym')
      # deltas = scipy.sparse.linalg.cg(self.p, self.q)[0]
      # deltas = np.linalg.inv(self.p).dot(self.q)
    nps = hypo.nps - deltas
    self.min_deltas = Parameters(hypo.pois, deltas, self.data.model)
    self.min_pars   = Parameters(hypo.pois, nps   , self.data.model)
    if self.verbosity > 2 :
      print('== NPMinimizer:')
      print('for hypo = ', hypo)
      print('obtained deltas = ', deltas)
      print('profiled nps = ', nps)
    return self.min_pars

  def profile_nll(self, hypo : Parameters, floor : float = None) -> float :
    """Compute the best-fit NLL for a given POI hypothesis

      Same as :meth:`NPMinimizer.profile` above, but returns the
      best-fit NLL value instead of the best-fit parameters. The
      best-fit parameters can still be accessed through the
      `min_pars` attribute
      Args:
         hypo  : the POI hypothesis at which to perform the minimization
         floor : minimal event yield to use in the NLL computation (see
         :meth:`.Model.nll` for details).
      Returns:
         The best-fit NLL value
    """
    self.profile(hypo)
    return self.data.model.nll(self.min_pars, self.data, floor=floor)


# -------------------------------------------------------------------------
class POIMinimizer :
  """Base class for POI minimizer algorithms

  Provides common pieces for the non-linear minimization
  over POIs.

  Attributes:
     niter (int) : number of iterations to perform when profiling NPs
     floor (float) : minimal event yield to use in the NLL computation (see
         :meth:`.Model.nll` for details).
     init_pois (Parameters) : initial values of the POI minimization
     bounds    (dict) : Bounds on the POIs, as dict of POI name -> ParBound object
     np_min (NPMinimizer) : the NP minimization algorithm
     min_nll (float) : the best-fit NLL stored by :meth:`POIMinimizer.profile_nps`.
     min_pars (Parameters) : the best-fit NPs stored by :meth:`POIMinimizer.profile_nps`.
     hypo_nll (float) : the NLL of the fixed-POI fit stored by :meth:`POIMinimizer.tmu`.
     hypo_pars (Parameters) : the best-fit parameters of the fixed-POI fit stored
                              by :meth:`POIMinimizer.profile_nps`.
     hypo_deltas (np.ndarray) : the best-fit deltas of the fixed-POI fit stored by :meth:`POIMinimizer.tmu`.
                                 (see :meth:`NPMinimizer.profile` for the definition of the deltas)
     free_nll (float) : the NLL of the firee-POI fit stored by :meth:`POIMinimizer.tmu`.
     free_pars (Parameters) : the best-fit parameters of the free-POI fit stored
                              by :meth:`POIMinimizer.profile_nps`.
     free_deltas (np.ndarray) : the best-fit deltas of the free-POI fit stored by :meth:`POIMinimizer.tmu`.
                                (see :meth:`NPMinimizer.profile` for the definition of the deltas)
     tmu_debug (float) : stores the raw value of `tmu` for debugging purposes

  """
  def __init__(self, niter : int = 1, floor : float = None) :
    """Initialize the POIMinimizer object

      Args:
        init_pois: the initial values of the POIs in the fit
        bounts: a dict of {(par_name, ParBound} pairs giving the parameter bounds
        niter : number of iterations to perform when minimizing over NPs
        floor :  minimal event yield to use in the NLL computation (see
            :meth:`.Model.nll` for details).
    """
    self.niter = niter
    self.floor = floor
    self.init_pois = None
    self.bounds = None
    self.np_min = None
    self.min_nll = None
    self.min_pars = None
    self.hypo_nll = None
    self.hypo_pars = None
    self.hypo_deltas = None
    self.free_nll = None
    self.free_nll = None
    self.free_deltas = None
    self.tmu_debug = 0

  def clone(self) :
    pass

  @abstractmethod
  def minimize(self, data : Data, init_pars : Parameters = None) -> float :
    """Abstract method to perform POI minimization

      This method needs to be implemented in derived classes. It performs
      the POI minimization starting at the provided hypothesis and returns
      the best-fit NLL.

      Args:
         data : dataset for which to compute the NLL
         init_pars : initial value (of POIs and NPs) for the minimization
      Returns:
         best-fit NLL
    """
    pass

  def set_pois(self, model : Model, init_pars : Parameters = None, bounds : list = None,
               hypo : dict = None, fix_hypo : bool = False) -> 'POIMinimizer' :
    """Set POI information

      Initial value and range information for the POIs is copied from either
      the ModelPOI objects in the model, or the provided arguments.
      If `hypo` is not None, override the POI values by those in the hypo.
      If `fix_hypo` is True, also set constant the POIs in the hypo.

      Args:
        model : the model to copy from
        init_pars : 
        bounds : a list of ParBounds
      Returns:
        self
    """
    # take the initial values from `init_pars` if not None, and else from the model `ref_pars`
    self.init_pois = model.ref_pars.clone() if init_pars is None else init_pars.clone()
    if hypo is not None : # overrride the inir_poi values with those in the hypo
      for par, val in hypo.items() : self.init_pois[par] = val
    # take the bounds from the model
    self.bounds = { poi.name : ParBound(poi.name, poi.min_value, poi.max_value) for poi in model.pois.values() }
    # and override with the ones provided as argument
    if bounds is not None :
      for bound in bounds : self.bounds[bound.par] = ParBound(bound.par, bound.min_value, bound.max_value)
    if fix_hypo :
      for poi in hypo : self.bounds[poi] = ParBound(poi, hypo[poi], hypo[poi]) # define 0-width bounds for fixed pars
    for bound in self.bounds.values() : # ensure the initial values are compatible with the bounds
      if not bound.test_value(self.init_pois[bound.par]) :
        print("Warning: resetting initial value of POI '%s' to %g (from %g) to ensure it verifies bound %s." 
              % (bound.par, (bound.min_value + bound.max_value)/2, self.init_pois[bound.par], str(bound)))
        self.init_pois[bound.par] = (bound.min_value + bound.max_value)/2
    return self

  def set_constant(self, parvals : dict) :
    """Set parameters to be constant

      Set the specified parameters to be constant at the 
      provided values.

      Args:
        parvals : { par_name :value } pairs providing the parameters and values.
    """
    for par, val in parvals.items() : self.bounds[par] = ParBound(par, val, val)

  def free_pois(self) -> list :
    """Provide the list of free parameters
    
      Returns :
        the list of free parameters in the fit
    """
    if self.init_pois is None : return None
    return [ poi for poi in self.init_pois.model.pois if poi not in self.bounds or not self.bounds[poi].is_fixed() ]


  def profile_nps(self, hypo : Parameters, data : Data) -> Parameters :
    """Compute best-fit NP values for given POI values

      This method computes the best-fit NPs for given POI values
      ("profiling"). The heavy lifting is delegated to :class:`NPMinimizer`.

      If the `niter` attribute is greather than 1, the minimization is performed
      `niter` times, with each iteration starting from the best-fit value of the
      previous one. This allows to account for non-linear impacts of the NPs: for
      instance asymmetric impacts for positive and negative NP variations, which
      are included in the model but cannot be used in the linear NP minimization.

      Args:
         hypo : Value of POIs and NPs for which to profile
         data : dataset for which to compute the NLL
      Returns:
         the best-fit parameters
    """
    self.np_min = NPMinimizer(data, verbosity=self.verbosity)
    self.min_pars = hypo
    for i in range(0, self.niter) :
      self.np_min.profile(self.min_pars)
      self.min_pars = self.np_min.min_pars
    self.min_nll = data.model.nll(self.min_pars, data, floor=self.floor)
    return self.min_pars

  def tmu(self, hypo : dict, data : Data, init_hypo : Parameters = None) -> float :
    """Computes the :math:`t_{\mu}` profile-likelihood ratio (PLR) test statistic

      The computation requires two minimizations:

      * A minimization over NPs, for the specified POI hypothesis

      * A minimization over both POI and NPs.

      The method performs both in turn, and stored the results as class
      attributes. The :math:`t_{\mu}` value is computed from the difference
      in the best-fit NLL values of the two minimizations listed above.

      Args:
         hypo      : A set of POI and NP values defining the hypothesis
         data      : the dataset for which to compute the NLL
         init_hypo : initial value (of POIs and NPs) for the minimization
      Returns:
         best-fit parameters
    """
    if isinstance(init_hypo, (int, float)) : init_hypo = Parameters(init_hypo, model=data.model)
    hypo_min = self.clone() # the minimizer for the hypo fit
    # Free fit initialization: do not fix the hypo (free hypo POIs)
    self.set_pois(data.model, init_pars=init_hypo, bounds=self.bounds.values() if self.bounds is not None else None, hypo=hypo, fix_hypo=False)
    # Hypo fit initialization : fix the hypo (fixed hypo POIs)
    # No need to check `self.bounds` since it was set by the previous command.
    hypo_min.set_pois(data.model, init_pars=init_hypo, bounds=self.bounds.values(), hypo=hypo, fix_hypo=True)
    # Hypo fit
    if hypo_min.minimize(data) is None : return None
    self.hypo_nll = hypo_min.min_nll
    self.hypo_pars = hypo_min.min_pars
    self.hypo_deltas = hypo_min.np_min.min_deltas
    # Free fit
    if self.minimize(data) is None : return None
    self.free_nll = self.min_nll
    self.free_pars = self.min_pars
    self.free_deltas = self.np_min.min_deltas
    tmu = 2*(self.hypo_nll - self.free_nll)
    if tmu < 0 :
      if tmu < -1E-10 :
        print('Warning: computed negative tmu = %g !' % tmu)
        if tmu < -5 :
          print('Hypothesis definition   :', hypo)
          print('Hypothesis fit result   :', self.hypo_pars)
          print('Free fit starting value :', init_hypo)
          print('Free fit result         :', self.free_pars)
          print(data.aux_obs)
        self.tmu_debug = tmu
      tmu = 0
    return tmu

# -------------------------------------------------------------------------
class ScanMinimizer (POIMinimizer) :
  """POI Minimizer using parameter scans

  The class implemented NLL minimization for linear parameters, following
  the algorithm described in the package documentation.

  Attributes:
     scan_pois  : POIs to scan
     pars       : best-fit parameters at each scan point
     nll        : best-fit NLL value at each scan point
     min_nll    : NLL value at minimum
     min_pars   : parameter values at minimum
     min_idx    : index of the minimum position
  """
  def __init__(self, model : Model, scan_pois : list, niter : int = 1) :
    """Initialize the ScanMinimizer object

      Args:
         model     : the statistical model to operate on
         scan_pois : the POI values to scan over
         niter     : number of iterations to perform in the NP minimization
    """
    super().__init__(niter)
    self.scan_pois = scan_pois
    self.pars = []
    for poi in scan_pois :
      self.pars.append(Parameters(poi, model.aux_obs, model)) # FIXME

  def clone(self) :
    """Clone the minimizer

      Returns:
         a deep copy of the minimizer
    """
    return ScanMinimizer(self.model, copy.copy(scan_pois), niter)

  def minimize(self, data : Data, init_pars : Parameters = None) -> float :
    """Minimization over POIs

      The method scans over the POI values in `scan_pois`, computes
      the NLL and profile NPs for each one, and returns the point
      that yields the minimal NLL.

      Args:
         data : dataset for which to compute the NLL
      Returns:
         best-fit parameters
    """
    if init_pars is not None or self.init_pois is None or self.bounds is None :
      self.set_pois(data.model, init_pars)
    self.nlls = np.zeros(self.scan_pois.size)
    for i in range(0, len(self.scan_pois)) :
      scan_hypo = init_hypo.clone().set_poi(self.scan_pois[i])
      np_min = NPMinimizer(data)
      self.nlls[i] = np_min.profile_nll(scan_hypo)
      self.pars[i] = np_min.min_pars
      #print('@poi(', i, ') =', poi, self.nlls[i], ahat, bhat)
    smooth_nll = InterpolatedUnivariateSpline(self.scan_pois, self.nlls, k=4)
    minima = smooth_nll.derivative().roots()
    self.min_nll = np.amin(self.nlls)
    self.min_idx = np.argmin(self.nlls)
    self.min_pois = np.array([self.scan_pois[self.min_idx]])
    if len(minima) == 1 :
      interp_min = smooth_nll(minima[0])
      if interp_min < self.min_nll :
        self.min_pois  = minima
        self.min_nll = interp_min
    self.min_pars = Parameters(self.min_pois, self.pars[self.min_idx], model=data.model)
    return self.min_nll

# -------------------------------------------------------------------------
class OptiMinimizer (POIMinimizer) :
  """POI Minimizer using scipy.optimize algorithms

  Implementation of :class:`POIMinimizer` using the scipy.optimize
  routines.

  Attributes:
     np_min     (NPMinimizer) : NP minimizer object
     init_pois  (Parameters)  : Initial values for the POI optimization
     bounds     (dict)        : Bounds on the POIs, as a dict mapping the
                                POI name to a (min, max) pair
     method     (str)         : optimization algorithm to apply
     alt_method (str)         : alternate optimization algorithm to apply if the
                                primary one (given by the `method` attribute) fails.
     rebound    (int)         : if > 0, perform `rebound` iterations while narrowing the bounds
                                by a factor 2 each time
     verbosity  (int)         : output level

  """
  def __init__(self, method : str = 'scalar', niter : int = 1, floor : float = 1E-7,
               rebound : int = 0, alt_method : str = None, force_num_jac : bool = False, verbosity : int = 0) :
    """Initialize the POIMinimizer object

      Args:
         method     : optimization algorithm to apply
         init_pois  : initial values of the POI minimization
         bounds     : Bounds on the POIs, as list of (min, max) pairs
         niter      : number of iterations to perform when minimizing over NPs
         floor      : minimal event yield to use in the NLL computation (see
                      :meth:`.Model.nll` for details).
         rebound    : if > 0, perform `rebound` iterations while narrowing the bounds
                      by a factor 2 each time
         alt_method : alternate optimization algorithm to apply if the
                      primary one (given by the `method` attribute) fails.
    """
    super().__init__(niter, floor)
    self.method = method
    self.rebound = rebound
    self.alt_method = alt_method
    self.verbosity = verbosity
    self.force_num_jac = force_num_jac
    self.np_min = None

  def clone(self) :
    """Clone the minimizer

      Returns:
         a deep copy of the minimizer
    """
    return OptiMinimizer(self.method, self.niter, self.floor, self.rebound, self.alt_method, self.verbosity)

  def minimize(self, data : Data, init_pars : Parameters = None) -> float :
    """Minimization over POIs

      The method implements two algorithms by default:

      * For the case of a single POI, the *bounded* method of
        :meth:`scipy.optimize.minimize_scalar` is used. This
        does not make use of the `init_pois` values, and only
        relies on the bounds provided by the `bounds` attribute.

      * For mutiple POIs, the *L-BFGS-B* gradient descent method
        is used, making use of the Jacobian expressions provided
        by the model.

      Other methods can be used by changing the `method` parameter.

      Args:
         data : dataset for which to compute the NLL
         init_pars : initial value (of POIs and NPs) for the minimization
      Returns:
         best-fit parameters
    """
    if init_pars is not None or self.init_pois is None or self.bounds is None :
      self.set_pois(data.model, init_pars)
    current_hypo = self.init_pois.clone()
    free_indices = [ i for i, bound in enumerate(self.bounds.values()) if bound.is_free() ]
    x0     = [ self.init_pois[bound.par] for bound in self.bounds.values() if bound.is_free() ]
    bounds = [ bound.bounds()            for bound in self.bounds.values() if bound.is_free() ]
    if len(x0) == 0 and self.method != '' :
      if self.verbosity > 0 : print("== OptiMinimizer: No free parameter of interest, will just minimize NPs.")
      self.method = ''
      self.profile_nps(current_hypo, data)
      self.min_pois = []
      self.nfev = 0
      return self.min_nll
    if len(x0) > 1 and self.method == 'scalar' :
      print("== OptiMinimizer: cannot use method 'scalar' for multiple POIs, switching to 'L-BFGS-B'.")
      self.method = 'L-BFGS-B'

    def update_current_hypo(pois) :
      for i, poi in zip(free_indices, pois) : current_hypo.pois[i] = poi
    def objective(pois) :
      #print('njpb pois = ', pois, type(pois))
      if isinstance(pois, (int, float)) : pois = np.array([ pois ])
      #print('njpb pois = ', pois, type(pois))
      update_current_hypo(pois)
      self.profile_nps(current_hypo, data)
      if self.verbosity > 2 : print('== OptMinimizer: eval at %s -> %g%s' % (str(pois), self.min_nll, ' with parameters' if self.verbosity > 3 else ''))
      if self.verbosity > 3 : print(current_hypo)
      if self.verbosity > 4 : print(self.min_pars)
      return self.min_nll
    def jacobian(pois) :
      update_current_hypo(pois)
      self.profile_nps(current_hypo, data)
      if self.verbosity > 2 : print('== Jacobian:', data.model.gradient(self.np_min.min_pars, data))
      return data.model.gradient(self.np_min.min_pars, data)
    #def hess_p(poi, v) :
      #update_current_hypo(pois)
      #self.profile_nps(current_hypo, data)
      #if self.verbosity > 0 : print('== Hessian:', data.model.hess_poi(self.np_min.min_pars, data)*v[0])
      #return np.array(data.model.hess_poi(self.np_min.min_pars, data)*v[0])
    jac = jacobian if data.model.gradient(self.init_pois, data) is not None and self.force_num_jac is False else None
    if self.verbosity > 0 :
      if jac is None :
        print("== OptiMinimizer: closed-form jacobian not available, will evaluate it numerically.")
      else:
        print("== OptiMinimizer: using closed-form jacobian.")
    if self.method == 'scalar' :
      bound = bounds[0]
      if self.verbosity > 0 : print("== OptiMinimizer: minimizing using scalar method 'bounded' in range %s" % str(bound))
      self.result = scipy.optimize.minimize_scalar(objective, bounds=bound, method='bounded', options={'xatol': 1e-5 })
    elif self.method == 'CG':
      if self.verbosity > 0 : print("== Optimizer: using method 'scalar' in range %s" % str(bound))
      self.result = scipy.optimize.minimize(objective, x0=x0, bounds=bounds, method='CG', jac=jac, options={'gtol': 1e-5, 'ftol':1e-5 })
    elif self.method == 'L-BFGS-B':
      if self.verbosity > 0 :
        print("== OptiMinimizer: minimizing the following parameters using method 'L-BFGS-B' :")
        for bound in self.bounds.values() : print('%10s = %10g (%s)' % (bound.par, self.init_pois[bound.par], str(bound)))
      self.result = scipy.optimize.minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B', jac=jac, options={'gtol': 1e-5, 'ftol':1e-5 })
    else :
      raise ValueError('Optiminimizer: unknown method %s.' % self.method)
    #print('Optimizer: done ----------------')
    if not self.result.success :
      print('Minimization failed, details below')
      print(dir(self.result))
      print('Current NPs:')
      print(self.min_pars)
      if hasattr(self.result, 'x')       : print('x       =', self.result.x)
      if hasattr(self.result, 'fun')     : print('fun     =', self.result.fun)
      if hasattr(self.result, 'status')  : print('status  =', self.result.status)
      if hasattr(self.result, 'message') : print('message =', self.result.message)
      return None, None
    self.min_nll = self.result.fun
    self.min_pois = self.result.x if isinstance(self.result.x, np.ndarray) else np.array([self.result.x])
    self.nfev = self.result.nfev
    if self.method == 'L-BFGS-B': 
      self.covmat = self.result.hess_inv.todense() # may not be available for
      self.errors = np.sqrt(self.covmat.diagonal())
      self.cormat = (self.covmat.T / self.errors).T / self.errors
    return self.min_nll

  def tmu(self, hypo : dict, data : Data, init_hypo : Parameters = None) -> float :
    """Computes the :math:`t_{\mu}` profile-likelihood ratio (PLR) test statistic

      This method overrides the default in :class:`POIMinimizer` by adding
      two alternates:

      * The *scalar* method used in 1D sometimes fails due to a too-wide
        parameter range. In this case, the `rebound` attribute implements an
        alternate in which the minimization is repeated with iteratively
        smaller ranges around the init_pois valut, until the minimization
        converges.

      * An alternate method can be specified in the `alt_method` attribute.
        If the primary method fails, this one is tried instead.

      Args:
         hypo      : POI hypothesis used for the fixed-POI minimization
         data      : dataset for which to compute the NLL
         init_hypo : initial value (of POIs and NPs) for the minimization
      Returns:
         best-fit parameters
    """
    tmu = super().tmu(hypo, data, init_hypo)
    if tmu == 0 and self.tmu_debug < 0 :
      if self.method == 'scalar' and self.rebound > 0 :
        init_pois = self.init_pois.pois if isinstance(self.init_pois, Parameters) else Parameters(self.init_pois).pois
        new_bounds = [ ( (init + bounds[0])/2, (init + bounds[1])/2 ) for init, bounds in zip(init_pois, self.bounds) ]
        print('Warning: tmu computation failed (tmu < 0) with bounds', self.bounds, ', repeating with narrower bounds: ', new_bounds, ' (%d attempts left).' % self.rebound)
        self.bounds = new_bounds
        self.rebound -= 1
        return self.tmu(hypo, data, init_hypo)
      if self.alt_method != None :
        print('Warning: tmu computation failed (tmu < 0) with method %s, repeating with alternate method %s.' % (self.method, self.alt_method))
        self.method = self.alt_method
        self.alt_method = None
        return self.tmu(hypo, data, init_hypo)
    return tmu
