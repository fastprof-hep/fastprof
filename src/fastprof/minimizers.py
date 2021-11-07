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

  The main inputs to the computation are stored in the likleihood model
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
  
  def __init__(self, data : Data) :
    """Initialize the NPMinimizer object

      Args:
         data     : the dataset used for the NLL computation
    """
    self.data = data
    self.min_pars = None
    self.min_deltas = None

  def pq_einsum(self, hypo : Parameters) -> (np.ndarray, np.ndarray) :
    """Non-public helper function to compute minimization inputs

      Args:
         hypo : the POI hypothesis at which to perform the minimization
      Returns:
         The P-matrix and Q-vector that enter the computation
         of the best-fit NPs
    """
    model = self.data.model
    n_nom = model.n_exp(hypo)
    t_nom = n_nom.sum(axis=0)
    delta_obs = t_nom - self.data.counts
    ratio_nom = n_nom / t_nom
    # i : bin index
    # k,l : sample indices
    # a,b,c : NP indices
    impacts = model.linear_impacts(hypo)
    ratio_impacts = np.einsum('ki,kia->ia', ratio_nom, impacts, optimize=self.optimize_einsum)
    q  = np.einsum('i,ia->a', delta_obs, ratio_impacts, optimize=self.optimize_einsum) + model.constraint_hessian.dot(hypo.nps - self.data.aux_obs)
    p = np.einsum('i,ia,ib->ab', self.data.counts, ratio_impacts, ratio_impacts, optimize=self.optimize_einsum)
    if model.use_lognormal_terms : p += np.einsum('ki,i,kia,kib->ab', ratio_nom, delta_obs, impacts, impacts, optimize=self.optimize_einsum)
    p += model.constraint_hessian
    return (p,q)

  def profile(self, hypo : Parameters) -> Parameters :
    """Compute the best-fit NP values for a given POI hypothesis

      The hypothesis provided as input can be a :class:`.Parameters`
      object, or any input that can be used to build one (see
      :meth:`.Parameters.__init__`).
      If only POIs are provided, the NPs will be set to the aux. obs.
      values in the dataset.

      Args:
         hypo : the POI hypothesis at which to perform the minimization
      Returns:
         A parameter set with POI values set to the hypothesis and NP
         values to the corresponding best-fit NPs.
    """
    if not isinstance(hypo, Parameters) :
      hypo = Parameters(hypo, model=model).set_from_aux(self.data)
    self.p, self.q = self.pq_einsum(hypo)
    d = np.linalg.det(self.p)
    if abs(d) < (1E-3)**self.q.size :
      print('Linear system has an ill-conditioned coefficient matrix (det=%g), returning null result' % d)
      print('Profiling at hypothesis', hypo)
      print(self.p)
      deltas = np.zeros(self.data.model.nnps)
    else :
      # all of the lines below should give the same results
      deltas = scipy.linalg.solve(self.p, self.q, assume_a='sym')
      # deltas = scipy.sparse.linalg.cg(self.p, self.q)[0]
      # deltas = np.linalg.inv(self.p).dot(self.q)
    nps = hypo.nps - deltas
    self.min_deltas = Parameters(hypo.pois, deltas, self.data.model)
    self.min_pars   = Parameters(hypo.pois, nps   , self.data.model)
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
     init_pois (Parameters) : initial values of the POI minimization
     bounds    (dict) : Bounds on the POIs, as dict of POI name -> ParBound object
     niter (int) : number of iterations to perform when profiling NPs
     floor (float) : minimal event yield to use in the NLL computation (see
         :meth:`.Model.nll` for details).
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
     tmu_debug (float) : stores the raw value o `tmu` for debugging purposes

  """
  def __init__(self, init_pois : Parameters = None, bounds : dict = None, niter : int = 1, floor : float = None) :
    """Initialize the POIMinimizer object

      Args:
         niter : number of iterations to perform when minimizing over NPs
         floor :  minimal event yield to use in the NLL computation (see
            :meth:`.Model.nll` for details).
    """
    self.init_pois = init_pois
    self.bounds = bounds
    self.niter = niter
    self.floor = floor
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
  def minimize(self, data : Data, init_hypo : Parameters = None) -> float :
    """Abstract method to perform POI minimization

      This method needs to be implemented in derived classes. It performs
      the POI minimization starting at the provided hypothesis and returns
      the best-fit NLL.

      Args:
         data : dataset for which to compute the NLL
         init_hypo : initial value (of POIs and NPs) for the minimization
      Returns:
         best-fit NLL
    """
    pass

  def set_pois(self, model : Model, init_pars : Parameters = None, hypo : dict = None, fix_hypo : bool = False) -> 'OptiMinimizer' :
    """Copy POI information from model

      Initial value and range information is copied from the contents of
      the ModelPOI objects in the model.

      Args:
        model : the model to copy from
      Returns:
        self
    """
    self.init_pois = Parameters({ poi.name : poi.initial_value for poi in model.pois.values() }, model=model) if init_pars is None else init_pars.clone()
    if hypo is not None :
      for par, val in hypo.items() : self.init_pois[par] = val
    self.bounds = { poi.name : ParBound(poi.name, poi.min_value, poi.max_value) for poi in model.pois.values() }
    if fix_hypo :
      for poi in hypo : self.bounds[poi] = ParBound(poi, hypo[poi], hypo[poi])
    for bound in self.bounds.values() :
      if not bound.test_value(self.init_pois[bound.par]) :
        if self.debug > 1 :
          print("Warning: resetting value of POI '%s' to %g (from %g) to ensure it verifies bound %s." % (bound.par, (bound.min_value + bound.max_value)/2, self.init_pois[bound.par], str(bound)))
        self.init_pois[bound.par] = (bound.min_value + bound.max_value)/2
    return self


  def free_pois(self) :
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
         best-fit parameters
    """
    self.np_min = NPMinimizer(data)
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
    hypo_minimizer = self.clone()
    hypo_minimizer.set_pois(data.model, init_pars=init_hypo, hypo=hypo, fix_hypo=True)
    self.set_pois(data.model, init_pars=init_hypo, hypo=hypo, fix_hypo=False)
    # Hypo fit
    if hypo_minimizer.minimize(data) is None : return None
    self.hypo_nll = hypo_minimizer.min_nll
    self.hypo_pars = hypo_minimizer.min_pars
    self.hypo_deltas = hypo_minimizer.np_min.min_deltas
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
    return ScanMinimizer(self.model, copy.copy(scan_pois), niter)

  def minimize(self, data : Data) -> float :
    """Minimization over POIs

      The method scans over the POI values in `scan_pois`, computes
      the NLL and profile NPs for each one, and returns the point
      that yields the minimal NLL.

      Args:
         data : dataset for which to compute the NLL
      Returns:
         best-fit parameters
    """
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
     debug      (int)         : level of debug output

  """
  def __init__(self, method : str = 'scalar', init_pois : Parameters = None, bounds : dict = None, niter : int = 1, floor : float = 1E-7, rebound : int = 0, alt_method : str = None, debug : int = 0) :
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
    super().__init__(init_pois, bounds, niter, floor)
    self.method = method
    self.rebound = rebound
    self.alt_method = alt_method
    self.debug = debug
    self.np_min = None

  def clone(self) :
    return OptiMinimizer(self.method, self.init_pois.clone() if self.init_pois is not None else None, copy.deepcopy(self.bounds), self.niter, self.floor, self.rebound, self.alt_method, self.debug)

  def minimize(self, data : Data) -> float :
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
         init_hypo : initial value (of POIs and NPs) for the minimization
      Returns:
         best-fit parameters
    """
    if self.init_pois is None or self.bounds is None : self.set_pois(data.model)
    current_hypo = self.init_pois.clone()
    free_indices = [ i for i, bound in enumerate(self.bounds.values()) if bound.is_free() ]
    x0     = [ self.init_pois[bound.par] for bound in self.bounds.values() if bound.is_free() ]
    bounds = [ bound.bounds()            for bound in self.bounds.values() if bound.is_free() ]
    jac = jacobian if data.model.gradient(self.init_pois, data) is not None else None
    if len(x0) == 0 and self.method != '' :
      if self.debug > 0 : print("== OptiMinimizer: No free parameter of interest, will just minimize NPs.")
      self.method = ''
      self.profile_nps(current_hypo, data)
      self.min_pois = []
      self.nfev = 0
      return self.min_nll
    if len(x0) > 1 and self.method == 'scalar' :
      print("== OptiMinimizer: Cannot use method 'scalar' for multiple POIs, switching to 'L-BFGS-B'.")
      self.method = 'L-BFGS-B'

    def update_current_hypo(pois) :
      for i, poi in zip(free_indices, pois) : current_hypo.pois[i] = poi
    def objective(pois) :
      #print('njpb pois = ', pois, type(pois))
      if isinstance(pois, (int, float)) : pois = np.array([ pois ])
      #print('njpb pois = ', pois, type(pois))
      update_current_hypo(pois)
      self.profile_nps(current_hypo, data)
      if self.debug > 2 : print('== OptMinimizer: eval at %s -> %g' % (str(pois), self.min_nll))
      if self.debug > 3 : print(current_hypo)
      if self.debug > 4 : print(self.min_pars)
      return self.min_nll
    def jacobian(pois) :
      update_current_hypo(pois)
      self.profile_nps(current_hypo, data)
      if self.debug > 0 : print('== Jacobian:', data.model.gradient(self.np_min.min_pars, data))
      return data.model.grad_poi(self.np_min.min_pars, data)
    #def hess_p(poi, v) :
      #update_current_hypo(pois)
      #self.profile_nps(current_hypo, data)
      #if self.debug > 0 : print('== Hessian:', data.model.hess_poi(self.np_min.min_pars, data)*v[0])
      #return np.array(data.model.hess_poi(self.np_min.min_pars, data)*v[0])

    if self.method == 'scalar' :
      bound = bounds[0]
      if self.debug > 0 : print("== OptiMinimizer: minimizing using scalar method 'bounded' in range %s" % str(bound))
      self.result = scipy.optimize.minimize_scalar(objective, bounds=bound, method='bounded', options={'xatol': 1e-5 })
    elif self.method == 'CG':
      if self.debug > 0 : print("== Optimizer: using method 'scalar' in range %s" % str(bound))
      self.result = scipy.optimize.minimize(objective, x0=x0, bounds=bounds, method='CG', jac=jac, options={'gtol': 1e-5, 'ftol':1e-5 })
    elif self.method == 'L-BFGS-B':
      if self.debug > 0 :
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
    return self.min_nll

  def tmu(self, hypo : Parameters, data : Data, init_hypo : Parameters = None) -> float :
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
