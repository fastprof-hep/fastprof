import numpy as np
import math
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize
from abc import abstractmethod
import copy

from .core import Model, Parameters, Data

# -------------------------------------------------------------------------
class NPMinimizer :
  def __init__(self, data) :
    self.data = data

# Same as above, more readable, and faster -- use by default
  def pq_einsum(self, hypo) :
    model = self.data.model
    n_nom = model.n_exp(hypo)
    t_nom = n_nom.sum(axis=0)
    delta_obs = t_nom - self.data.counts
    ratio_nom = n_nom / t_nom
    # i : bin index
    # k,l : sample indices
    # a,b,c : NP indices
    ratio_impacts = np.einsum('ki,kia->ia', ratio_nom, model.sym_impacts)
    q  = np.einsum('i,ia->a', delta_obs, ratio_impacts) + model.constraint_hessian.dot(hypo.nps - self.data.aux_obs)
    p = np.einsum('i,ia,ib->ab', self.data.counts, ratio_impacts, ratio_impacts)
    if model.lognormal_terms : p += np.einsum('ki,i,kia,kib->ab', ratio_nom, delta_obs, model.sym_impacts, model.sym_impacts)
    p += model.constraint_hessian
    return (p,q)
  
  def profile(self, hypo) :
    if isinstance(hypo, (int, float)) :
      hypo = self.data.model.expected_pars(hypo).set_from_aux(self.data)
    p, q = self.pq_einsum(hypo)
    d = np.linalg.det(p)
    if abs(d) < 1E-8 :
      print('Linear system has an ill-conditioned coefficient matrix (det= %g), returning null result' % d)
      nps = self.data.aux_obs, np.zeros(self.model.f_dim)
    else :
      deltas = np.linalg.inv(p).dot(q)
      nps = hypo.nps - deltas
    self.min_deltas = Parameters(hypo.pois, deltas, self.data.model)
    self.min_pars   = Parameters(hypo.pois, nps   , self.data.model)
    return self.min_pars
  
  def profile_nll(self, hypo = None, floor = None) :
    self.profile(hypo)
    return self.data.model.nll(self.min_pars, self.data, floor=floor)
  
  
# -------------------------------------------------------------------------
class POIMinimizer :
  def __init__(self, niter = 1, floor = None) :
    self.niter = niter
    self.floor = floor
    self.tmu_debug = 0
  @abstractmethod
  def minimize(self, data, init_hypo = None) :
    pass
  def profile_nps(self, hypo, data) :
    if isinstance(hypo, (int, float)) : 
      hypo = data.model.expected_pars(hypo)
    self.np_min = NPMinimizer(data)
    self.min_pars = hypo
    for i in range(0, self.niter) :
      self.np_min.profile(self.min_pars)
      self.min_pars = self.np_min.min_pars
    self.nll_min = data.model.nll(self.min_pars, data, floor=self.floor)
    #print('profile NPs @ %g' % poi)
    #print(str(self.min_pars))
    return self.min_pars
  def tmu(self, hypo, data, free=None) :
    if isinstance(hypo, (int, float)) :
      hypo = data.model.expected_pars(hypo, self)
    if isinstance(free, (int, float)) :
      free = data.model.expected_pars(free, self)
    #print('tmu @ %g' % hypo.poi)
    self.profile_nps(hypo)
    self.hypo_nll = self.nll_min
    self.hypo_pars = self.min_pars
    self.hypo_deltas = self.np_min.min_deltas
    self.minimize(free)
    if self.nll_min == None : return None
    self.free_nll = self.nll_min
    self.free_pars = self.min_pars
    self.free_deltas = self.np_min.min_deltas
    #print(self.free_nll, str(self.free_pars))
    #print(self.hypo_nll, str(self.hypo_pars))
    tmu = 2*(self.hypo_nll - self.free_nll)
    if tmu < 0 :
      print('Warning: computed negative tmu = %g !' % tmu)
      if tmu < -5 :
        print('Hypothesis definition   :', hypo)
        print('Hypothesis fit result   :', self.hypo_pars)
        print('Free fit starting value :', free)
        print('Free fit result         :', self.free_pars)
        print(data.aux_obs)
      self.tmu_debug = tmu
      tmu = 0
    return tmu

# -------------------------------------------------------------------------
class ScanMinimizer (POIMinimizer) :
  def __init__(self, model, scan_pois, niter=1) :
    super().__init__(niter)
    self.scan_pois = scan_pois
    self.pars = []
    for poi in scan_pois :
      self.pars.append(Parameters(poi, model.aux_obs, model)) # FIXME

  def minimize(self, data, init_hypo = None) :
    self.nlls = np.zeros(self.scan_pois.size)
    for i in range(0, len(self.scan_pois)) :
      scan_hypo = init_hypo.clone().set_poi(self.scan_pois[i])
      np_min = NPMinimizer(data)
      self.nlls[i] = np_min.profile_nll(scan_hypo)
      self.pars[i] = np_min.min_pars
      #print('@poi(', i, ') =', poi, self.nlls[i], ahat, bhat)
    smooth_nll = InterpolatedUnivariateSpline(self.scan_pois, self.nlls, k=4)
    minima = smooth_nll.derivative().roots()
    self.nll_min = np.amin(self.nlls)
    self.min_idx = np.argmin(self.nlls)
    self.min_poi  = self.scan_pois[self.min_idx]
    if len(minima) == 1 :
      interp_min = smooth_nll(minima[0])
      if interp_min < self.nll_min :
        self.min_poi  = minima[0]
        self.nll_min = interp_min
    self.min_pars = Parameters(self.min_poi, self.pars[self.min_idx].alphas, self.pars[self.min_idx].betas, self.pars[self.min_idx].gammas)
    return self.nll_min, self.min_poi

# -------------------------------------------------------------------------
class OptiMinimizer (POIMinimizer) :
  def __init__(self, poi0 = 0, bounds = (0, 10), method = 'scalar', niter = 1, floor = 1E-7, rebound = 0, alt_method = None) :
    super().__init__(niter, floor)
    self.np_min = None
    self.poi0 = poi0
    self.bounds = bounds
    self.method = method
    self.rebound = rebound
    self.alt_method = alt_method
    self.debug = 0
   
  def minimize(self, data, init_hypo = None) :
    if init_hypo == None :
      current_hypo = data.model.expected_pars(self.poi0, self)
    else :
      current_hypo = init_hypo.clone()
    def objective(poi) :
      if isinstance(poi, np.ndarray) : poi = poi[0]
      if isinstance(poi, np.ndarray) : poi = poi[0]
      self.profile_nps(current_hypo.set(list(data.model.pois)[0], poi))
      if self.debug > 0 : print('== OptMinimizer: eval at %g -> %g' % (poi, self.nll_min))
      if self.debug > 1 : print(current_hypo)
      if self.debug > 1 : print(self.min_pars)
      return self.nll_min
    def jacobian(poi) :
      if isinstance(poi, np.ndarray) : poi = poi[0]
      if isinstance(poi, np.ndarray) : poi = poi[0]
      self.profile_nps(current_hypo.set_poi(poi))
      if self.debug > 0 : print('== Jacobian:', self.np_min.data.model.grad_poi(self.np_min.min_pars, data))
      return np.array([ self.np_min.data.model.grad_poi(self.np_min.min_pars, data) ])
    def hess_p(poi, v) :
      if isinstance(poi, np.ndarray) : poi = poi[0]
      if isinstance(poi, np.ndarray) : poi = poi[0]
      self.profile_nps(current_hypo.set_poi(poi))
      if self.debug > 0 : print('== Hessian:', self.np_min.data.model.hess_poi(self.np_min.min_pars, data)*v[0])
      return np.array([ self.np_min.data.model.hess_poi(self.np_min.min_pars, data)*v[0] ])
    if self.method == 'scalar' :
      if self.debug > 0 : print('== Optimizer: using scalar  ----------------')
      result = scipy.optimize.minimize_scalar(objective, bounds=self.bounds, method='bounded', options={'xatol': 1e-5 })
    elif self.method == 'L-BFGS-B':
      result = scipy.optimize.minimize(objective, x0=self.poi0, bounds=(self.bounds,), method='L-BFGS-B', jac=jacobian, options={'gtol': 1e-5, 'ftol':1e-5 })
      if not result.success :
        result = scipy.optimize.minimize_scalar(objective, bounds=self.bounds, method='bounded', options={'xtol': 1e-3 })
    else :
      raise ValueError('Optiminimizer: unknown method %s.' % self.method)
    #print('Optimizer: done ----------------')
    if not result.success :
      print('Minimization failed, details below')
      print(dir(result))
      print('Current NPs:')
      print(self.min_pars)
      if hasattr(result, 'x')       : print('x       =', result.x)
      if hasattr(result, 'fun')     : print('fun     =', result.fun)
      if hasattr(result, 'status')  : print('status  =', result.status)
      if hasattr(result, 'message') : print('message =', result.message)
      return None, None
    self.nll_min = result.fun
    self.min_poi = result.x
    if isinstance(self.min_poi, np.ndarray) : self.min_poi = self.min_poi[0] # needed for L-BGS-B
    self.nfev = result.nfev
    #print(self.min_poi)
    return self.nll_min, self.min_poi

  def tmu(self, hypo, data, free=None) :
    tmu = super().tmu(hypo, free)
    if tmu == 0 and self.tmu_debug < 0 :
      if self.method == 'scalar' and self.rebound > 0 :
        new_bounds = ((self.poi0 + self.bounds[0])/2, (self.poi0 + self.bounds[1])/2)
        print('Warning: tmu computation failed (tmu < 0) with bounds', self.bounds, ', repeating with narrower bounds: ', new_bounds, ' (%d attempts left).' % self.rebound)
        self.bounds = new_bounds
        self.rebound -= 1
        return self.tmu(hypo, free)
      if self.alt_method != None :
        print('Warning: tmu computation failed (tmu < 0) with method %s, repeating with alternate method %s.' % (self.method, self.alt_method))
        self.method = self.alt_method
        self.alt_method = None
        return self.tmu(hypo, free)
    return tmu
