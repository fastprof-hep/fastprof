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
    self.model = data.model
    self.data = data

# Same as above, more readable, and faster -- use by default
  def pq_einsum(self, hypo) :
    snom = self.model.s_exp(hypo)
    bnom = self.model.b_exp(hypo)
    nnom = snom + bnom
    rB = bnom/nnom
    rS = snom/nnom
    dN = nnom - self.data.n
    qA = np.einsum('i,i,ij->j', rS, dN, self.model.a) + hypo.alphas + self.model.ref_alphas - self.data.aux_alphas
    qB = np.einsum('i,i,ij->j', rB, dN, self.model.b) + hypo.betas  + self.model.ref_betas  - self.data.aux_betas
    qC = np.einsum('i,i,ij->j', rB, dN, self.model.c)
    pAA = np.einsum('i,i,ij,ik->jk',   rS, rS *self.data.n + dN, self.model.a, self.model.a) + self.model.diag_alphas
    pAB = np.einsum('i,i,i,ij,ik->jk', rS, rB, self.data.n     , self.model.a, self.model.b)
    pAC = np.einsum('i,i,i,ij,ik->jk', rS, rB, self.data.n     , self.model.a, self.model.c)
    pBB = np.einsum('i,i,i,ij,ik->jk', rB, rB, self.data.n     , self.model.b, self.model.b) + self.model.diag_betas
    pBC = np.einsum('i,i,i,ij,ik->jk', rB, rB, self.data.n     , self.model.b, self.model.c)
    pCC = np.einsum('i,i,i,ij,ik->jk', rB, rB, self.data.n     , self.model.c, self.model.c) + self.model.diag_gammas
    return np.block([[pAA, pAB, pAC], [np.transpose(pAB), pBB, pBC], [np.transpose(pAC), np.transpose(pBC), pCC]]), np.block([qA, qB, qC])
  
  def profile(self, hypo) :
    if isinstance(hypo, (int, float)) :
      hypo = self.model.expected_pars(hypo).set_from_aux(self.data)
    p, q = self.pq_einsum(hypo)
    d = np.linalg.det(p)
    if abs(d) < 1E-8 :
      print('Linear system has an ill-conditioned coefficient matrix (det= %g), returning null result' % d)
      nps = self.data.aux_alphas, self.data.aux_betas, np.zeros(self.model.nc)
    else :
      v = np.linalg.inv(p).dot(q)
      nps = hypo.alphas - v[:self.model.na], \
            hypo.betas  - v[self.model.na:self.model.nsyst], \
            hypo.gammas - v[self.model.nsyst:]
    self.min_pars = Parameters(hypo.poi, *nps, self.model)
    return self.min_pars
  
  def profile_nll(self, hypo = None, floor = None) :
    self.profile(hypo)
    return self.model.nll(self.min_pars, self.data, floor=floor)
  
  
# -------------------------------------------------------------------------
class POIMinimizer :
  def __init__(self, data, niter = 1, floor = None) :
    self.model = data.model
    self.data = data
    self.niter = niter
    self.floor = floor
    self.tmu_debug = 0
  @abstractmethod
  def minimize(self, init_hypo) :
    pass
  def profile_nps(self, hypo) :
    if isinstance(hypo, (int, float)) : 
      hypo = self.model.expected_pars(hypo)
    self.np_min = NPMinimizer(self.data)
    self.min_pars = hypo
    for i in range(0, self.niter) :
      self.np_min.profile(self.min_pars)
      self.min_pars = self.np_min.min_pars
    self.nll_min = self.model.nll(self.min_pars, self.data, floor=self.floor)
    #print('profile NPs @ %g' % poi)
    #print(str(self.min_pars))
    return self.min_pars
  def tmu(self, hypo, free=None) :
    if isinstance(hypo, (int, float)) :
      hypo = self.model.expected_pars(hypo, self)
    if isinstance(free, (int, float)) :
      free = self.model.expected_pars(free, self)
    #print('tmu @ %g' % hypo.poi)
    self.profile_nps(hypo)
    self.hypo_nll = self.nll_min
    self.hypo_pars = self.min_pars
    self.minimize(free)
    if self.nll_min == None : return None
    self.free_nll = self.nll_min
    self.free_pars = self.min_pars
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
        print(self.data.aux_alphas, self.data.aux_betas)
      self.tmu_debug = tmu
      tmu = 0
    return tmu
  def asimov_clone(self, poi) :
    clone = copy.copy(self)
    #print('generating asimov')
    clone.data = self.model.generate_expected(poi, self)
    return clone

# -------------------------------------------------------------------------
class ScanMinimizer (POIMinimizer) :
  def __init__(self, data, scan_pois, niter=1) :
    super().__init__(data, niter)
    self.scan_pois = scan_pois
    self.pars = []
    for poi in scan_pois :
      self.pars.append(Parameters(poi, self.data.aux_alphas, self.data.aux_betas, np.zeros(self.model.nc), self.model))

  def minimize(self, init_hypo) :
    self.nlls = np.zeros(self.scan_pois.size)
    for i in range(0, len(self.scan_pois)) :
      scan_hypo = copy.deepcopy(init_hypo).set_poi(self.scan_pois[i])
      np_min = NPMinimizer(self.data)
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
  def __init__(self, data, poi0 = 0, bounds = (0, 10), method = 'scalar', niter = 1, floor = 1E-7, rebound = 0, alt_method = 'L-BFGS-B') :
    super().__init__(data, niter, floor)
    self.np_min = None
    self.poi0 = poi0
    self.bounds = bounds
    self.method = method
    self.rebound = rebound
    self.alt_method = alt_method
    self.debug = 0
   
  def minimize(self, init_hypo) :
    if init_hypo == None :
      current_hypo = self.model.expected_pars(self.poi0, self)
    else :
      current_hypo = copy.deepcopy(init_hypo)
    def objective(poi) :
      if isinstance(poi, np.ndarray) : poi = poi[0]
      if isinstance(poi, np.ndarray) : poi = poi[0]
      self.profile_nps(current_hypo.set_poi(poi))
      if self.debug > 0 : print('== OptMinimizer: eval at %g -> %g' % (poi, self.nll_min))
      if self.debug > 1 : print(current_hypo)
      if self.debug > 1 : print(self.min_pars)
      return self.nll_min
    def jacobian(poi) :
      if isinstance(poi, np.ndarray) : poi = poi[0]
      if isinstance(poi, np.ndarray) : poi = poi[0]
      self.profile_nps(current_hypo.set_poi(poi))
      if self.debug > 0 : print('== Jacobian:', self.np_min.model.grad_poi(self.np_min.min_pars, self.data))
      return np.array([ self.np_min.model.grad_poi(self.np_min.min_pars, self.data) ])
    def hess_p(poi, v) :
      if isinstance(poi, np.ndarray) : poi = poi[0]
      if isinstance(poi, np.ndarray) : poi = poi[0]
      self.profile_nps(current_hypo.set_poi(poi))
      if self.debug > 0 : print('== Hessian:', self.np_min.model.hess_poi(self.np_min.min_pars, self.data)*v[0])
      return np.array([ self.np_min.model.hess_poi(self.np_min.min_pars, self.data)*v[0] ])
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
    self.nfev = result.nfev
    #print(self.min_poi)
    return self.nll_min, self.min_poi

  def tmu(self, hypo, free=None) :
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
