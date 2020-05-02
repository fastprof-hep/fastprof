import numpy as np
import math
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize
from abc import abstractmethod
import copy

from .core import Model, Parameters, Data

# -------------------------------------------------------------------------
class NPMinimizer :
  def __init__(self, mu, data) :
    self.model = data.model
    self.mu = mu
    self.data = data
    
  def pq_naive(self) :
    na = self.model.na
    nb = self.model.nb
    nc = self.model.nc
    ns = self.model.nsyst
    nabc = ns + nc
    p = np.zeros((nabc, nabc))
    q = np.zeros(nabc)
    pars_nom = Parameters(self.mu, self.data.aux_alphas, self.data.aux_betas, np.zeros(self.model.nc))
    snom = self.model.s_exp(pars_nom)
    nnom = snom + self.model.b_exp(pars_nom)
    r1 = 1 - self.data.n/nnom
    r2 = self.data.n/nnom/nnom
    r3 = 1 - self.data.n/nnom*(1 - snom/nnom)
    #print('PQ| model = ',self. model)
    #print('PQ| mu = ', self.mu) 
    #print('PQ| data = ', self.data)
    #print('PQ| p_nom = ', pars_nom)
    #print('PQ| snom = ', snom)
    #print('PQ| nnom = ', nnom)
    for i in range(0, self.model.nbins) :
      for ia in range(0, na) :
        q[ia] += snom[i]*r1[i]*self.model.a[i,ia]
        for ja in range(0, na) :  p[ia,      ja] += snom[i]                  *r3[i]*self.model.a[i,ia]*self.model.a[i,ja]
        for jb in range(0, nb)  : p[ia, na + jb] += snom[i]*self.model.bkg[i]*r2[i]*self.model.a[i,ia]*self.model.b[i,jb]
        for jc in range(0, nc)  : p[ia, ns + jc] += snom[i]*self.model.bkg[i]*r2[i]*self.model.a[i,ia]*self.model.c[i,jc]
      for ib in range(0, nb) :
        q[na + ib] += self.model.bkg[i]*r1[i]*self.model.b[i,ib]
        for ja in range(0, na) :  p[na + ib,      ja] += snom[i]          *self.model.bkg[i]*r2[i]*self.model.b[i,ib]*self.model.a[i,ja]
        for jb in range(0, nb)  : p[na + ib, na + jb] += self.model.bkg[i]*self.model.bkg[i]*r2[i]*self.model.b[i,ib]*self.model.b[i,jb]
        for jc in range(0, nc)  : p[na + ib, ns + jc] += self.model.bkg[i]*self.model.bkg[i]*r2[i]*self.model.b[i,ib]*self.model.c[i,jc]
      for ic in range(0, nc) :
        q[ns + ic] += self.model.bkg[i]*r1[i]*self.model.c[i,ic]
        for ja in range(0, na) :  p[ns + ic,      ja] += snom[i]          *self.model.bkg[i]*r2[i]*self.model.c[i,ic]*self.model.a[i,ja]
        for jb in range(0, nb)  : p[ns + ic, na + jb] += self.model.bkg[i]*self.model.bkg[i]*r2[i]*self.model.c[i,ic]*self.model.b[i,jb]
        for jc in range(0, nc)  : p[ns + ic, ns + jc] += self.model.bkg[i]*self.model.bkg[i]*r2[i]*self.model.c[i,ic]*self.model.c[i,jc]
    for ia in range(0, na) : p[ia     ,      ia] += 1
    for ib in range(0, nb) : p[na + ib, na + ib] += 1
    return p, q

# Same as above, more readable, and faster -- use by default
  def pq_einsum(self) :
    pars_nom = Parameters(self.mu, self.data.aux_alphas, self.data.aux_betas, np.zeros(self.model.nc))
    snom = self.model.s_exp(pars_nom)
    nnom = snom + self.model.b_exp(pars_nom)
    r1 = 1 - self.data.n/nnom
    r2 = self.data.n/nnom/nnom
    r3 = 1 - self.data.n/nnom*(1 - snom/nnom)
    qA = np.einsum('i,i,ij->j', snom, r1, self.model.a)
    qB = np.einsum('i,i,ij->j', self.model.bkg, r1, self.model.b)
    qC = np.einsum('i,i,ij->j', self.model.bkg, r1, self.model.c)
    pAA = np.einsum('i,i,ij,ik->jk', snom, r3, self.model.a, self.model.a) + np.identity(self.model.na)
    pAB = np.einsum('i,i,i,ij,ik->jk', snom, self.model.bkg, r2, self.model.a, self.model.b)
    pAC = np.einsum('i,i,i,ij,ik->jk', snom, self.model.bkg, r2, self.model.a, self.model.c)
    pBB = np.einsum('i,i,i,ij,ik->jk', self.model.bkg,self.model.bkg,r2, self.model.b, self.model.b) + np.identity(self.model.nb)
    pBC = np.einsum('i,i,i,ij,ik->jk', self.model.bkg,self.model.bkg,r2, self.model.b, self.model.c)
    pCC = np.einsum('i,i,i,ij,ik->jk', self.model.bkg,self.model.bkg,r2, self.model.c, self.model.c)
    return np.block([[pAA, pAB, pAC], [np.transpose(pAB), pBB, pBC], [np.transpose(pAC), np.transpose(pBC), pCC]]), np.block([qA, qB, qC])
  
  def profile(self) :
    p, q = self.pq_einsum() # can switch to pq_naive for tests
    d = np.linalg.det(p)
    if abs(d) < 1E-8 :
      print('Linear system has an ill-conditioned coefficient matrix (det= %g), returning null result' % d)
      nps =  self.data.aux_alphas, self.data.aux_betas, np.zeros(self.model.nc)
    else :
      v = np.linalg.inv(p).dot(q)
      nps = self.data.aux_alphas - v[:self.model.na], self.data.aux_betas - v[self.model.na:self.model.nsyst], -v[self.model.nsyst:]
    self.min_pars = Parameters(self.mu, *nps, self.model)
    return nps
  
  def profile_nll(self) :
    self.profile()
    return self.model.nll(self.min_pars, self.data)
  
  
# -------------------------------------------------------------------------
class POIMinimizer :
  def __init__(self, data) : 
    self.model = data.model
    self.data = data
  @abstractmethod
  def minimize(self) :
    pass  
  def tmu(self, mu) :
    nll_min, min_pos = self.minimize()
    if nll_min == None : return None, None
    np_min = NPMinimizer(mu, self.data)
    nll_hypo = np_min.profile_nll()
    self.hypo_pars = np_min.min_pars
    return 2*(nll_hypo - nll_min), min_pos
  def asimov_clone(self, mu) :
    clone = copy.copy(self)
    clone.data = Data(self.model).set_expected(self.model.expected_pars(mu))
    return clone

# -------------------------------------------------------------------------
class ScanMinimizer (POIMinimizer) :
  def __init__(self, data, scan_mus) :
    super().__init__(data)
    self.scan_mus = scan_mus
    self.pars = []
    for mu in scan_mus :
      self.pars.append(Parameters(mu, self.data.aux_alphas, self.data.aux_betas, np.zeros(self.model.nc), self.model))

  def minimize(self) :
    self.nlls = np.zeros(self.scan_mus.size)
    for i in range(0, len(self.scan_mus)) :
      np_min = NPMinimizer(self.scan_mus[i], self.data)
      self.nlls[i] = np_min.profile_nll()
      self.pars[i] = np_min.min_pars
      #print('@mu(', i, ') =', mu, self.nlls[i], ahat, bhat)
    smooth_nll = InterpolatedUnivariateSpline(self.scan_mus, self.nlls, k=4)
    minima = smooth_nll.derivative().roots()
    self.nll_min = np.amin(self.nlls)
    self.min_idx = np.argmin(self.nlls)
    min_mu  = self.scan_mus[self.min_idx]
    if len(minima) == 1 :
      interp_min = smooth_nll(minima[0])
      if interp_min < self.nll_min :
        min_mu  = minima[0]
        self.nll_min = interp_min
    self.min_pars = Parameters(min_mu, self.pars[self.min_idx].alphas, self.pars[self.min_idx].betas, self.pars[self.min_idx].gammas)
    return self.nll_min, min_mu

# -------------------------------------------------------------------------
class OptiMinimizer (POIMinimizer) :
  def __init__(self, data, mu0 = 1, bounds = (0, 999999), method = 'scalar') :
    super().__init__(data)
    self.np_min = None
    self.mu0 = mu0
    self.bounds = bounds
    self.method = method
   
  def minimize(self) :    
    def objective(mu) :
      if isinstance(mu, np.ndarray) : mu = mu[0]
      if isinstance(mu, np.ndarray) : mu = mu[0]
      #print('obj',mu, type(mu))
      self.np_min = NPMinimizer(mu, self.data)
      self.nll_min = self.np_min.profile_nll()
      #print('== OptMinimizer: eval at %g -> %g' % (mu, self.nll_min))
      return self.nll_min
    def jacobian(mu) :
      if isinstance(mu, np.ndarray) : mu = mu[0]
      if isinstance(mu, np.ndarray) : mu = mu[0]
      #print('jac', mu, type(mu))
      self.np_min = NPMinimizer(mu, self.data)
      self.np_min.profile()
      #print('jac out', self.np_min.model.grad_poi(self.np_min.min_pars, self.data))
      return np.array([ self.np_min.model.grad_poi(self.np_min.min_pars, self.data) ])
    def hess_p(mu, v) :
      if isinstance(mu, np.ndarray) : mu = mu[0]
      if isinstance(mu, np.ndarray) : mu = mu[0]
      #print('hess', mu, type(mu))
      self.np_min = NPMinimizer(mu, self.data)
      self.np_min.profile()
      #print('hess out', self.np_min.model.hess_poi(self.np_min.min_pars, self.data)*v[0])
      return np.array([ self.np_min.model.hess_poi(self.np_min.min_pars, self.data)*v[0] ])
    #print('Optimizer: start bounded ----------------')
    # print('Optimizer: switch to non-scalar  ----------------')
    if self.method == 'scalar' :
      result = scipy.optimize.minimize_scalar(objective, bounds=self.bounds, method='bounded', options={'xatol': 1e-3 })
    else :
      result = scipy.optimize.minimize(objective, x0=self.mu0, bounds=(self.bounds,), method='L-BFGS-B', jac=jacobian, hessp=hess_p, options={'gtol': 1e-3, 'ftol':1e-3, 'xtol':1e-3 })
      if not result.success:
        result = scipy.optimize.minimize_scalar(objective, bounds=self.bounds, method='bounded', options={'xtol': 1e-3 })
    #print('Optimizer: done ----------------')
    if not result.success :
      print('Minimization failed, details below')
      print(dir(result))
      if hasattr(result, 'status')  : print('status  =', result.status)
      if hasattr(result, 'message') : print('message =', result.message)
      return None, None
    self.min_pars = self.np_min.min_pars
    self.nll_min = result.fun
    self.min_mu = result.x
    return self.nll_min, self.min_mu
