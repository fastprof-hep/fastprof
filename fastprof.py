import numpy as np
import math
import scipy.stats
from abc import abstractmethod
from scipy.interpolate import InterpolatedUnivariateSpline

# -------------------------------------------------------------------------
class Model :
  def __init__(self, sig, bkg, a, b, check = True) :
    if check:
      if sig.ndim != 1 or bkg.ndim != 1 or a.ndim != 2 or b.ndim != 2 :
        raise ValueError('Input data to fastprof model should be 1D vectors for (sig, bkg) and 2D matrices for (a, b)')
      if sig.size != bkg.size :
        raise ValueError('Inputs (sig, bkg) to fastprof model have different dimensions: got ' + str(dims) + '. All dimensions should be equal to the number of bins.')
      if a.shape[0] != sig.size or b.shape[0] != sig.size :
        raise ValueError('Inputs (a, b) to fastprof model should have a row count equal to the number of bins,  ' + str(dims[0]))
    self.sig = sig
    self.bkg = bkg
    self.a = a
    self.b = b
    
  def n_bins(self)  : return self.sig.size
  def n_alpha(self) : return self.a.shape[1]
  def n_beta(self)  : return self.b.shape[1]
  def n_syst(self)  : return self.n_alpha() + self.n_beta()

  def s_exp(self, pars) : 
    ds = self.a.dot(pars.alpha)
    return pars.mu*self.sig*(1 + ds)

  def b_exp(self, pars) :
    db = self.b.dot(pars.beta)
    return self.bkg*(1 + db)

  def n_exp(self, pars) : return self.s_exp(pars) + self.b_exp(pars)

  def nll(self, pars, data) :
    ntot = self.n_exp(pars)
    da = data.aux_alpha - pars.alpha
    db = data.aux_beta  - pars.beta
    return np.sum(ntot - data.n*np.log(ntot)) + 0.5*np.dot(da,da) + 0.5*np.dot(db,db)

  #def derivatives(self, pars) :
    #der_a = np.zeros(self.n_alpha())
    #der_b = np.zeros(self.n_beta())
    #da = self.aux_alpha - alpha
    #db = self.aux_beta  - beta
    #stot = self.s_exp(pars)
    #r1 = 1 - self.n/self.n_exp(pars)
    #return np.dot(stot*r1, self.a) - da, np.dot(self.bkg*r1, self.b) - db

  def grad_poi(self, pars, data) :
    sexp = self.s_exp(pars) # This is for a "mu" POI, i.e. d(S_i)/dmu = S_i^exp (true for S_i = mu S_i^exp).
    nexp = sexp + self.b_exp(pars) # This is for a "mu" POI, i.e. d(S_i)/dmu = S_i^exp (true for S_i = mu S_i^exp).
    #r1 = 1 - data.n/nexp
    #return sexp.dot(r1)
    s = 0
    for i in range(0, sexp.size) : s += sexp[i]*(1 - data.n[i]/nexp[i])
    return s
  
  def hess_poi(self, pars, data) : # Hessian wrt to a mu-type POI
    sexp = self.s_exp(pars) # This is for a "mu" POI, i.e. d(S_i)/dmu = S_i^exp (true for S_i = mu S_i^exp).
    nexp = sexp + self.b_exp(pars) # This is for a "mu" POI, i.e. d(S_i)/dmu = S_i^exp (true for S_i = mu S_i^exp).
    s = 0
    for i in range(0, sexp.size) : s += sexp[i]*data.n[i]/nexp[i]**2
    return s
    

  def generate_data(self, pars) :
    return Data(self, np.random.poisson(self.n_exp(pars)), np.random.normal(pars.alpha, 1), np.random.normal(pars.beta, 1))

  def __str__(self) :
    s = ''
    s += 'sig   = ' + str(self.sig)   + '\n'
    s += 'bkg   = ' + str(self.bkg)   + '\n'
    s += 'a     = ' + str(self.a)     + '\n'
    s += 'b     = ' + str(self.b)
    return s

# -------------------------------------------------------------------------
class Parameters :
  def __init__(self, mu, alpha, beta) :
    if not isinstance(alpha, np.ndarray) : alpha = np.array([alpha])
    if not isinstance(beta , np.ndarray) : beta  = np.array([beta])
    self.mu = mu
    self.alpha = alpha
    self.beta = beta

  def array(self) : 
    return np.concatenate( ( np.array([ self.mu ]), self.alpha, self.beta ) )
  
  def __str__(self) :
    s = ''
    s += 'mu    = ' + str(self.mu)    + '\n'
    s += 'alpha = ' + str(self.alpha) + '\n'
    s += 'beta  = ' + str(self.beta)  + '\n'
    return s
  
  
# -------------------------------------------------------------------------
class Data :
  def __init__(self, model, n = np.array([]), aux_alpha = np.array([]), aux_beta = np.array([]), check = True) :
    self.model = model
    if n.size > 0 :
      if check:
        if n.ndim != 1 :
          raise ValueError('Input data "n" should be a 1D vector, got ' + str(n))
        if n.size != self.model.sig.size :
          raise ValueError('Input data "n" should have the same size as "sig" and "bkg", got ' + str(n))
      self.n = n
    else :
      self.n = self.model.sig + self.model.bkg

    if aux_alpha.size > 0 :
      if check :
        if aux_alpha.size != self.model.a.shape[1] :
          raise ValueError('Input data "aux_alpha" should have the same size as the width of model "a", got ' + str(aux_alpha))
      self.aux_alpha = aux_alpha
    else :
      self.aux_alpha = np.zeros(self.model.a.shape[1])
    
    if aux_beta.size > 0 :
      if check :
        if aux_beta.size != self.model.b.shape[1] :
          raise ValueError('Input data "aux_beta" should have the same size as the width of model "b", got ' + str(aux_beta))
      self.aux_beta = aux_beta
    else :
      self.aux_beta = np.zeros(self.model.b.shape[1])

  def set_n(self, n) : 
    if n.shape != self.n.shape :
      raise ValueError
    self.n = n
    return self
  
  def set_aux(self, aux_alpha, aux_beta) : 
    if aux_alpha.shape != self.aux_alpha.shape :
      raise ValueError
    self.aux_alpha = aux_beta
    if aux_beta.shape != self.aux_beta.shape :
      raise ValueError
    self.aux_beta = aux_beta
    return self
  
  def set_data(self, n, aux_alpha, aux_beta) :
    self.set_n(n)
    self.set_aux(aux_alpha, aux_beta)
    return self

  def set_expected(self, pars) :
    self.set_data(self.model.n_exp(pars), pars.alpha, pars.beta)
    return self
  
  def set_from_pyhf_data(self, pyhf_data, pyhf_model) :
    self.set_data(pyhf_data[:-2], pyhf_data[3:4], pyhf_data[2:3])
    return self
  
  def export_pyhf_data(self, pyhf_model) : # TODO : actual implementation instead of this hack
    return np.concatenate( (self.n, self.aux_beta, self.aux_alpha) )

  def __str__(self) :
    s = ''
    s += 'n     = ' + str(self.n)     + '\n'
    s += 'aux_a = ' + str(self.aux_alpha) + '\n'
    s += 'aux_b = ' + str(self.aux_beta)  + '\n'
    return s

# -------------------------------------------------------------------------
class TestStatistic :
  def __init__(self) : 
    pass
  def __float__(self) :
    return value()
  @abstractmethod
  def value() :
    pass  
  @abstractmethod
  def asymptotic_cl() :
    pass
    
class TMu(TestStatistic) :
  def __init__(self, twice_dll) :
    self.value = twice_dll
  def value(self) : 
    return self.value
  def asymptotic_cl(self) :
    return scipy.stats.norm.sf(math.sqrt(self.value))
  
class QMu(TestStatistic) :
  def __init__(self, twice_dll, test_mu, best_mu) :
    self.value = twice_dll if best_mu < test_mu else None
  def value(self) :
    return self.value
  def asymptotic_cl(self) :
    return scipy.stats.norm.sf(math.sqrt(self.value)) if (self.value != None and self.value > 0) else 0.5

# -------------------------------------------------------------------------
class NPMinimizer :
  def __init__(self, mu, data) :
    self.model = data.model
    self.mu = mu
    self.data = data
    
  def pq(self) :
    p = np.zeros((self.model.n_syst(), self.model.n_syst()))
    q = np.zeros(self.model.n_syst())
    pars_obs = Parameters(self.mu, self.data.aux_alpha, self.data.aux_beta)
    sobs = self.model.s_exp(pars_obs)
    nobs = sobs + self.model.b_exp(pars_obs)
    r1 = 1 - self.data.n/nobs
    r2 = self.data.n/nobs/nobs
    r3 = 1 - self.data.n/nobs*(1 - sobs/nobs)
    #print('PQ| model = ',self. model)
    #print('PQ| mu = ', self.mu) 
    #print('PQ| data = ', self.data)
    #print('PQ| p_obs = ', pars_obs)
    #print('PQ| sobs = ', sobs)
    #print('PQ| nobs = ', nobs)
    na = self.model.n_alpha()
    nb = self.model.n_beta()
    for i in range(0, self.model.n_bins()) :
      for ia in range(0, na) :
        q[ia] += sobs[i]*r1[i]*self.model.a[i,ia]
        for ja in range(0, na) :  p[ia,      ja] += sobs[i]                  *r3[i]*self.model.a[i,ia]*self.model.a[i,ja]
        for jb in range(0, nb)  : p[ia, na + jb] += sobs[i]*self.model.bkg[i]*r2[i]*self.model.a[i,ia]*self.model.b[i,jb]
      for ib in range(0, nb) :
        q[na + ib] += self.model.bkg[i]*r1[i]*self.model.b[i,ib]
        for ja in range(0, na) :  p[na + ib,      ja] += sobs[i]          *self.model.bkg[i]*r2[i]*self.model.b[i,ib]*self.model.a[i,ja]
        for jb in range(0, nb)  : p[na + ib, na + jb] += self.model.bkg[i]*self.model.bkg[i]*r2[i]*self.model.b[i,ib]*self.model.b[i,jb]
    for ia in range(0, na) : p[ia     ,      ia] += 1
    for ib in range(0, nb) : p[na + ib, na + ib] += 1
    return p, q
# Same as above, more readable, but slower...
#def pq(self) :
  #sobs = self.model.s_exp(self.model.aux_alpha)
  #nobs = sobs + self.model.b_exp(self.model.aux_beta)
  #r1 = 1 - self.model.n/nobs
  #r2 = self.model.n/nobs/nobs
  #r3 = 1 - self.model.n/nobs*(1 - sobs/nobs)
  #qA = np.einsum('i,i,ij->j', sobs, r1, self.model.a)
  #qB = np.einsum('i,i,ij->j', self.model.bkg, r1, self.model.b)
  #pAA = np.einsum('i,i,ij,ik->jk', sobs, r3, self.model.a, self.model.a) + np.identity(na)
  #pAB = np.einsum('i,i,i,ij,ik->jk', sobs, self.model.bkg, r2, self.model.a, self.model.b)
  #pBB = np.einsum('i,i,i,ij,ik->jk', self.model.bkg,self.model.bkg,r2, self.model.b, self.model.b) + np.identity(self.model.n_beta())
  #return np.block([[pAA, pAB], [np.transpose(pAB), pBB]]), np.block([qA, qB])

  def profile(self) :
    p, q = self.pq()
    v = np.linalg.inv(p).dot(q)
    nps = self.data.aux_alpha - v[:self.model.n_alpha()], self.data.aux_beta - v[self.model.n_alpha():]
    self.min_pars = Parameters(self.mu, *nps)
    return nps
  
  def profile_nll(self) :
    self.profile()
    return self.model.nll(self.min_pars, self.data)
  
# -------------------------------------------------------------------------
class ScanMinimizer :
  def __init__(self, data, scan_mus) :
    self.model = data.model
    self.data = data
    self.scan_mus = scan_mus
    self.pars = []
    for mu in scan_mus :
      self.pars.append(Parameters(mu, self.data.aux_alpha, self.data.aux_beta))
  def minimize(self, debug = False) :
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
    self.min_pars = Parameters(min_mu, self.pars[self.min_idx].alpha, self.pars[self.min_idx].beta)
    return self.nll_min, min_mu

# -------------------------------------------------------------------------
class OptiMinimizer :
  def __init__(self, data, x0 = 1, bounds = (0.1, 20), method = 'scalar' ) :
    self.model = data.model
    self.data = data
    self.np_min = None
    self.x0 = x0
    self.bounds = bounds
    self.method = method
   
  def minimize(self, debug = False) :    
    def objective(mu) :
      if isinstance(mu, np.ndarray) : mu = mu[0]
      if isinstance(mu, np.ndarray) : mu = mu[0]
      #print('obj',mu, type(mu))
      self.np_min = NPMinimizer(mu, self.data)
      self.nll_min = self.np_min.profile_nll()
      #print('OptMinimizer: eval at %g -> %g' % (mu, self.nll_min))
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
      result = scipy.optimize.minimize_scalar(objective, bounds=self.bounds, method='bounded', options={'xtol': 1e-3 })
    else :
      result = scipy.optimize.minimize(objective, x0=self.x0, bounds=(self.bounds,), method='L-BFGS-B', jac=jacobian, hessp=hess_p, options={'gtol': 1e-3, 'ftol':1e-3, 'xtol':1e-3 })
      if not result.success:
        result = scipy.optimize.minimize_scalar(objective, bounds=self.bounds, method='bounded', options={'xtol': 1e-3 })
    #print('Optimizer: done ----------------')
    if not result.success :
      print(dir(result))
      print(result.status)
      raise FloatingPointError(result.message)
    self.min_pars = self.np_min.min_pars
    self.nll_min = result.fun
    self.min_mu = result.x
    return self.nll_min, self.min_mu

  
