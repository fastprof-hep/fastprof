import numpy as np
import math
import scipy.stats
from abc import abstractmethod
from scipy.interpolate import InterpolatedUnivariateSpline
import json

# -------------------------------------------------------------------------
class Model :
  def __init__(self, sig = np.array([]), bkg = np.array([]), alphas = [], betas = [], gammas = [],
               a = np.ndarray((0,0)), b = np.ndarray((0,0)), c = np.ndarray((0,0))) :
    self.alphas = alphas
    self.betas  = betas
    self.gammas = gammas
    if sig.ndim != 1 or bkg.ndim != 1 or a.ndim != 2 or b.ndim != 2 or c.ndim != 2 :
      raise ValueError('Input data to fastprof model should be 1D vectors for (sig, bkg) and 2D matrices for (a, b)')
    if sig.size != bkg.size :
      raise ValueError('Inputs (sig, bkg) to fastprof model have different dimensions (%d, %d). Both dimensions should be equal to the number of bins in the model.' % (sig.size, bkg.size))
    if a.shape != (0,0) and a.shape[0] != sig.size :
      raise ValueError('Input "a" to fastprof model should have a row count equal to the number of bins (%d). ' % sig.size)
    if b.shape != (0,0) and b.shape[0] != sig.size :
      raise ValueError('Input "b" to fastprof model should have a row count equal to the number of bins (%d). ' % sig.size)
    if c.shape != (0,0) and c.shape[0] != sig.size :
      raise ValueError('Input "c" to fastprof model should have a row count equal to the number of bins (%d). ' % sig.size)
    if a.shape != (0,0) and a.shape[1] != len(alphas) :
      raise ValueError('Input "a" to fastprof model should have a column count equal to the number of alpha parameters (%d).' % len(alphas))
    if b.shape != (0,0) and b.shape[1] != len(betas) :
      raise ValueError('Input "b" to fastprof model should have a column count equal to the number of beta parameters (%d).' % len(betas))
    if c.shape != (0,0) and c.shape[1] != len(gammas) :
      raise ValueError('Input "c" to fastprof model should have a column count equal to the number of gamma parameters (%d).' % len(gammas))
    self.sig = sig
    self.bkg = bkg
    self.a = a
    self.b = b
    self.c = c
    self.na = self.a.shape[1] if self.a.shape != (0,) else 0
    self.nb = self.b.shape[1] if self.b.shape != (0,) else 0
    self.nc = self.c.shape[1] if self.c.shape != (0,) else 0
    self.nbins = self.sig.size
    self.nsyst = self.na + self.nb
    self.n_nps = self.na + self.nb + self.nc
    
  def alphas(self) : return self.alphas
  def betas (self) : return self.betas
  def gammas(self) : return self.gammas

  def s_exp(self, pars) : 
    ds = self.a.dot(pars.alphas)
    return pars.mu*self.sig*(1 + ds)

  def b_exp(self, pars) :
    d = 1
    if self.b.shape != (0,0) : d += self.b.dot(pars.betas)
    if self.c.shape != (0,0) : d += self.c.dot(pars.gammas)
    return self.bkg*d

  def n_exp(self, pars) : return self.s_exp(pars) + self.b_exp(pars)

  def nll(self, pars, data) :
    nexp = self.n_exp(pars)
    da = data.aux_alphas - pars.alphas
    db = data.aux_betas  - pars.betas
    return np.sum(nexp - data.n*np.log(nexp)) + 0.5*np.dot(da,da) + 0.5*np.dot(db,db)

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
    return Data(self, np.random.poisson(self.n_exp(pars)), np.random.normal(pars.alphas, 1), np.random.normal(pars.betas, 1))

  @staticmethod
  def create(filename) :
    return Model().load(filename)
  
  def load(self, filename) :
    with open(filename, 'r') as fd :
      jdict = json.load(fd)
      return self.load_jdict(jdict)

  def save(self, filename) :
    with open(filename, 'w') as fd :
      jdict = self.dump_jdict()
      return json.dump(jdict, fd, ensure_ascii=True, indent=3)
      
  def load_json(self, js) :
    jdict = json.loads(js)
    return self.load_jdict(jdict)
  
  def dump_json(self) :
    jdict = self.dump_jdict()
    return json.dumps(jdict)

  def load_jdict(self, jdict) :
    self.sig = np.array(jdict['signal'])
    self.bkg = np.array(jdict['background'])
    
    self.alphas = []
    self.a = np.ndarray((self.sig.size, len(jdict['alphas'])))
    for i, alpha in enumerate(jdict['alphas']) :
      self.alphas.append(alpha['name'])
      self.a[:,i] = alpha['impact']
    
    self.betas = []
    self.b = np.ndarray((self.sig.size, len(jdict['betas'])))
    for i, beta in enumerate(jdict['betas']) :
      self.betas.append(beta['name'])
      self.b[:,i] = beta['impact']

    self.gammas = []
    self.c = np.ndarray((self.sig.size, len(jdict['gammas'])))
    for i, gamma in enumerate(jdict['gammas']) :
      self.gammas.append(gamma['name'])
      self.c[:,i] = gamma['impact']
    return self
  
  def dump_jdict(self) :
    jdict = {}
    jdict['signal'] = self.sig.tolist()
    jdict['background'] = self.bkg.tolist()

    alphas = []
    for i, alpha in enumerate(self.alphas) :
      alphas.append({ 'name' : alpha, 'impact' : self.a[:,i].tolist() })
    jdict['alphas'] = alphas

    betas = []
    for i, beta in enumerate(self.betas) :
      betas.append({ 'name' : beta, 'impact' : self.b[:,i].tolist() })
    jdict['betas'] = betas

    gammas= []
    for i, gamma in enumerate(self.gammas) :
      gammas.append({ 'name' : gamma, 'impact' : self.c[:,i].tolist() })
    jdict['gammas'] = gammas
    return jdict
    
  def __str__(self) :
    s = ''
    s += 'sig    = ' + str(self.sig)    + '\n'
    s += 'bkg    = ' + str(self.bkg)    + '\n'
    if len(self.alphas) > 0 :
      s += 'alphas = ' + ', '.join(self.alphas) + '\n'
      s += 'a      = ' +       str(self.a)      + '\n'
    if len(self.betas) > 0 :
      s += 'betas = '  + ', '.join(self.betas) + '\n'
      s += 'b      = ' +       str(self.b)     + '\n'
    if len(self.gammas) > 0 :
      s += 'gammas = ' + ', '.join(self.gammas) + '\n'
      s += 'c      = ' +       str(self.c)      + '\n'
    return s

# -------------------------------------------------------------------------
class Parameters :
  def __init__(self, mu, alphas = np.array([]), betas = np.array([]), gammas = np.array([])) :
    if not isinstance(alphas, np.ndarray) : alphas = np.array([alphas])
    if not isinstance(betas , np.ndarray) : betas  = np.array([betas ])
    if not isinstance(gammas, np.ndarray) : gammas = np.array([gammas])
    self.mu = mu
    self.alphas = alphas
    self.betas = betas
    self.gammas = gammas

  def array(self) : 
    return np.concatenate( ( np.array([ self.mu ]), self.alphas, self.betas, self.gammas ) )
  
  def __str__(self) :
    s = ''
    s += 'mu     = ' + str(self.mu)     + '\n'
    s += 'alphas = ' + str(self.alphas) + '\n'
    s += 'betas  = ' + str(self.betas)  + '\n'
    s += 'gammas = ' + str(self.betas)  + '\n'
    return s
  
  
# -------------------------------------------------------------------------
class Data :
  def __init__(self, model, n = np.array([]), aux_alphas = np.array([]), aux_betas = np.array([])) :
    self.model = model
    if n.size > 0 :
      if n.ndim != 1 :
        raise ValueError('Input data "n" should be a 1D vector, got ' + str(n))
      if n.size != self.model.sig.size :
        raise ValueError('Input data "n" should have the same size as "sig" and "bkg", got ' + str(n))
      self.n = n
    else :
      self.n = self.model.sig + self.model.bkg

    if aux_alphas.size > 0 :
      if aux_alphas.size != self.model.a.shape[1] :
        raise ValueError('Input data "aux_alphas" should have the same size as the width of model "a", got ' + str(aux_alphas))
      self.aux_alphas = aux_alphas
    else :
      self.aux_alphas = np.zeros(self.model.a.shape[1])
    
    if aux_betas.size > 0 :
      if aux_betas.size != self.model.b.shape[1] :
        raise ValueError('Input data "aux_beta" should have the same size as the width of model "b", got ' + str(aux_betas))
      self.aux_betas = aux_betas
    else :
      self.aux_betas = np.zeros(self.model.b.shape[1])
    
  def set_n(self, n) : 
    if n.shape != self.n.shape :
      raise ValueError
    self.n = n
    return self
  
  def set_aux(self, aux_alphas, aux_betas) : 
    if aux_alphas.shape != self.aux_alphas.shape :
      raise ValueError
    self.aux_alphas = aux_alphas
    if aux_betas.shape != self.aux_betas.shape :
      raise ValueError
    self.aux_betas = aux_betas
    return self
  
  def set_data(self, n, aux_alphas, aux_betas) :
    self.set_n(n)
    self.set_aux(aux_alphas, aux_betas)
    return self

  def set_expected(self, pars) :
    self.set_data(self.model.n_exp(pars), pars.alphas, pars.betas)
    return self
  
  def set_from_pyhf_data(self, pyhf_data, pyhf_model) : # TODO : actual implementation instead of this hack
    self.set_data(pyhf_data[:-2], pyhf_data[3:4], pyhf_data[2:3])
    return self
  
  def export_pyhf_data(self, pyhf_model) : # TODO : actual implementation instead of this hack
    return np.concatenate( (self.n, self.aux_betas, self.aux_alphas) )

  def __str__(self) :
    s = ''
    s += 'n     = ' + str(self.n)     + '\n'
    s += 'aux_a = ' + str(self.aux_alphas) + '\n'
    s += 'aux_b = ' + str(self.aux_betas)  + '\n'
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
# Same as above, more readable, but slower... [needs migration to include gammas!]
#def pq(self) :
  #snom = self.model.s_exp(self.model.aux_alpha)
  #nnom = snom + self.model.b_exp(self.model.aux_beta)
  #r1 = 1 - self.model.n/nnom
  #r2 = self.model.n/nnom/nnom
  #r3 = 1 - self.model.n/nnom*(1 - snom/nnom)
  #qA = np.einsum('i,i,ij->j', snom, r1, self.model.a)
  #qB = np.einsum('i,i,ij->j', self.model.bkg, r1, self.model.b)
  #pAA = np.einsum('i,i,ij,ik->jk', snom, r3, self.model.a, self.model.a) + np.identity(na)
  #pAB = np.einsum('i,i,i,ij,ik->jk', snom, self.model.bkg, r2, self.model.a, self.model.b)
  #pBB = np.einsum('i,i,i,ij,ik->jk', self.model.bkg,self.model.bkg,r2, self.model.b, self.model.b) + np.identity(self.model.n_beta())
  #return np.block([[pAA, pAB], [np.transpose(pAB), pBB]]), np.block([qA, qB])

  def profile(self) :
    p, q = self.pq()
    v = np.linalg.inv(p).dot(q)
    nps = self.data.aux_alphas - v[:self.model.na], self.data.aux_betas - v[self.model.na:self.model.nsyst], -v[self.model.nsyst:]
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
      self.pars.append(Parameters(mu, self.data.aux_alphas, self.data.aux_betas, np.zeros(self.model.nc)))
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
    self.min_pars = Parameters(min_mu, self.pars[self.min_idx].alphas, self.pars[self.min_idx].betas, self.pars[self.min_idx].gammas)
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
      result = scipy.optimize.minimize_scalar(objective, bounds=self.bounds, method='bounded', options={'xatol': 1e-3 })
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
  
