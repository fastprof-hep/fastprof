import numpy as np
import math
from abc import abstractmethod
import json
import matplotlib.pyplot as plt
import copy

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
    self.init_vars()

  def init_vars(self) :
    self.na = self.a.shape[1] if self.a.shape != (0,) else 0
    self.nb = self.b.shape[1] if self.b.shape != (0,) else 0
    self.nc = self.c.shape[1] if self.c.shape != (0,) else 0
    self.nbins = self.sig.size
    self.nsyst = self.na + self.nb
    self.n_nps = self.na + self.nb + self.nc
    self.ln_a = np.log(1 + self.a)
    self.ln_b = np.log(1 + self.b)
    self.ln_c = np.log(1 + self.c)
    
  def alphas(self) : return self.alphas
  def betas (self) : return self.betas
  def gammas(self) : return self.gammas

  def s_exp(self, pars) :
    ks = np.exp(self.ln_a.dot(pars.alphas))
    return pars.mu*self.sig*ks

  def b_exp(self, pars) :
    bexp = self.bkg.copy()
    if self.b.shape[1] != 0 : bexp *= np.exp(np.log(1 + self.b).dot(pars.betas))
    if self.c.shape[1] != 0 : bexp *= np.exp(np.log(1 + self.c).dot(pars.gammas))
    return bexp

  def n_exp(self, pars) : return self.s_exp(pars) + self.b_exp(pars)

  def nll_naive(self, pars, data) :
    nexp = self.n_exp(pars)
    da = data.aux_alphas - pars.alphas
    db = data.aux_betas  - pars.betas
    poisson = 0
    for nob, nex in zip(data.n, nexp) :
      if nex > 0 :
        poisson += nex - nob*math.log(nex)
      else :
        if nob == 0 :
          poisson += nex
        else :
          print('Warning: negative expected yields for the parameter values below, returning +INF')
          print(pars)
          print(nexp)
          print(data)
          return np.Infinity
    return poisson + 0.5*np.dot(da,da) + 0.5*np.dot(db,db)

  def nll(self, pars, data) :
    nexp = self.n_exp(pars)
    da = data.aux_alphas - pars.alphas
    db = data.aux_betas  - pars.betas
    try :
      return np.sum(nexp - data.n*np.log(nexp)) + 0.5*np.dot(da,da) + 0.5*np.dot(db,db)
    except Exception as inst:
      print('Fast NLL computation failed with the following exception, switching to slower-but-safer method')
      print(inst)
      return self.nll_naive(pars, data)

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
    
  def expected_pars(self, mu) :
    return Parameters(mu, np.zeros(self.na), np.zeros(self.nb), np.zeros(self.nc))

  def generate_data(self, pars) :
    return Data(self, np.random.poisson(self.n_exp(pars)), np.random.normal(pars.alphas, 1), np.random.normal(pars.betas, 1))

  def plot(self, pars, data = None, bkg_only = True, variations=[], residuals = False) :
    #print(np.linspace(0,self.sig.size - 1,self.sig.size))
    #print(self.b_exp(pars))
    grid = np.linspace(0,self.nbins, self.nbins)
    nex = self.n_exp(pars)
    if bkg_only :
      yvals = self.b_exp(pars) if not residuals else -self.s_exp(pars)
      plt.hist(grid, weights=yvals, bins=grid, histtype='step',color='b', linestyle='--', label='bkg-only')
    yvals = nex if not residuals else np.zeros(self.nbins)
    plt.hist(grid, weights=yvals, bins=grid, histtype='step',color='b', label='Model')
    if data : 
      yerrs = [ math.sqrt(n) if n > 0 else 0 for n in data.n ]
      yvals = [ data.n[i] if not residuals else data.n[i] - nex[i] for i in range(0, self.nbins) ]
      plt.errorbar(grid + 0.5, yvals, xerr=[0]*self.nbins, yerr=yerrs, fmt='ko', label='Data')
    plt.xlim(0, self.nbins)
    for v in variations :
      vpars = copy.deepcopy(pars)
      if v[0] in self.alphas : vpars.alphas[self.alphas.index(v[0])] = v[1]
      if v[0] in self.betas  : vpars.betas [self.betas .index(v[0])] = v[1]
      if v[0] in self.gammas : vpars.gammas[self.gammas.index(v[0])] = v[1]
      col = 'r' if len(v) < 3 else v[2]
      plt.hist(grid, weights=self.n_exp(vpars), bins=grid, histtype='step',color=col, linestyle='--', label='%s=%+g' %(v[0], v[1]))
      plt.legend()
    #plt.bar(np.linspace(0,self.sig.size - 1,self.sig.size), self.n_exp(pars), width=1, edgecolor='b', color='', linestyle='dashed')

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
    self.init_vars()
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
    if not isinstance(alphas, np.ndarray) : alphas = np.array(alphas)
    if not isinstance(betas , np.ndarray) : betas  = np.array(betas)
    if not isinstance(gammas, np.ndarray) : gammas = np.array(gammas)
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
    s += 'gammas = ' + str(self.gammas) + '\n'
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
      if aux_alphas.size != self.model.na :
        raise ValueError('Input data "aux_alphas" should have the same size as the "alphas" of the model (%d), got %s.' % (self.model.na, str(aux_alphas)))
      self.aux_alphas = aux_alphas
    else :
      self.aux_alphas = np.zeros(self.model.a.shape[1])

    if aux_betas.size > 0 :
      if aux_betas.size != self.model.b.shape[1] :
        raise ValueError('Input data "aux_beta" should have the same size as the "betas" of the model (%d), got %s.' % (self.model.nb, str(aux_betas)))
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
    self.n          = np.array(jdict['data']['bin_counts'])
    self.aux_alphas = np.array(jdict['data']['aux_alphas'])
    self.aux_betas  = np.array(jdict['data']['aux_betas'])
    return self

  def dump_jdict(self) :
    jdict = {}
    jdict['data']['bin_counts'] = self.n.tolist()
    jdict['data']['aux_alphas'] = self.aux_alphas.tolist()
    jdict['data']['aux_betas' ] = self.aux_betas .tolist()
    return jdict

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
