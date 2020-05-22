import numpy as np
import math
from abc import abstractmethod
import json
import matplotlib.pyplot as plt
import copy

# -------------------------------------------------------------------------
class JSONSerializable :
  def __init__(self) :
    pass

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
  @abstractmethod
  def load_jdict(self, jdict) :
    pass
  @abstractmethod
  def dump_jdict(self) :
    pass

# -------------------------------------------------------------------------
class Model (JSONSerializable) :
  def __init__(self, sig = np.array([]), bkg = np.array([]), poi = None, alphas = [], betas = [], gammas = [],
               a = np.ndarray((0,0)), b = np.ndarray((0,0)), c = np.ndarray((0,0)), sigma_alphas = np.ndarray((0,0)), sigma_betas = np.ndarray((0,0)),
               name=None, bins=[], linear_nps = False) :
    super().__init__()
    self.poi    = poi
    self.alphas = alphas
    self.betas  = betas
    self.gammas = gammas
    self.pars = {}
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
      raise ValueError('Input "a" to fastprof model should have a column count (here %g) equal to the number of alpha parameters (%d).' % (a.shape[1], len(alphas)))
    if b.shape != (0,0) and b.shape[1] != len(betas) :
      raise ValueError('Input "b" to fastprof model should have a column count (here %g) equal to the number of beta parameters (%d).' % (b.shape[1], len(betas)))
    if c.shape != (0,0) and c.shape[1] != len(gammas) :
      raise ValueError('Input "c" to fastprof model should have a column count (here %g) equal to the number of gamma parameters (%d).' % (c.shape[1], len(gammas)))
    self.sig = sig
    self.bkg = bkg
    self.a = a
    self.b = b
    self.c = c
    self.sigma_alphas = sigma_alphas
    self.sigma_betas  = sigma_betas
    self.sigma_gammas = np.ndarray((0,0))
    self.name = name
    self.bins = bins
    self.linear_nps = linear_nps
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
    self.diag_alphas = np.diag(1/self.sigma_alphas**2) if self.sigma_alphas.shape != (0,0) else np.identity(self.na)
    self.diag_betas  = np.diag(1/self.sigma_betas **2) if self.sigma_betas .shape != (0,0) else np.identity(self.nb)
    self.diag_gammas = np.diag(1/self.sigma_gammas**2) if self.sigma_gammas.shape != (0,0) else np.zeros((self.nc, self.nc))

  def set_gamma_regularization(self, gamma_regularization) :
    self.sigma_gammas = np.ones(self.nc)*gamma_regularization
    self.init_vars()

  def poi   (self) : return self.poi
  def alphas(self) : return self.alphas
  def betas (self) : return self.betas
  def gammas(self) : return self.gammas

  def s_exp(self, pars) :
    if self.linear_nps : return pars.poi*self.sig*(1 + self.a.dot(pars.alphas))
    ks = np.exp(self.ln_a.dot(pars.alphas))
    return pars.poi*self.sig*ks

  def b_exp(self, pars) :
    if self.linear_nps : return self.bkg*(1 + self.b.dot(pars.betas) + self.c.dot(pars.gammas))
    bexp = self.bkg.copy()
    if self.nb != 0 : bexp *= np.exp(self.ln_b.dot(pars.betas))
    if self.nc != 0 : bexp *= np.exp(self.ln_c.dot(pars.gammas))
    return bexp

  def n_exp(self, pars) : return self.s_exp(pars) + self.b_exp(pars)

  def nll(self, pars, data, offset = True) :
    da = data.aux_alphas - pars.alphas
    db = data.aux_betas  - pars.betas
    dc = pars.gammas
    nexp = self.n_exp(pars)
    try :
      if not offset :
        result = np.sum(nexp - data.n*np.log(nexp)) \
          + 0.5*np.linalg.multi_dot((da, self.diag_alphas, da)) \
          + 0.5*np.linalg.multi_dot((db, self.diag_betas , db)) \
          + 0.5*np.linalg.multi_dot((dc, self.diag_gammas, dc))
      else :
        nexp0 = self.sig + self.bkg
        result = np.sum(nexp - nexp0 - data.n*(np.log(nexp/nexp0))) \
          + 0.5*np.linalg.multi_dot((da, self.diag_alphas, da)) \
          + 0.5*np.linalg.multi_dot((db, self.diag_betas , db)) \
          + 0.5*np.linalg.multi_dot((dc, self.diag_gammas, dc))

      if math.isnan(result) : result = math.inf
      return result
    except Exception as inst:
      print('Fast NLL computation failed with the following exception, returning +Inf')
      print(inst)
      return np.Infinity

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
    
  def expected_pars(self, poi, minimizer = None) :
    if minimizer :
      return minimizer.profile_nps(poi)
    else :
      return Parameters(poi, np.zeros(self.na), np.zeros(self.nb), np.zeros(self.nc), self)

  def generate_data(self, pars) :
    return Data(self, np.random.poisson(self.n_exp(pars)), np.random.normal(pars.alphas, 1), np.random.normal(pars.betas, 1))

  def generate_asimov(self, pars) :
    return Data(self).set_data(self.n_exp(pars), pars.alphas, pars.betas)

  def generate_expected(self, poi, minimizer = None) :
    return self.generate_asimov(self.expected_pars(poi, minimizer))

  def plot(self, pars, data = None, bkg_only = True, variations=[], residuals = False, canvas=None) :
    if canvas == None : canvas = plt.gca()
    if len(self.bins) > 0 :
      grid = [ b['lo_edge'] for b in self.bins ]
      grid.append(self.bins[-1]['hi_edge'])
    else :
      grid = np.linspace(0, self.nbins, self.nbins)
    xvals = [ (grid[i] + grid[i+1])/2 for i in range(0, len(grid) - 1) ]
    nex = self.n_exp(pars)
    if bkg_only :
      yvals = self.b_exp(pars) if not residuals else self.s_exp(pars)
      canvas.hist(xvals, weights=yvals, bins=grid, histtype='step',color='b', linestyle='--', label='bkg-only')
    yvals = nex if not residuals or not data else nex - data.n
    canvas.hist(xvals, weights=yvals, bins=grid, histtype='step',color='b', label='Model')
    if data : 
      yerrs = [ math.sqrt(n) if n > 0 else 0 for n in data.n ]
      yvals = data.n if not residuals else np.zeros(self.nbins)
      canvas.errorbar(xvals, yvals, xerr=[0]*self.nbins, yerr=yerrs, fmt='ko', label='Data')
    canvas.set_xlim(grid[0], grid[-1])
    for v in variations :
      vpars = copy.deepcopy(pars)
      if v[0] in self.alphas : vpars.alphas[self.alphas.index(v[0])] = v[1]
      if v[0] in self.betas  : vpars.betas [self.betas .index(v[0])] = v[1]
      if v[0] in self.gammas : vpars.gammas[self.gammas.index(v[0])] = v[1]
      col = 'r' if len(v) < 3 else v[2]
      style = '--' if v[1] > 0 else '-.'
      canvas.hist(xvals, weights=self.n_exp(vpars), bins=grid, histtype='step',color=col, linestyle=style, label='%s=%+g' %(v[0], v[1]))
      canvas.legend()
    canvas.set_title(self.name)
    canvas.set_xlabel('$' + self.obs_name + '$' + ((' ['  + self.obs_unit + ']') if self.obs_unit != '' else ''))
    canvas.set_ylabel('Events / bin')
    #plt.bar(np.linspace(0,self.sig.size - 1,self.sig.size), self.n_exp(pars), width=1, edgecolor='b', color='', linestyle='dashed')

  @staticmethod
  def create(filename) :
    return Model().load(filename)
  
  def load_jdict(self, jdict) :
    self.sig  = np.array(jdict['signal'])
    self.bkg  = np.array(jdict['background'])
    if 'model_name' in jdict : self.name     = jdict['model_name']
    if 'obs_name'   in jdict : self.obs_name = jdict['obs_name']
    if 'obs_unit'   in jdict : self.obs_unit = jdict['obs_unit']
    if 'bins'       in jdict : self.bins     = np.array(jdict['bins'])
    self.poi = jdict['poi']

    self.alphas = []
    self.a = np.ndarray((self.sig.size, len(jdict['alphas'])))
    for i, spec in enumerate(jdict['alphas']) :
      self.alphas.append(spec['name'])
      self.a[:,i] = spec['impact']
      self.pars[spec['name']] = { 'nominal' : spec['nominal'], 'variation' : spec['variation'] }
    
    self.betas = []
    self.b = np.ndarray((self.sig.size, len(jdict['betas'])))
    for i, spec in enumerate(jdict['betas']) :
      self.betas.append(spec['name'])
      self.b[:,i] = spec['impact']
      self.pars[spec['name']] = { 'nominal' : spec['nominal'], 'variation' : spec['variation'] }

    self.gammas = []
    self.c = np.ndarray((self.sig.size, len(jdict['gammas'])))
    for i, spec in enumerate(jdict['gammas']) :
      self.gammas.append(spec['name'])
      self.c[:,i] = spec['impact']
      self.pars[spec['name']] = { 'nominal' : spec['nominal'], 'variation' : spec['variation'] }

    self.init_vars()
    return self
  
  def dump_jdict(self) :
    jdict = {}
    jdict['signal']     = self.sig.tolist()
    jdict['background'] = self.bkg.tolist()
    jdict['model_name'] = self.name
    jdict['obs_name']   = self.obs_name
    jdict['obs_unit']   = self.obs_unit
    jdict['bins']       = self.bins.toarray()

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

  def closure_exact(self, pars, data) : # add alphas eventually
    eq_b = np.einsum('i,ij,i->j',self.bkg, self.b, (1 - data.n/self.n_exp(pars))) - (data.aux_betas - pars.betas)
    eq_c = np.einsum('i,ij,i->j',self.bkg, self.c, (1 - data.n/self.n_exp(pars)))
    return eq_b, eq_c

  def closure_approx(self, pars, data, order = 1) :  # add alphas eventually
    pars0 = self.expected_pars(pars.poi).set_from_aux(data)
    eq_b = np.einsum('i,ij,i->j',self.bkg, self.b, 1 - data.n/self.n_exp(pars0)*(1 - self.bkg/self.n_exp(pars0)*self.b.dot(pars.betas )))
    eq_b -= (data.aux_betas - pars.betas)
    eq_c = np.einsum('i,ij,i->j',self.bkg, self.c, 1 - data.n/self.n_exp(pars0)*(1 - self.bkg/self.n_exp(pars0)*self.c.dot(pars.gammas)))
    if order > 1 :
      eq_b += np.einsum('i,ij,i,i,i->j', self.bkg, self.b, -data.n/self.n_exp(pars0), self.bkg/self.n_exp(pars0)*self.b.dot(pars.betas ), self.bkg/self.n_exp(pars0)*self.b.dot(pars.betas ))
      eq_c += np.einsum('i,ij,i,i,i->j', self.bkg, self.c, -data.n/self.n_exp(pars0), self.bkg/self.n_exp(pars0)*self.c.dot(pars.gammas), self.bkg/self.n_exp(pars0)*self.c.dot(pars.gammas))
    if order > 2 :
      eq_b += np.einsum('i,ij,i,i,i,i->j', self.bkg, self.b, data.n/self.n_exp(pars0), self.bkg/self.n_exp(pars0)*self.b.dot(pars.betas ), self.bkg/self.n_exp(pars0)*self.b.dot(pars.betas ), self.bkg/self.n_exp(pars0)*self.b.dot(pars.betas ))
      eq_c += np.einsum('i,ij,i,i,i,i->j', self.bkg, self.c, data.n/self.n_exp(pars0), self.bkg/self.n_exp(pars0)*self.c.dot(pars.gammas), self.bkg/self.n_exp(pars0)*self.c.dot(pars.gammas), self.bkg/self.n_exp(pars0)*self.c.dot(pars.gammas))
    if order > 3 :
      eq_b += np.einsum('i,ij,i,i,i,i,i->j', self.bkg, self.b, -data.n/self.n_exp(pars0), self.bkg/self.n_exp(pars0)*self.n.dot(pars.betas ), self.bkg/self.n_exp(pars0)*self.b.dot(pars.betas ), self.bkg/model.n_exp(pars0)*self.b.dot(pars.betas ), self.bkg/model.n_exp(pars0)*self.n.dot(pars.betas ))
      eq_c += np.einsum('i,ij,i,i,i,i,i->j', self.bkg, self.c, -data.n/self.n_exp(pars0), self.bkg/self.n_exp(pars0)*self.c.dot(pars.gammas), self.bkg/self.n_exp(pars0)*self.c.dot(pars.gammas), self.bkg/model.n_exp(pars0)*self.c.dot(pars.gammas), self.bkg/model.n_exp(pars0)*self.c.dot(pars.gammas))
    return eq_b, eq_c

# -------------------------------------------------------------------------
class Parameters :
  def __init__(self, poi = 0, alphas = np.array([]), betas = np.array([]), gammas = np.array([]), model = None) :
    if not isinstance(alphas, np.ndarray) : alphas = np.array(alphas)
    if not isinstance(betas , np.ndarray) : betas  = np.array(betas)
    if not isinstance(gammas, np.ndarray) : gammas = np.array(gammas)
    self.model = model
    self.poi = poi
    if model :
      if alphas.shape[0] == 0 : alphas = np.zeros(model.na)
      if betas .shape[0] == 0 : betas  = np.zeros(model.nb)
      if gammas.shape[0] == 0 : gammas = np.zeros(model.nc)
    self.alphas = alphas
    self.betas = betas
    self.gammas = gammas

  def array(self) : 
    return np.concatenate( ( np.array([ self.poi ]), self.alphas, self.betas, self.gammas ) )
  
  def __str__(self) :
    s = ''
    if self.model == None :
      s += 'poi    = ' + str(self.poi)    + '\n'
      s += 'alphas = ' + str(self.alphas) + '\n'
      s += 'betas  = ' + str(self.betas)  + '\n'
      s += 'gammas = ' + str(self.gammas)
    else :
      s += 'poi    : %-12s = %8.4f' %  (self.model.poi, self.poi) + '\n'
      s += 'alphas : ' + '\n         '.join( [ '%-12s = %8.4f (unscaled : %12.4f)' % (p,v, self.unscaled(p,v)) for p,v in zip(self.model.alphas, self.alphas) ] ) + '\n'
      s += 'betas  : ' + '\n         '.join( [ '%-12s = %8.4f (unscaled : %12.4f)' % (p,v, self.unscaled(p,v)) for p,v in zip(self.model.betas , self.betas ) ] ) + '\n'
      s += 'gammas : ' + '\n         '.join( [ '%-12s = %8.4f (unscaled : %12.4f)' % (p,v, self.unscaled(p,v)) for p,v in zip(self.model.gammas, self.gammas) ] )
    return s

  def __getitem__(self, par):
    if par == self.model.poi : return self.poi
    try :
      i = self.model.alphas.index(par)
      return self.alphas[i]
    except:
      try :
        i = self.model.betas.index(par)
        return self.betas[i]
      except:
        try :
          i = self.model.gammas.index(par)
          return self.gammas[i]
        except:
          raise KeyError('Model parameter %s not found' % par)
    return None

  def set_poi(self, poi) :
    self.poi = poi
    return self

  def set_np(self, par, val, unscaled=False) :
    if unscaled :
      try :
        p = self.model.pars[par]
        val = (val - p['nominal'])/p['variation']
      except:
        raise KeyError('Model parameter %s not found' % par)
    try :
      i = self.model.alphas.index(par)
      self.alphas[i] = val
    except:
      try :
        i = self.model.betas.index(par)
        self.betas[i] = val
      except:
        try :
          i = self.model.gammas.index(par)
          self.gammas[i] = val
        except:
          raise KeyError('Model parameter %s not found' % par)
    return self
  
  def unscaled(self, par, val) :
    try :
      p = self.model.pars[par]
      return p['nominal'] + p['variation']*val
    except:
      raise KeyError('Model parameter %s not found' % par)

  def set_from_aux(self, data) :
    self.alphas = np.array(data.aux_alphas)
    self.betas  = np.array(data.aux_betas )
    return self


# -------------------------------------------------------------------------
class Data (JSONSerializable) :
  def __init__(self, model, n = np.array([]), aux_alphas = np.array([]), aux_betas = np.array([])) :
    super().__init__()
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
  
  def load_jdict(self, jdict) :
    self.n          = np.array(jdict['data']['bin_counts'])
    self.aux_alphas = np.array(jdict['data']['aux_alphas'])
    self.aux_betas  = np.array(jdict['data']['aux_betas'])
    return self

  def dump_jdict(self) :
    jdict = {}
    jdict['data'] = {}
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
