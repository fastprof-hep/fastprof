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
      jdict = self.dump_jdict(jdict)
      return json.dump(jdict, fd, ensure_ascii=True, indent=3)
  def load_json(self, js) :
    jdict = json.loads(js)
    return self.load_jdict(jdict)
  def dump_json(self) :
    jdict = self.dump_jdict(jdict)
    return json.dumps(jdict)
  def dump_jdict(self) :
    jdict = {}
    self.fill_jdict(jdict)
    return jdict
  def load_field(self, key, dic, default = None, types = []) :
    if not key in dic :
      if default != None : return default
      raise KeyError('Key %s not found in JSON dictionary' % key)
    val = dic[key]
    if not isinstance(types, list) : types = [ types ]
    if types != [] and not any([isinstance(val, t) for t in types]) :
      raise TypeError('Object at key %s in JSON dictionary has type %s, not the expected %s' % 
                      (key, val.__class__.__name__, '|'.join([t.__name__ for t in types])))
    if types == [ list ] : val = np.array(val)
    return val
  @abstractmethod
  def load_jdict(self, jdict) :
    pass
  @abstractmethod
  def fill_jdict(self, jdict) :
    pass

class ModelPOI(JSONSerializable) :
  def __init__(self, name = '', min_val = None, max_val = None) :
    self.name = name
    self.min_val = min_val
    self.max_val = max_val
  def load_jdict(self, jdict) : 
    self.name = self.load_field('name', jdict, '', str)
    self.min_val = self.load_field('min_val', jdict, '', [int, float])
    self.max_val = self.load_field('max_val', jdict, '', [int, float])
    return self
  def fill_jdict(self, jdict) :
    jdict['name'] = self.name
    jdict['min_val'] = self.min_val
    jdict['max_val'] = self.max_val

class ModelAux(JSONSerializable) :
  def __init__(self, name = '', min_val = None, max_val = None) :
    self.name = name
    self.min_val = min_val
    self.max_val = max_val
  def load_jdict(self, jdict) : 
    self.name = self.load_field('name', jdict, '', str)
    self.min_val = self.load_field('min_val', jdict, '', [int, float])
    self.max_val = self.load_field('max_val', jdict, '', [int, float])
    return self
  def fill_jdict(self, jdict) :
    jdict['name'] = self.name
    jdict['min_val'] = self.min_val
    jdict['max_val'] = self.max_val
  
class ModelNP(JSONSerializable) :
  def __init__(self, name = '', nominal_value = 0, variation = 1, constraint = None, aux_obs = None) :
    self.name = name
    self.nominal_value = nominal_value
    self.variation = variation
    self.constraint = constraint
    self.aux_obs = aux_obs
  def is_free(self) :
    return self.constraint == None
  def generate_aux(self, value) :
    if self.constraint == None : return 0
    return np.random.normal(value, self.constraint)
  def load_jdict(self, jdict) : 
    self.name = self.load_field('name', jdict, '', str)
    self.nominal_value = self.load_field('nominal_val', jdict, None, [int, float])
    self.variation = self.load_field('variation', jdict, None, [int, float])
    self.constraint = self.load_field('constraint', jdict)
    if self.constraint != None :
      self.aux_obs = self.load_field('aux_obs', jdict, '', str)
    else :
      self.aux_obs = None
    return self
  def fill_jdict(self, jdict) :
    jdict['name'] = self.name
    jdict['nominal_val'] = self.nominal_value
    jdict['variation'] = self.variation
    jdict['constraint'] = self.constraint
    jdict['aux_obs'] = self.aux_obs

class Sample(JSONSerializable) :
  def __init__(self, name = '', norm = '', nominal_norm = None, nominal_yields = None, impacts = None) :
    self.name = name
    self.norm_expr = norm
    self.nominal_norm = nominal_norm
    self.nominal_yields = nominal_yields
    self.impacts = impacts
  def norm(self, pars) :
    try:
      return eval(self.norm_expr, pars.dict())/self.nominal_norm
    except Exception as inst:
      print('Error while evaluating the normalization %s of sample %s.' % (self.norm_expr, self.name))
      print(inst)
      return None
  def load_jdict(self, jdict) : 
    self.name = self.load_field('name', jdict, '', str)
    self.norm_expr = self.load_field('norm', jdict, '', str)
    self.nominal_norm = self.load_field('nominal_norm', jdict, None, [float, int])
    self.nominal_yields = self.load_field('nominal_yields', jdict, None, list)
    self.impacts = self.load_field('impacts', jdict, None, dict)
    return self
  def fill_jdict(self, jdict) :
    jdict['name'] = self.name
    jdict['norm'] = self.norm_expr
    jdict['nominal_norm'] = self.nominal_norm
    jdict['nominal_yields'] = self.nominal_yields
    jdict['impacts'] = self.impacts
  
class Channel(JSONSerializable) :
  def __init__(self, name = '', chan_type = 'count', bins = []) :
    self.name = name
    self.type = chan_type
    self.bins = bins
    self.samples = {}
  def dim(self) :
    return len(self.bins)
  def load_jdict(self, jdict) : 
    self.name = jdict['name']
    self.type = jdict['type']
    if self.type == 'binned_range' :
      self.bins = np.array(jdict['bins'])
      self.obs_name = self.load_field('obs_name', jdict, '', str)
      self.obs_unit = self.load_field('obs_unit', jdict, '', str)
      for json_sample in jdict['samples'] :
        sample = Sample()
        sample.load_jdict(json_sample)
        self.samples[sample.name] = sample
    return self
  def fill_jdict(self, jdict) :
    jdict['name'] = self.name
    jdict['type'] = self.type
    if self.type == 'binned_range' :
      jdict['bins'] = self.bins
      jdict['obs_name'] = self.obs_name
      jdict['obs_unit'] = self.obs_unit
      jdict['samples'] = []
      for sample in self.samples : jdict['samples'].append(sample.dump_jdict())

# -------------------------------------------------------------------------
class Model (JSONSerializable) :
  def __init__(self, pois = [], nps = [], aux_obs = [], channels = [], linear_nps = False) :
    super().__init__()
    self.pois = { poi.name : poi for poi in pois }
    self.nps  = {}
    for np in nps :
      if not np.is_free() : self.nps[par.name] = par
    ncons = len(self.nps)
    for np in nps :
      if np.is_free() : self.nps[par.name] = par
    self.aux_obs = { par.name : par for par in aux_obs }
    if len(self.aux_obs) != ncons :
      raise ValueError('Number of auxiliary observables (%d) does not match the number of constrained NPs (%d)' % (len(self.aux_obs), self.ncons))
    self.channels = { channel.name : channel for channel in channels }
    self.linear_nps = linear_nps
    self.init_vars()

  def init_vars(self) :
    self.npois = len(self.pois)
    self.nnps  = len(self.nps)
    self.ncons = len(self.aux_obs)
    self.nfree = self.nnps - self.ncons
    self.samples = {}
    self.sample_indices = {}
    self.channel_offsets = {}
    self.nbins = 0
    for channel in self.channels.values() :
      self.channel_offsets[channel.name] = self.nbins
      self.nbins += channel.dim()
      for sample in channel.samples.values() :
        if not sample.name in self.sample_indices :
          self.samples[sample.name] = sample
          self.sample_indices[sample.name] = len(self.sample_indices)
    self.nominal_yields = np.zeros((len(self.sample_indices), self.nbins))
    self.impacts = np.zeros((len(self.sample_indices), self.nbins, len(self.nps)))
    for channel in self.channels.values() :
      for sample in channel.samples.values() :
        self.nominal_yields[self.sample_indices[sample.name], self.channel_offsets[channel.name]:] = sample.nominal_yields
        for p, par in enumerate(self.nps.values()) :
          self.impacts[self.sample_indices[sample.name], self.channel_offsets[channel.name]:, p] = sample.impacts[par.name]
    self.log_impacts = np.log(1 + self.impacts)
    self.constraint_hessian = np.zeros((self.nnps, self.nnps))
    self.np_nominal_values = np.array([ par.nominal_value for par in self.nps.values() ])
    self.np_variations     = np.array([ par.variation     for par in self.nps.values() ])
    for p, par in enumerate(self.nps.values()) :
      if par.constraint == None : break # we've reached the end of the constrained NPs in the NP list
      self.constraint_hessian[p,p] = 1/par.constraint**2

  def set_constraint(self, par, val) :
    for par in self.nps :
      if par.name == par or par == None : par.constraint = val
    self.init_vars()

  def n_exp(self, pars) :
    nnom = (self.nominal_yields.T*np.array([ sample.norm(pars) for sample in self.samples.values() ])).T
    if self.linear_nps : 
      return nnom*(1 + self.impacts.dot(pars.nps))
    else :
      return nnom*np.exp(self.log_impacts.dot(pars.nps))

  def tot_exp(self, pars, floor = None) :
    ntot = self.n_exp(pars).sum(axis=0)
    return ntot if floor == None else np.maximum(ntot, floor)

  def nll(self, pars, data, offset = True, floor = None, no_constraints=False) :
    delta = data.aux_obs - pars.nps
    ntot = self.tot_exp(pars, floor)
    try :
      if not offset :
        result = np.sum(ntot - data.counts*np.log(ntot))
      else :
        nexp0 = self.nominal_yields.sum(axis=0)
        result = np.sum(ntot - nexp0 - data.counts*(np.log(ntot/nexp0)))
      if not no_constraints :
         result += 0.5*np.linalg.multi_dot((delta, self.constraint_hessian, delta))
      if math.isnan(result) : result = math.inf
      return result
    except Exception as inst:
      print('Fast NLL computation failed with the following exception, returning +Inf')
      print(inst)
      return np.Infinity

  def plot(self, pars, data = None, channel = None, exclude = [], variations = [], residuals = False, canvas=None) :
    if canvas == None : canvas = plt.gca()
    if not isinstance(exclude, list) : exclude = [ exclude ]
    if channel == None :
      channel = self.channels[list(self.channels)[0]]
    else :
      if not channel in self.channels : raise ValueError('ERROR: Channel %s is not defined.' % channel)
      channel = self.channels[channel]
    if len(channel.bins) > 0 :
      grid = [ b['lo_edge'] for b in channel.bins ]
      grid.append(channel.bins[-1]['hi_edge'])
    else :
      grid = np.linspace(0, channel.nbins, channel.nbins)
    xvals = [ (grid[i] + grid[i+1])/2 for i in range(0, len(grid) - 1) ]
    offset = self.channel_offsets[channel.name]
    nexp = self.n_exp(pars)[:,offset:offset + channel.dim()]
    if len(exclude) == 0 :
      tot_exp = nexp.sum(axis=0)
      line_style = '-'
      title = 'Model'
    else :
      samples = []
      for ex in exclude :
        if not ex in channel.samples : raise ValueError('Sample %s is not defined.' % ex)
        samples.append(list(channel.samples).index(ex))
      tot_exp = nexp[samples,:].sum(axis=0)
      line_style = '--'
      title = 'Model excluding ' + ','.join(exclude)     
    yvals = tot_exp if not residuals or not data else tot_exp - data.counts
    canvas.hist(xvals, weights=yvals, bins=grid, histtype='step',color='b', linestyle=line_style, label=title)
    if data : 
      yerrs = [ math.sqrt(n) if n > 0 else 0 for n in data.counts ]
      yvals = data.counts if not residuals else np.zeros(channel.nbins)
      canvas.errorbar(xvals, yvals, xerr=[0]*channel.dim(), yerr=yerrs, fmt='ko', label='Data')
    canvas.set_xlim(grid[0], grid[-1])
    for v in variations :
      vpars = copy.deepcopy(pars)
      vpars.set(v[0], v[1])
      col = 'r' if len(v) < 3 else v[2]
      style = '--' if v[1] > 0 else '-.'
      tot_exp = self.n_exp(vpars)[:,offset:offset + channel.nbins].sum(axis=0)
      canvas.hist(xvals, weights=tot_exp, bins=grid, histtype='step',color=col, linestyle=style, label='%s=%+g' %(v[0], v[1]))
      canvas.legend()
    canvas.set_title(self.name)
    canvas.set_xlabel('$' + channel.obs_name + '$' + ((' ['  + channel.obs_unit + ']') if channel.obs_unit != '' else ''))
    canvas.set_ylabel('Events / bin')
    #plt.bar(np.linspace(0,self.sig.size - 1,self.sig.size), self.n_exp(pars), width=1, edgecolor='b', color='', linestyle='dashed')

# TODO: update to a proper POI scheme
  def grad_poi(self, pars, data) :
    sexp = self.s_exp(pars) # This is for a "mu" POI, i.e. d(S_i)/dmu = S_i^exp (true for S_i = mu S_i^exp).
    nexp = sexp + self.b_exp(pars) # This is for a "mu" POI, i.e. d(S_i)/dmu = S_i^exp (true for S_i = mu S_i^exp).
    s = 0
    for i in range(0, sexp.size) : s += sexp[i]*(1 - data.n[i]/nexp[i])
    return s
  
# TODO: update to a proper POI scheme  
  def hess_poi(self, pars, data) : # Hessian wrt to a mu-type POI
    sexp = self.s_exp(pars) # This is for a "mu" POI, i.e. d(S_i)/dmu = S_i^exp (true for S_i = mu S_i^exp).
    nexp = sexp + self.b_exp(pars) # This is for a "mu" POI, i.e. d(S_i)/dmu = S_i^exp (true for S_i = mu S_i^exp).
    s = 0
    for i in range(0, sexp.size) : s += sexp[i]*data.n[i]/nexp[i]**2
    return s
    
  def expected_pars(self, pois, minimizer = None) :
    if not isinstance(pois, np.ndarray) : pois = np.array([ pois ])
    pars = Parameters(pois, np.zeros(self.nnps), self)
    if minimizer :
      return minimizer.profile_nps(pars)
    else :
      return pars

  def generate_data(self, pars) :
    return Data(self, np.random.poisson(self.tot_exp(pars)), [ par.generate_aux(pars[par.name]) for par in self.nps.values() ])

  def generate_asimov(self, pars) :
    return Data(self).set_data(self.tot_exp(pars), pars.nps)

  def generate_expected(self, pois, minimizer = None) :
    return self.generate_asimov(self.expected_pars(pois, minimizer))

  @staticmethod
  def create(filename) :
    return Model().load(filename)
  
  def load_jdict(self, jdict) :
    self.name = self.load_field('name', jdict, '', str)
    self.pois = {}
    if not 'model'    in jdict : raise KeyError("No 'model' section in specified JSON file")
    if not 'POIs'     in jdict['model'] : raise KeyError("No 'POIs' section in specified JSON file")
    if not 'NPs'      in jdict['model'] : raise KeyError("No 'NPs' section in specified JSON file")
    if not 'aux_obs'  in jdict['model'] : raise KeyError("No 'aux_obs' section in specified JSON file")
    if not 'channels' in jdict['model'] : raise KeyError("No 'channels' section in specified JSON file")
    for json_poi in jdict['model']['POIs'] :
      poi = ModelPOI()
      poi.load_jdict(json_poi)
      if poi.name in self.pois :
        raise ValueError('ERROR: multiple POIs defined with the same name (%s)' % poi.name)
      self.pois[poi.name] = poi
    self.aux_obs = {}
    for json_aux in jdict['model']['aux_obs'] :
      par = ModelAux()
      par.load_jdict(json_aux)
      if par.name in self.aux_obs :
        raise ValueError('ERROR: multiple auxiliary observables defined with the same name (%s)' % par.name)
      self.aux_obs[par.name] = par
    self.nps = {}
    for json_np in jdict['model']['NPs'] :
      par = ModelNP()
      par.load_jdict(json_np)
      if par.aux_obs != None and not par.aux_obs in self.aux_obs :
        raise ValueError('ERROR: auxiliary observable %s for NP %s was not defined.' % (par.aux_obs, par.name))
      if par.name in self.nps :
        raise ValueError('ERROR: multiple NPs defined with the same name (%s)' % par.name)
      self.nps[par.name] = par
    self.channels = {}
    for json_channel in jdict['model']['channels'] :
      channel = Channel()
      channel.load_jdict(json_channel)
      if channel.name in self.channels :
        raise ValueError('ERROR: multiple channels defined with the same name (%s)' % channel.name)
      self.channels[channel.name] = channel
    self.init_vars()
    return self
  
  def fill_jdict(self, jdict) :
    jdict['model'] = {}
    jdict['model']['name'] = self.name
    jdict['model']['channels'] = []
    for poi in self.pois    : jdict['model']['POIs']   .append(poi.dump_jdict())
    for aux in self.aux_obs : jdict['model']['aux_obs'].append(aux.dump_jdict())
    for par in self.nps     : jdict['model']['NPs']    .append(par.dump_jdict())
    for channel in self.channels : jdict['model']['channels'].append(channel.dump_jdict())
    
  def __str__(self) :
    s = 'POIs :\n'
    for poi in self.pois : s += str(poi) + '\n'
    s = 'NPs :\n'
    for par in self.nps : s += str(par) + '\n'
    s = 'Channels :\n'
    for channel in channels : s += str(channel) + '\n'
    return s

# -------------------------------------------------------------------------
class Parameters :
  def __init__(self, pois, nps = np.array([]), model = None) :
    if not isinstance(pois, np.ndarray) : pois = np.array([ pois])
    if not isinstance(nps , np.ndarray) : nps  = np.array([ nps ])
    if pois.ndim != 1 :
        raise ValueError('Input POI array should be a 1D vector, got ' + str(pois))
    if nps.ndim != 1 :
        raise ValueError('Input POI array should be a 1D vector, got ' + str(nps))
    if model :
      if nps.size == 0 : nps = np.zeros(model.nnps)
      if pois.size != model.npois : raise ValueError('Cannot initialize Parameters with %d POIs, when %d are defined in the model' % (pois.size, model.npois))
      if nps .size != model.nnps  : raise ValueError('Cannot initialize Parameters with %d NPs, when %d are defined in the model'  % (nps .size, model.nnps))
    self.pois = pois
    self.nps  = nps
    self.model = model

  def __str__(self) :
    s = ''
    if self.model == None :
      s += 'pois = ' + str(self.pois) + '\n'
      s += 'nps  = ' + str(self.nps)  + '\n'
    else :
      s += 'pois : ' + '\n        '.join( [ '%-12s = %8.4f' % (p.name,v) for p, v in zip(self.model.pois.values(), self.pois) ] ) + '\n'
      s += 'nps  : ' + '\n       ' .join( [ '%-12s = %8.4f (unscaled : %12.4f)' % (p.name,v, self.unscaled(p.name)) for p, v in zip(self.model.nps .values(), self.nps ) ] )
    return s

  def __contains__(self, par) :
    if self.model == None : raise ValueError('Cannot perform operation without a model.')
    return par in self.model.pois or par in self.model.nps

  def __getitem__(self, par):
    if self.model == None : raise ValueError('Cannot perform operation without a model.')
    if par in self.model.pois : return self.pois[list(self.model.pois).index(par)]
    if par in self.model.nps  : return self.nps [list(self.model.nps ).index(par)]
    raise KeyError('Model parameter %s not found' % par)

  def set(self, par, val, unscaled=False) :
    if self.model == None : raise ValueError('Cannot perform operation without a model.')
    if par in self.model.pois :
      self.pois[list(self.model.pois).index(par)] = val
      return self
    if par in self.model.nps :
      if unscaled :
        par_obj = self.model.nps[par]
        val = (val - par_obj.nominal_value)/par_obj.variation
      self.nps[list(self.model.nps ).index(par)] = val
      return self
    raise KeyError('Model parameter %s not found' % par)

  def __setitem__(self, par, val) :
    return self.set(par, val)

  def unscaled_nps(self) :
    if self.model == None : raise ValueError('Cannot perform operation without a model.')
    return self.model.np_nominal_values + self.nps*self.model.np_variations

  def unscaled(self, par) :
    if self.model == None : raise ValueError('Cannot perform operation without a model.')
    if par in self.model.nps :
      par_obj = self.model.nps[par]
      return par_obj.nominal_value + self.__getitem__(par)*par_obj.variation
    raise KeyError('Model nuisance parameter %s not found' % par)

  def dict(self, unscaled = True) :
    if self.model == None : raise ValueError('Cannot perform operation without a model.')
    dic = {}
    for poi, val in zip(self.model.pois.keys(), self.pois) : dic[poi] = val
    for par, val in zip(self.model.nps .keys(), self.unscaled_nps() if unscaled else self.nps) : dic[par] = val
    return dic

  def set_from_aux(self, data) :
    if self.model == None : raise ValueError('Cannot perform operation without a model.')
    self.nps[self.model.ncons:] = data.aux_obs[self.model.ncons:]
    return self


# -------------------------------------------------------------------------
class Data (JSONSerializable) :
  def __init__(self, model, counts = np.array([]), aux_obs = np.array([])) :
    super().__init__()
    self.model = model
    self.set_counts(counts)
    self.set_aux_obs(aux_obs)

  def set_counts(self, counts) :
    if isinstance(counts, list) : counts = np.array( counts )
    if not isinstance(counts, np.ndarray) : counts = np.array([ counts ])
    if counts.size > 0 :
      if counts.ndim != 1 :
        raise ValueError('Input data counts should be a 1D vector, got ' + str(counts))
      if counts.size != self.model.nbins :
        raise ValueError('Input data counts should have a size equal to the number of model bins (%d), got %d.' % (model.nbins, len(counts)))
      self.counts = counts
    else :
      self.counts = np.zeros(self.model.nbins)
    return self

  def set_aux_obs(self, aux_obs) : 
    if isinstance(aux_obs, list) : aux_obs = np.array( aux_obs )
    if not isinstance(aux_obs, np.ndarray) : aux_obs = np.array([ aux_obs ])
    if aux_obs.size == 0 :
      self.aux_obs = self.model.np_nominal_values
    else :
      if aux_obs.ndim != 1 :
        raise ValueError('Input aux data should be a 1D vector, got ' + str(aux_obs))
      if aux_obs.size == self.model.nnps :
        self.aux_obs = aux_obs
      elif aux_obs.size == self.model.ncons :
        self.aux_obs = np.concatenate((aux_obs, self.model.np_nominal_values[self.model.ncons:]))
      else :
        raise ValueError('Input aux data should have a size equal to the number of model auxiliary observables (%d), got %d.' % (self.model.ncons, str(aux_obs)))
    return self

  def set_data(self, counts, aux_obs) :
    self.set_counts(counts)
    self.set_aux_obs(aux_obs)
    return self

  def set_expected(self, pars) :
    self.set_data(self.model.tot_exp(pars), [ par.value for par in pars.constrained_nps() ])
    return self
  
  def load_jdict(self, jdict) :
    if not 'data'    in jdict : raise KeyError("No 'data' section in specified JSON file")
    if not 'channels'  in jdict['data'] : raise KeyError("No 'channels' section in specified JSON file")
    for channel in jdict['data']['channels'] :
      name = channel['name'] if 'name' in channel else ''
      if not name in self.model.channels : raise ValueError("Data channel '%s' in specified JSON file is not defined in the model." % name)
      model_channel = self.model.channels[name]
      if not 'bins'  in channel : raise KeyError("No 'counts' section defined for data channel '%s' in specified JSON file." % name)
      if len(channel['bins']) != model_channel.dim() :
        raise ValueError("Data channel '%s' in specified JSON file has %d bins, but the model channel has %d." % (len(channel['bins']), model_channel.dim()))
      offset = self.model.channel_offsets[name]
      for b, bin_data in enumerate(channel['bins']) :
        if bin_data['lo_edge'] != model_channel.bins[b]['lo_edge'] or bin_data['hi_edge'] != model_channel.bins[b]['hi_edge'] :
          raise ValueError("Bin %d in data channel '%s' spans [%g,%g], but the model bin spans [%g,%g]." % 
                           (bin_data['lo_edge'], bin_data['hi_edge'], model_channel.bins[b]['lo_edge'], model_channel.bins[b]['hi_edge']))
        self.counts[offset + b] = bin_data['counts'] 
    if not 'aux_obs' in jdict['data'] : raise KeyError("No 'aux_obs' section defined in specified JSON file." % name)
    data_aux_obs = { aux_obs['name'] : aux_obs['value'] for aux_obs in jdict['data']['aux_obs'] }
    aux_obs_values = []
    for par in self.model.nps.values() :
      if par.aux_obs == None : 
        aux_obs_values.append(0)
        continue
      if not par.aux_obs in data_aux_obs : raise('Auxiliary observable %s defined in model, but not provided in the data' % par.aux_obs)
      aux_obs_values.append((data_aux_obs[par.aux_obs] - par.nominal_value)/par.variation)
    self.set_aux_obs(np.array(aux_obs_values))
    return self

  def fill_jdict(self, jdict) :
    jdict['data'] = {}
    jdict['data']['counts'] = self.counts.tolist()
    jdict['data']['aux_obs'] = self.aux_obs.tolist()

  def __str__(self) :
    s = ''
    s += 'counts  = ' + str(self.counts)  + '\n'
    s += 'aux_obs = ' + str(self.aux_obs) + '\n'
    return s
