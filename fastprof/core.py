"""Module containing the core fastprof code

The code is organized around the following classes:

  * :class:`Model` : the class implementing the likelihood model. This has the HistFactory structure, as follows
    
    * The model has a number of parameters:
    
       * *Parameters of interest* (POIs), implemented as :class:`ModelPOI` objects
       
       * *Nuisance parameters* (NPs), implemented as :class:`ModelNP` objects. 
    
    * The model is split into a number of :class:`Channel` objects, which each define
    
       * A number of measurement bins
       
       * A number of :class:`Sample` objects, each defining a contribution to the expected bin yields.
       
    
    * The :class:`Sample` objects store their nominal bin yields, an overall normalization and variations as a function of the NPs

  * :class:`Parameters` : a class storing a set of values for model POIs and NPs.
  
  * :class:`Data` : a class storing the observed data: the observed bin yields for each channel and the auxiliary observables for each NP. 
  
  All the classes support loading from / saving to JSON files. The basic mechanism for this is implemented in the :class:`JSONSerializable` base class from which they all derive.
"""

import numpy as np
import math
from abc import abstractmethod
import json
import matplotlib.pyplot as plt
import copy

# -------------------------------------------------------------------------
class JSONSerializable :
  """An abstract base class for objects that load from / save to a JSON filename
  
  The class implements
  
  * load() and save() methods with filename arguments,
  
  * load_json() and save_json() with JSON object arguments.
  
  Both sets are implemented in terms of the abstract methods load_jdict() and
  save_jdict(), which operate on dictionaries and should be implemented
  in the derived classes.
  """

  def __init__(self) :
    pass

  def load(self, filename : str) :
    """Load the object from a JSON file
      
      Args:
        filename: name of the file to load from
      
      Returns:
        JSONSerializable: self
    """
    with open(filename, 'r') as fd :
      jdict = json.load(fd)
      return self.load_jdict(jdict)

  def save(self, filename) :
    """Save the object to a JSON file
      
      Args:
        filename: name of the file to load from      

      Returns:
        JSONSerializable: self
    """
    with open(filename, 'w') as fd :
      jdict = self.dump_jdict()
      return json.dump(jdict, fd, ensure_ascii=True, indent=3)

  def load_json(self, js : str) :
    """Load the object from a JSON string
      
      Args:
        js: JSON data to load, in string format
        
      Returns:
        JSONSerializable: self
    """
    jdict = json.loads(js)
    return self.load_jdict(jdict)

  def dump_json(self) -> str :
    """Dumps the object as a JSON string
        
      Returns:
        str: the JSON string encoding the object contents
    """
    jdict = self.dump_jdict(jdict)
    return json.dumps(jdict)

  def dump_jdict(self) :
    """Dumps the object as a dictionary of JSON data
        
      Returns:
        dict: dictionary with the object contents
    """
    jdict = {}
    self.fill_jdict(jdict)
    return jdict

  def load_field(self, key : str, dic : dict, default = None, types : list = []) :
    """Load an field from a dictionary of JSON data
      
      If the key is not present, or if the value type is not among the 
      ones listed in `types`, then `default` is returned instead.
      
      Args:
         key    : key to look up in the dictionary
         dict   : dictionary object in which to look up the key
         default: default value to return if `key` is absent in `dic`, or the return type is not listed (default: None)
         types  : list of strings giving allowed return types (default: [], allows all types)
        
      Returns:
        (depends): the value indexed by `key` in `dic`, or `default` if not present or not of the expected type.
    """
    if not key in dic :
      if default != None : return default
      raise KeyError('Key %s not found in JSON dictionary' % key)
    val = dic[key]
    if not isinstance(types, list) : types = [ types ]
    if types != [] and not any([isinstance(val, t) for t in types]) :
      raise TypeError('Object at key %s in JSON dictionary has type %s, not the expected %s' % 
                      (key, val.__class__.__name__, '|'.join([t.__name__ for t in types])))
    if types == [ list ] : val = np.array(val, dtype=float)
    return val

  @abstractmethod
  def load_jdict(self, jdict: dict) -> 'JSONSerializable' :
    """Abstract method to load information from a dictionary of JSON data
        
      Args:
        jdict: A dictionary containing JSON data

      Returns:
        self
    """    
    return self

  @abstractmethod
  def fill_jdict(self, jdict) :
    """Abstract method to save information to a dictionary of JSON data
        
      Args:
         jdict: A dictionary containing JSON data

      Returns:
         None
    """    
    pass


class ModelPOI(JSONSerializable) :
  """Class representing a POI of the model
  
  Stores the information relevant for POIs (see attributes),
  implements JSON loading/saving through JSONSerializable
  base class.
        
  Attributes:
     name          (str)   : the name of the parameter
     value         (float) : the value of the parameter (either a best-fit value or a fixed hypothesis value)
     error         (float) : the uncertainty on the parameter value
     min_value     (float) : the lower bound of the allowed range of the parameter
     max_value     (float) : the upper bound of the allowed range of the parameter
     initial_value (float) : the initial value of the parameter when performing fits to data
  """

  def __init__(self, name = '', value = None, error = None, min_value = None, max_value = None, initial_value = None) :
    """Initialize object attributes
      
      Missing arguments are set to None.
      
      Args:
        name          (str)   : the name of the parameter
        value         (float) : the value of the parameter (either a best-fit value or a fixed hypothesis value)
        error         (float) : the uncertainty on the parameter value
        min_value     (float) : the lower bound of the allowed range of the parameter
        max_value     (float) : the upper bound of the allowed range of the parameter
        initial_value (float) : the initial value of the parameter when performing fits to data
    """    
    self.name = name
    self.value = value
    self.error = error
    self.min_value = min_value
    self.max_value = max_value
    self.initial_value = initial_value

  def __str__(self) :
    """Provides a description string
      
      Returns:
        The object description
    """
    s = "parameter '%s' :" % self.name
    if self.value is not None : s += ' %g' % self.value
    if self.error is not None : s += ' +/- %g'  % self.error
    s +=' (min = %g, max = %g)' % (self.min_value, self.max_value)
    if self.initial_value is not None : s += ' init = %g' % self.initial_value
    return s

  def load_jdict(self, jdict) : 
    """load object information from a dictionary of JSON data
        
      Args:
        jdict: A dictionary containing JSON data

      Returns:
        ModelPOI: self
    """    
    self.name      = self.load_field('name'     , jdict,  self.name, str)
    self.value     = self.load_field('value'    , jdict,  0, [int, float])
    self.error     = self.load_field('error'    , jdict,  0, [int, float])
    self.min_value = self.load_field('min_value', jdict,  0, [int, float])
    self.max_value = self.load_field('max_value', jdict,  0, [int, float])
    self.initial_value = self.load_field('initial_value', jdict, 0, [int, float])
    return self

  def fill_jdict(self, jdict) :
    """Save information to a dictionary of JSON data
        
      Args:
         jdict: A dictionary containing JSON data
    """    
    jdict['name']      = self.name
    jdict['value']     = self.value
    jdict['error']     = self.error
    jdict['min_value'] = self.min_value
    jdict['max_value'] = self.max_value
    jdict['initial_value'] = self.max_value


class ModelAux(JSONSerializable) :
  """Class representing an auxiliary observable of the model
  
  Stores the information relevant for the auxiliary observables
  that provide a constraint on some model NPs (see attributes).
  Implements JSON loading/saving through JSONSerializable
  base class.
        
  Attributes:
     name          (str)   : the name of the aux. obs.
     min_value     (float) : the lower bound of the allowed range of the parameter
     max_value     (float) : the upper bound of the allowed range of the parameter
  """

  def __init__(self, name = '', min_value = None, max_value = None) :
    self.name = name
    self.min_value = min_value
    self.max_value = max_value

  def __str__(self) -> str :
    """Provides a description string
      Returns:
        The object description
    """
    s = "Auxiliary observable '%s' : min = %g, max = %g" % (self.name, self.min_value, self.max_value)
    return s

  def load_jdict(self, jdict) :
    """load object information from a dictionary of JSON data
        
      Args:
        jdict: A dictionary containing JSON data

      Returns:
        ModelAux: self
    """    
    self.name = self.load_field('name', jdict, '', str)
    self.min_value = self.load_field('min_value', jdict, '', [int, float])
    self.max_value = self.load_field('max_value', jdict, '', [int, float])
    return self

  def fill_jdict(self, jdict) :
    """Save information to a dictionary of JSON data
        
      Args:
         jdict: A dictionary containing JSON data
    """        
    jdict['name'] = self.name
    jdict['min_value'] = self.min_value
    jdict['max_value'] = self.max_value

  
class ModelNP(JSONSerializable) :
  """Class representing a NP of the model
  
  Stores the information relevant for NPs (see attributes),
  implements JSON loading/saving through JSONSerializable
  base class.
  
  The NPs have two representations:
  
  * Their original representation in the full model, with a nominal value
    given by the `nominal_value` attribute and a variations given by the
    `variations` attribute

  * A `scaled` representation In the linearized model (see :class:`Model`),
    where their values represent pulls : the nominal is 0, and 1 represents
    a :math:`1\sigma` deviation from the nominal
  
  The scaled representation is obtained from the original as
  `scaled` = (`original` - `nominal_value`)/`variation`.
  
  Attributes:
     name           (str)   : the name of the parameter
     nominal_value  (float) : the reference value of the parameter used to compute nominal sample yields
     variation      (float) : the uncertainty on the parameter value
     constraint     (float) : the value of the width of the constraint Gaussian,
                              for constrained NPs (otherwise None).
     aux_obs        (:class:`ModelAux`) : pointer to the corresponding aux. obs. object,
                                          for constrained NPs.
  """

  def __init__(self, name = '', nominal_value = 0, variation = 1, constraint = None, aux_obs = None) :
    self.name = name
    self.nominal_value = nominal_value
    self.variation = variation
    self.constraint = constraint
    self.aux_obs = aux_obs

  def is_free(self) :
    """specifies if an NP is free (unconstrained) or constrained by an aux.obs
        
      Returns:
         bool: True for a free parameter, False for a constrained one
    """    
    return self.constraint is None

  def unscaled_value(self, scaled_value) :
    """computes the unscaled value of a NP, for a given scaled value
        
      See the class description (:class:`ModelNP`) for an explanation of scaled/unscaled.
      
      Args:
        scaled_value (float): the scaled value of the NP
      Returns:
         float: the corresponding unscaled value
    """    
    return self.nominal_value + scaled_value*self.variation

  def scaled_value(self, unscaled_value) :
    """computes the scaled value of a NP, for a given unscaled value
        
      See the class description (:class:`ModelNP`) for an explanation of scaled/unscaled.
      
      Args:
        unscaled_value (float): the unscaled value of the NP
      Returns:
         float: the corresponding scaled value
    """    
    return (unscaled_value - self.nominal_value)/self.variation

  def generate_aux(self, value) :
    """randomly generate an aux. obs value for this NP
        
      Constrained NPs represent the mean of a Gaussian of width `constraint`
      
      Args:
        unscaled_value (float): the unscaled value of the NP
      Returns:
         float: the corresponding scaled value
    """    
    if self.constraint is None : return 0
    return np.random.normal(value, self.constraint)

  def __str__(self) -> str :
    """Provides a description string
      
      Returns:
        The object description
    """
    s = "Nuisance parameter '%s' : nominal = %g, variation = %g, " % (self.name, self.nominal_value, self.variation)
    s += 'constraint = %g' % self.constraint if self.constraint is not None else 'free parameter'
    return s

  def load_jdict(self, jdict) : 
    """load object information from a dictionary of JSON data
        
      Args:
        jdict: A dictionary containing JSON data

      Returns:
        ModelNP: self
    """    
    self.name = self.load_field('name', jdict, '', str)
    self.nominal_value = self.load_field('nominal_value', jdict, None, [int, float])
    self.variation = self.load_field('variation', jdict, None, [int, float])
    self.constraint = self.load_field('constraint', jdict)
    if self.constraint is not None :
      self.aux_obs = self.load_field('aux_obs', jdict, '', str)
    else :
      self.aux_obs = None
    return self

  def fill_jdict(self, jdict) :
    """Save information to a dictionary of JSON data
        
      Args:
         jdict: A dictionary containing JSON data
    """    
    jdict['name'] = self.name
    jdict['nominal_val'] = self.nominal_value
    jdict['variation'] = self.variation
    jdict['constraint'] = self.constraint
    jdict['aux_obs'] = self.aux_obs


class Sample(JSONSerializable) :
  """Class representing a model sample
  
  Provides the functionality for HistFactory sample structures,
  which represent a contribution from a given process to a
  :class:`Channel` with a given binning.
  
  It provides the following:
  
  * An overall normalization term, which is a function of the
    model POIs (see :class:`ModelPOI`)
  
  * An expected event yield contribution to each channel bin
  
  * The relative variation of the per-bin yields with each model NP
    (see :class:`ModelNP`)

  Attributes:
     name          (str) : the name of the sample
     norm_expr     (str) : a string defining the normalization term
                           as a function of the POIs
     nominal_norm  (float) : the nominal value of the normalization term
                             for which the expected yields correspond
     nominal_yields (:class:`np.ndarray`) :
          the nominal per-bin yields, provided as an 1D np.array with 
          a size equal to the number of channel bins
     impacts (dict) : the relative variations in the per-bin yields
          corresponding to :math:`\pm1\sigma` variations
          in the value of each NP; provided as a python dict
          with a key for each NP, and each value itself a dict with
          keys 'pos' and 'neg', associated with the :math:`+1\sigma`
          and  :math:`-1\sigma` relative variations respectively.
  """    

  def __init__(self, name : str = '', norm : str = '', nominal_norm : float = None, nominal_yields : np.ndarray = None, impacts : np.ndarray = None) :
    """Create a new Sample object
        
      Args:
         name: sample name
         norm: expression of the sample normalization factor, as a function of the POIs
         nominal_norm: value of the normalization factor for which the nominal yields are provided
         nominal_yields: nominal event yields for all channel bins (np. array of size `nbins`)
         impacts: relative change of the nominal event yields for each NP (see class description for format)         
    """    
    self.name = name
    self.norm_expr = norm
    self.nominal_norm = nominal_norm
    self.nominal_yields = nominal_yields
    self.impacts = impacts

  def impact(self, par : str, which : str = 'pos') -> np.array :
    """provides the relative variations of the per-bin event yields for a given NP
        
      Args:
         par   : name of the NP
         which : direction of the variation : 'pos' (default) for positive NP values, 'neg' for
                negative NP values
      Returns:
         np.array : per-bin relative variations
    """
    if not par in self.impacts : raise KeyError('No impact defined in sample %s for parameters %s.' % (self.name, par))
    try:
      imp = np.array([ imp[which] for imp in self.impacts[par] ], dtype=float)
      return imp if which == 'pos' else 1/(1+imp) - 1
    except Exception as inst:
      print('Impact computation failed for sample %s, parameter %s, impact %s' % (self.name, par, which))
      raise(inst)

  def sym_impact(self, par : str) -> np.array :
    """Provides the symmetrized relative variations of the per-bin event yields for a given NP
        
      Args:
         par : name of the NP
      Returns:
         symmetrized per-bin relative variations
    """
    try:
      return np.sqrt((1 + self.impact(par, 'pos'))*(1 + self.impact(par, 'neg'))) - 1
    except Exception as inst:
      print('Symmetric impact computation failed, returning the positive impacts instead')
      print(inst)
      return self.impact(par, 'pos')

  def norm(self, pars_dict : dict) -> float :
    """Computes the overall normalization factor
        
      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         normalization factor value
    """
    if self.norm_expr == '' : return 1
    try:
      return eval(self.norm_expr, pars_dict)/self.nominal_norm
    except Exception as inst:
      print("Error while evaluating the normalization '%s' of sample '%s'." % (self.norm_expr, self.name))
      raise(inst)

  def __str__(self) -> str :
    """Provides a description string
      
      Returns:
        The object description
    """
    s = "Sample '%s', norm = %s (nominal = %g)" % (self.name, self.norm_expr, self.nominal_norm)
    return s

  def load_jdict(self, jdict) -> 'Sample' : 
    """Load object information from a dictionary of JSON data
        
      Args:
        jdict: A dictionary containing JSON data

      Returns:
        self
    """    
    self.name = self.load_field('name', jdict, '', str)
    self.norm_expr = self.load_field('norm', jdict, '', str)
    self.nominal_norm = self.load_field('nominal_norm', jdict, None, [float, int])
    self.nominal_yields = self.load_field('nominal_yields', jdict, None, list)
    self.impacts = self.load_field('impacts', jdict, None, dict)
    return self

  def fill_jdict(self, jdict) :
    """Save information to a dictionary of JSON data
        
      Args:
         jdict: A dictionary containing JSON data
    """    
    jdict['name'] = self.name
    jdict['norm'] = self.norm_expr
    jdict['nominal_norm'] = self.nominal_norm
    jdict['nominal_yields'] = self.nominal_yields
    jdict['impacts'] = self.impacts


class Channel(JSONSerializable) :
  """Class representing a model channel
  
  Provides the functionality for HistFactory channel structures,
  representing a set of measurement bins. Two types of channels
  are currently implemented, differing in how the
  bin list is handled:
  
  * `count` : a channel with a single measurement bin
  
  * `binned_range` : a channel with multiple bins spanning
    a range of a continuous observable.
  
  They provide in both cases:
  
  * A list of bins, each one defined by a python dict with the
    bin information
  
    * `count` type: dict of the form { 'name' : <bin name> }
    
    * `binned_range` type: dict of the form 
      { 'lo_edge' : <float value>, 'hi_edge': <float value> }
  
  * A list of :class:`Sample` objects representing the processes
    contributing to the event yield in each bin.

  Attributes:
     name (str) : the name of the channel
     type (str) : the channel type (`count` or 'binned_range`, default: `count`)
     bins (list) : a list of python dict objects defining each bin, in the
        format given above
     samples (dict) : the channel samples, as a dict mapping the sample names
        to the sample objects (see :class:`Sample`).
  """    

  def __init__(self, name : str = '', chan_type : str = 'count', bins : list = []) :
    """Initializes the Channel class
        
      Args:
         name : channel name
         type : channel type: 'count' (default) or 'binned_range' (see class description)
         bins : list of bin defiinitions (depends on `type`, see class description)
    """
    self.name = name
    self.type = chan_type
    self.bins = bins
    self.samples = {}

  def nbins(self) -> int :
    """Returns the number of bins in the channel

      Returns:
        the number of bins
    """
    return len(self.bins)

  def sample(self, name : str) :
    return self.samples[name] if name in self.samples else None

  def __str__(self) -> str :
    """Provides a description string
      
      Returns:
        The object description
    """
    s = "Channel '%s', type = %s" % (self.name, self.type)
    for sample in self.samples.values() : s += '\n    o ' + str(sample)
    return s

  def load_jdict(self, jdict : dict) : 
    """Load object information from a dictionary of JSON data
        
      Args:
        jdict: A dictionary containing JSON data

      Returns:
        Channel: self
    """    
    self.name = jdict['name']
    self.type = jdict['type']
    if self.type == 'binned_range' :
      self.bins = jdict['bins']
      self.obs_name = self.load_field('obs_name', jdict, '', str)
      self.obs_unit = self.load_field('obs_unit', jdict, '', str)
      for json_sample in jdict['samples'] :
        sample = Sample()
        sample.load_jdict(json_sample)
        self.samples[sample.name] = sample
    return self

  def fill_jdict(self, jdict) :
    """Save information to a dictionary of JSON data
        
      Args:
         jdict: A dictionary containing JSON data
    """    
    jdict['name'] = self.name
    jdict['type'] = self.type
    if self.type == 'binned_range' :
      jdict['bins'] = self.bins
      jdict['obs_name'] = self.obs_name
      jdict['obs_unit'] = self.obs_unit
      jdict['samples'] = []
      for sample in self.samples : jdict['samples'].append(sample.dump_jdict())


# -------------------------------------------------------------------------
class Parameters :
  """Class representing a set of parameter values
  
  Stores one full set of model parameter values, both POI and NP.
  Only the numerical values are stored. The parameter names and properties
  can be accessed through the optional `model` attribute, if it set. However
  these are not required for the basic functionality.
  Storage is in 2 np.arrays, one for POIs and one for NPs.

  NPs are stored in their scaled form (see the description of :class:`ModelNP`),
  but the unscaled form can be used as well if `model` is set (as this requires
  knowledge of parameter properties not stored locally).

  Attributes:
     pois (np.array): the POI values
     nps (np.array): the NP values
     model (Model): pointer to a :class:`Model` object containing the full
       model information, including parameter names and properties.
  """    

  def __init__(self, pois : np.ndarray, nps : np.ndarray = None, model : 'Model' = None) :
    """Initialize a Parameters object from POI and NP values
        
      Args:
        pois  : float-array of POI values
        nps   : float-array of NP values
        model : optional pointer to a :class:`Model` object
    """    
    if model is not None and isinstance(pois, dict) :
      poi_array = np.array([ np.nan ]*model.npois)
      poi_list = list(model.pois)
      for poi, val in pois.items() : 
        if poi in poi_list : poi_array[poi_list.index(poi)] = val
      if np.isnan(poi_array).any() : raise ValueError('Input POI dictionary did not contain a valid numerival value for each POI : %s' % str(pois))
      pois = poi_array
    if isinstance(pois, float) or isinstance(pois, int) : pois = np.array([ pois], dtype=float)
    if not isinstance(pois, np.ndarray) or pois.ndim != 1 : raise ValueError('Input POIs should be a 1D np.array, got ' + str(pois))
    if model is not None and pois.size != model.npois : raise ValueError('Cannot initialize Parameters with %d POIs, when %d are defined in the model' % (pois.size, model.npois))
    self.pois = pois
    if nps is None : nps  = np.array([], dtype=float)
    if model is not None and nps.size == 0 : nps = np.zeros(model.nnps)
    if isinstance(nps , float) or isinstance(nps , int) : nps  = np.array([ nps ], dtype=float)
    if not isinstance(nps, np.ndarray) or nps.ndim != 1 : raise ValueError('Input NPs should be a 1D np.array, got ' + str(nps))
    if model is not None and nps.size != model.nnps : raise ValueError('Cannot initialize Parameters with %d NPs, when %d are defined in the model'  % (nps .size, model.nnps))
    self.nps  = nps
    self.model = model

  def clone(self) -> 'Parameters' :
    """Clone a Parameters object
        
      Performs a deep-copy operation at the required level: deep-copy
      the np.arrays, but shallow-copy the model pointer
      
      Returns:
        the new clone
    """    
    return Parameters(np.array(self.pois, dtype=float), np.array(self.nps, dtype=float), self.model)

  def __str__(self) -> str :
    """Provides a description string
      
      Returns:
        The object description
    """
    s = ''
    if self.model is None :
      s += 'POIs = ' + str(self.pois) + '\n'
      s += 'NPs  = ' + str(self.nps)  + '\n'
    else :
      s += 'POIs : ' + '\n        '.join( [ '%-12s = %8.4f' % (p.name,v) for p, v in zip(self.model.pois.values(), self.pois) ] ) + '\n'
      s += 'NPs  : ' + '\n       ' .join( [ '%-12s = %8.4f (unscaled : %12.4f)' % (p.name,v, self.unscaled(p.name)) for p, v in zip(self.model.nps .values(), self.nps ) ] )
    return s

  def __contains__(self, par : str) -> bool :
    """Tests if a parameter is present
        
      Args:
        par : name of a parameter (either POI or NP)

      Returns:
        True if a parameter of this name is present, False otherwise
    """    
    if self.model is None : raise ValueError('Cannot perform operation without a model.')
    return par in self.model.pois or par in self.model.nps

  def __getitem__(self, par : str) -> float :
    """Implement [] lookup of POI and NP names
        
      Args:
        par : name of a parameter (either POI or NP)

      Returns:
        The value of the parameter
    """    
    if self.model is None : raise ValueError('Cannot perform operation without a model.')
    if par in self.model.pois : return self.pois[list(self.model.pois).index(par)]
    if par in self.model.nps  : return self.nps [list(self.model.nps ).index(par)]
    raise KeyError('Model parameter %s not found' % par)

  def set(self, par : str, val : float, unscaled : bool = False) -> 'Parameters' :
    """Set the value of a parameter (POI or NP)
        
      Args:
        par : name of a parameter (either POI or NP)
        val : parameter value to set
        unscaled : for NPs, interpret `val` as a `scaled` (False) or `unscaled` (True) value,
                   see the description of :class:`ModelNP` for details
      Returns:
        self
    """    
    if self.model is None : raise ValueError('Cannot perform operation without a model.')
    if par in self.model.pois :
      self.pois[list(self.model.pois).index(par)] = val
      return self
    if par in self.model.nps :
      self.nps[list(self.model.nps).index(par)] = val if not unscaled else self.model.nps[par].scaled_value(val)
      return self
    raise KeyError('Model parameter %s not found' % par)

  def __setitem__(self, par : str, val : float) -> 'Parameters' :
    """Implement setting parameters using pars['parname'] = 3.14 syntax, for both POIs and NPs

      For NPs, `val` is considered to be a `scaled` value (use :meth:`Parameters.set` to
      set `unscaled` values.
        
      Args:
        par : name of a parameter (either POI or NP)
        val : parameter value to set
      Returns:
        self
    """    
    return self.set(par, val)

  def unscaled_nps(self) -> np.array :
    """Returns an np.array of the unscaled values of all NPs
        
      Returns:
        array of unscaled values of all NPs.
    """    
    if self.model is None : raise ValueError('Cannot perform operation without a model.')
    return self.model.np_nominal_values + self.nps*self.model.np_variations

  def unscaled(self, par : str) -> float :
    """Returns the unscaled value of an NP
        
      Args:
        par : an NP name

      Returns:
        the unscaled value of the NP
    """    
    if self.model is None : raise ValueError('Cannot perform operation without a model.')
    if par in self.model.nps : return self.model.nps[par].unscaled_value(self.__getitem__(par))
    raise KeyError('Model nuisance parameter %s not found' % par)

  def dict(self, nominal_nps : bool = False, unscaled_nps : bool = True, pois_only : bool = False) -> dict :
    """Returns a dictionary of parameter name : value pairs
        
      Args:
        nominal_nps : set NPs to their nominal values
        unscaled_nps : specifies whether to use the `unscaled` (True) or `scaled` (False)
                      value for NPs.
        pois_only : only include POIs

      Returns:
        Dictionary of parameter name : value pairs
    """    
    if self.model is None : raise ValueError('Cannot perform operation without a model.')
    if nominal_nps : return Parameters(self.pois, model=self.model).dict(nominal_nps=False, unscaled_nps=unscaled_nps)
    dic = {}
    for poi, val in zip(self.model.pois.keys(), self.pois) : dic[poi] = val
    if pois_only : return dic
    for par, val in zip(self.model.nps .keys(), self.unscaled_nps() if unscaled_nps else self.nps) : dic[par] = val
    return dic

  def set_from_dict(self, dic : dict, unscaled_nps : bool = False) -> 'Parameters' :
    """Set parameter values form a dictionary of parameter name : value pairs
        
      Args:
        dic : a dictionary containing parameter name : value pairs
        unscaled_nps : specifies whether NP values in `dic` should be
          considered as `unscaled` (True) or `scaled` (False).

      Returns:
        self
    """    
    for par, val in dic.items() : self.set(par, val, unscaled_nps)
    return self

  def set_from_aux(self, data : 'Data') -> 'Parameters' :
    """Set NP values to those of auxiliary observables
        
      Args:
        data : an observed dataset, from which aux. obs. values are taken

      Returns:
        self
    """    
    if self.model is None : raise ValueError('Cannot perform operation without a model.')
    self.nps[self.model.ncons:] = data.aux_obs[self.model.ncons:]
    return self


# -------------------------------------------------------------------------
class Model (JSONSerializable) :
  """Class representing the statistical model
  
  This class represents the top-level structure in the module, providing a
  description of the full statistical model. It is constructed from:
  
  * a list of :class:`Channel` objects, each describing
    a measurement in a separate region
  
  * lists of POIs (:class:`ModelPOI` objects) and NPs (:class:`ModelNP` objects).
  
  The main purpose of the class is to store the inputs to the fast liklelihood
  maximization algorithm of :class:`NPMinimizer`. For this purpose, the
  structures provided by the :class:`Channel` and :class:`Sample` classes are
  flattened into a number of np.array objects. These use a simplified description
  of measurement bins, in which the bins for all the channels are concatenated into
  a single large collection of size `nbins`. The `channel_offsets` attribute provides
  the indices of the first bin of each channel within this larger bin array.
  Other arrays store the expected event yields for each sample, and their variations
  as a function of the NPs.
  
  The model functionality is mainly accessed through the :meth:`Model.n_exp` method, which
  returns the expected event yield for a given set of parameter values `pars`, and the
  :meth:`Model.nll` method, which return the negative log-likelihood value.

  Attributes:
     pois (dict): the model POIs (as a dict mapping POI name to :class:`ModelPOI` object)
     nps  (dict): the model NPs (as a dict mapping NP name to :class:`ModelNP` object)
     aux_obs (dict): the model auxiliary observables that constrain the NPs
        (as a dict mapping aux. obs. name to :class:`ModelAux` object)
     npois (int): number of model pois
     nnps (int): number of model NPs
     ncons (int): number of constrained NPs (=number of aux. obs)
     nfree (int): number of free NPs (= nnps - ncons)
     channels (dict): the model channels (as a dict mapping channel name to :class:`Channel` object)
     samples (dict): the the model samples, compiled over all channels (as a dict mapping sample name
        to :class:`Sample` object)
     sample_indices (dict): maps sample name to its index in `samples` (which is an ordered dict)
     nbins (int): total number of measurement bins, compiled over channels
     channel_offsets (dict): maps channel name to the index of the first bin for this sample, among the list
       (of size `nbins`) concatenating the measurement bins of each channel.
     nominal_yields (np.array): expected event yields for each sample, as a 2D array of size 
       `nsamples` x `nbins`.
     pos_impacts (np.array): array of the per-sample event yield variations for positive NP values, as 
        a 3D array of size `nsamples` x `nbins` x `nnps`.
     neg_impacts (np.array): array of the per-sample event yield variations for negative NP values, as 
        a 3D array of size `nsamples` x `nbins` x `nnps`.
     sym_impacts (np.array): array of symmetrized per-sample event yield variations, as 
        a 3D array of size `nsamples` x `nbins` x `nnps`. The variations are computed as the average of
        the positive and negative impacts.
     log_pos_impacts (np.array): array of the logs of the per-sample event yield variations for positive NP values, as 
        a 3D array of size `nsamples` x `nbins` x `nnps`.
     log_neg_impacts (np.array): array of the logs of the per-sample event yield variations for negative NP values, as 
        a 3D array of size `nsamples` x `nbins` x `nnps`.
     log_sym_impacts (np.array): array of the logs of the symmetrized per-sample event yield variations, as 
        a 3D array of size `nsamples` x `nbins` x `nnps`.
     constraint_hessian (np.array): inverse of the covariance matrix of the NP constraint Gaussian
     np_nominal_values (np.array): nominal values of the unscaled NPs (see the description of :class:`ModelNP` for details)
     np_variations (np.array): variations of the unscaled NPs (see the description of :class:`ModelNP` for details)
     asym_impacts (bool): use asymmetric impact terms when computing yield variation (True, default), or
        use symmetrized impacts instead (False).
     linear_nps (bool): compute yield variations using an exponential form (False, default) or a linear
        form (True).
     lognormal_terms (bool): include the derivatives of the exponential terms in the NP minimization
        procedure (True) or not (False, default).
     cutoff (float): regularization term that caps the relative variations in event yields
  """    

  def __init__(self, pois : dict = {}, nps : dict = {}, aux_obs : dict = {}, channels : dict = {}, asym_impacts : bool = True, linear_nps : bool = False, lognormal_terms : bool = False) :
    """Initialize Model object
        
      Args:
        pois     : the model POIs, as a dict mapping POI names to :class:`ModelPOI` objects
        nps      : the model NPs, as a dict mapping NP names to :class:`ModelNP` objects
        aux_obs  : the model aux. obs., as a dict mapping names to :class:`ModelAux` objects
        channels : the model channelsm as a dict mapping channel names to :class:`Channel` objects
        asym_impacts : option to use symmetric or asymmetric NP impacts (see class description, default: True)
        linear_nps   : option to use the linear or exp form of NP impact on yields (see class description, default: False)
        lognormal_terms : option to include exp derivatives when minimizing nll (see class description, default: False)
    """        
    super().__init__()
    self.pois = { poi.name : poi for poi in pois }
    self.nps = {}
    for np in nps :
      if not np.is_free() : self.nps[par.name] = par
    ncons = len(self.nps)
    for np in nps :
      if np.is_free() : self.nps[par.name] = par
    self.aux_obs = { par.name : par for par in aux_obs }
    if len(self.aux_obs) != ncons :
      raise ValueError('Number of auxiliary observables (%d) does not match the number of constrained NPs (%d)' % (len(self.aux_obs), self.ncons))
    self.channels = { channel.name : channel for channel in channels }
    self.asym_impacts = asym_impacts
    self.linear_nps = linear_nps
    self.lognormal_terms = lognormal_terms
    self.cutoff = 0
    self.init_vars()

  def init_vars(self) :
    """Private method to initialize internal attributes
      
      The Model class contains both primary atttributes and secondary
      attributes that are pre-computed from the primary ones to speed
      up computations later. The primary->secondary computation is
      performed by this method, which is called from both 
      :meth:`Model.__init__` and :meth:`Model.load_jdict`.
    """    
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
      self.nbins += channel.nbins()
      for sample in channel.samples.values() :
        if not sample.name in self.sample_indices :
          self.samples[sample.name] = sample
          self.sample_indices[sample.name] = len(self.sample_indices)
    self.nominal_yields = np.zeros((len(self.sample_indices), self.nbins))
    self.pos_impacts = np.zeros((len(self.sample_indices), self.nbins, len(self.nps)))
    self.neg_impacts = np.zeros((len(self.sample_indices), self.nbins, len(self.nps)))
    self.sym_impacts = np.zeros((len(self.sample_indices), self.nbins, len(self.nps)))
    for channel in self.channels.values() :
      for sample in channel.samples.values() :
        self.nominal_yields[self.sample_indices[sample.name], self.channel_offsets[channel.name]:] = sample.nominal_yields
        for p, par in enumerate(self.nps.values()) :
          self.pos_impacts[self.sample_indices[sample.name], self.channel_offsets[channel.name]:, p] = sample.impact(par.name, 'pos')
          self.neg_impacts[self.sample_indices[sample.name], self.channel_offsets[channel.name]:, p] = sample.impact(par.name, 'neg')
          self.sym_impacts[self.sample_indices[sample.name], self.channel_offsets[channel.name]:, p] = sample.sym_impact(par.name)
    self.log_pos_impacts = np.log(1 + self.pos_impacts)
    self.log_neg_impacts = np.log(1 + self.neg_impacts)
    self.log_sym_impacts = np.log(1 + self.sym_impacts)
    self.constraint_hessian = np.zeros((self.nnps, self.nnps))
    self.np_nominal_values = np.array([ par.nominal_value for par in self.nps.values() ], dtype=float)
    self.np_variations     = np.array([ par.variation     for par in self.nps.values() ], dtype=float)
    for p, par in enumerate(self.nps.values()) :
      if par.constraint is None : break # we've reached the end of the constrained NPs in the NP list
      self.constraint_hessian[p,p] = 1/par.constraint**2

  def channel(self, name : str) -> Channel :
    """Returns a channel object by name
        
      Args:
         name : a channel name
      Returns:
         The channel object of that name
    """    
    return self.channels[name] if name in self.channels else None

  def all_pars(self) -> dict :
    """Returns all model parameters
        
      Returns:
         A dictionary of parameter name : object pairs containing
         all POIs and NPs.
    """    
    pars = {}
    for par in self.pois.values() : pars[par.name] = par
    for par in self.nps.values()  : pars[par.name] = par
    return pars

  def set_constraint(self, par : str, val : float) :
    """Set the value of the constraint on a NP
        
      If `par` is set to `None`, set the constraint on all NPs.
      See the documentation of :class:`ModelNP` for more details
      on constraints
        
      Args:
         par : a NP name
         val : a constraint value
    """    
    for par in self.nps :
      if par is None or par.name == par : par.constraint = val
    self.init_vars()

  def k_exp(self, pars : Parameters) -> np.array :
    """Returns the modifier to event yields due to NPs
      
      The expected event yield is modified by the NPs in a way
      that depends on the modeling options (see the documentation of
      :class:`Model` for details). This function returns a 2D 
      np.array with dimensions `nbins` x `nsamples`,
      where each value is the event yield modifier for each sample
      in each bin.
      
      Args:
         pars : a set of parameter values (only the NP values are used)
      Returns:
         Event yield modifiers
    """    
    if self.asym_impacts :
      if self.linear_nps :
        return 1 + self.pos_impacts.dot(np.maximum(pars.nps, 0)) + self.neg_impacts.dot(np.minimum(pars.nps, 0))
      else :
        return np.exp(self.log_pos_impacts.dot(np.maximum(pars.nps, 0)) + self.log_neg_impacts.dot(np.minimum(pars.nps, 0)))
    else :
      if self.linear_nps :
        return 1 + self.sym_impacts.dot(pars.nps)
      else :
        return np.exp(self.log_sym_impacts.dot(pars.nps))

  def n_exp(self, pars : Parameters) -> np.array :
    """Returns the expected event yields for a given parameter value
    
    The expected yields correspond to the nominal yields for each sample,
    corrected for the overall normalization terms (function of the POIs)
    and the NP impacts (function of the NPs, see :meth:`Model.k_exp`)
    They provided for each sample in each measurement bin, as a 2D 
    np.array with dimensions`nbins` x `nsamples`.
        
      Args:
         pars: a set of parameter values
      Returns:
         Expected event yields per sample per bin
    """    
    nnom = (self.nominal_yields.T*np.array([ sample.norm(pars.dict(nominal_nps=True)) for sample in self.samples.values() ], dtype=float)).T
    k = self.k_exp(pars)
    if self.cutoff == 0 : return nnom*k
    return nnom*(1 + self.cutoff*np.tanh((k-1)/self.cutoff))

  def tot_exp(self, pars, floor = None) -> np.array :
    """Returns the total expected event yields for a given parameter value
    
      Same as :meth:`Model.n_exp`, except that the yields are summed over
      all samples. They are provided as a 1D np.array of size `nbins`.
        
      Args:
         pars: a set of parameter values
      Returns:
         Expected event yields per bin
    """    
    ntot = self.n_exp(pars).sum(axis=0)
    return ntot if floor is None else np.maximum(ntot, floor)

  def nll(self, pars : Parameters, data : 'Data', offset : bool = True, floor : bool = None, no_constraints : bool = False) -> float :
    """Returns the negative log-likelihood value for a given parameter set and dataset
      
      If the `offset` argument is `True` (default), the nll is computed relatively
      to the case where all yields are nominal. This leads to smaller nll values,
      which reduces potential floating-point issues. When computing the difference
      of two nll values as in a profile-likelihood ratio computation, the offset
      cancels out in the difference.
      
      Args:
         pars   : a set of parameter values
         data   : an observed dataset
         offset : if True, use offsetting to reduce floating-point precision issues
         floor  : if a positive number is provided, will check for negative event yields
                  and replace them with the floor value
         no_constraints : omit the penalty  terms from the constraint in the computation.
      Returns:
         The negative log-likelihood value
    """    
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

  def plot(self, pars : Parameters, data : 'Data' = None, channel : Channel = None, exclude : list = None, 
           variations : list = None, residuals : bool = False, canvas : plt.Figure = None) :
    """Plot the expected event yields and optionally data as well
      
      The plot is performed for a single model, which must be of `binned_range` type.
      The event yields are plotted as a histogram, as a function of the channel
      observable.
      The `variations` arg allows to plot yield variations for selected NP values. The
      format is { ('par1', val1), ... } , which will plot the yields for the case where
      NP par1 is set to val1 (while other NPs remain at nominal), etc.
      
      Args:
         pars       : parameter values for which to compute the expected yields
         data       : observed dataset to plot alongside the expected yields
         channel    : name of the channel to plot. If `None`, plot the first channel.
         exclude    : list of sample names to exclude from the plot
         variations : list of NP variations to plot, as a list of (str, float) pairs
                      providing the NP name and the value to set.
         residuals  : if True, also show an inset plot with the data-model difference
         canvas     : a matplotlib Figure on which to plot (if None, plt.gca() is used)
    """    
    if canvas is None : canvas = plt.gca()
    if not isinstance(exclude, list) and exclude is not None : exclude = [ exclude ]
    if channel is None :
      for chan in self.channels.values() :
        if chan.type == 'binned_range' :
          channel = chan
          break
      if channel is None : raise ValueError('ERROR: Model does not contain a channel of binned_range type.')
    else :
      if not channel in self.channels : raise KeyError('ERROR: Channel %s is not defined.' % channel)
      channel = self.channels[channel]
    if len(channel.bins) > 0 :
      grid = [ b['lo_edge'] for b in channel.bins ]
      grid.append(channel.bins[-1]['hi_edge'])
    else :
      grid = np.linspace(0, channel.nbins, channel.nbins)
    xvals = [ (grid[i] + grid[i+1])/2 for i in range(0, len(grid) - 1) ]
    offset = self.channel_offsets[channel.name]
    nexp = self.n_exp(pars)[:,offset:offset + channel.nbins()]
    if exclude is None :
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
      canvas.errorbar(xvals, yvals, xerr=[0]*channel.nbins(), yerr=yerrs, fmt='ko', label='Data')
    canvas.set_xlim(grid[0], grid[-1])
    if variations is not None :
      for v in variations :
        vpars = pars.clone()
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

  def grad_poi(self, pars : Parameters, data : 'Data') -> np.ndarray :
    """Returns the derivatives of the negative log-likelihood wrt the POIs
        
      Output format: 1D np.ndarray of size `npois`.
      TODO : update to the new POI scheme, code below is obsolete
      
      Args:
         pars : parameter values at which to compute the likelihood
         data : observed dataset for which to compute the likelihood
      Returns:
         Values of the derivatives of the negative log-likelihood wrt the POIs.
    """    
    sexp = self.s_exp(pars) # This is for a "mu" POI, i.e. d(S_i)/dmu = S_i^exp (true for S_i = mu S_i^exp).
    nexp = sexp + self.b_exp(pars) # This is for a "mu" POI, i.e. d(S_i)/dmu = S_i^exp (true for S_i = mu S_i^exp).
    s = 0
    for i in range(0, sexp.size) : s += sexp[i]*(1 - data.n[i]/nexp[i])
    return s
  
  def hess_poi(self, pars : Parameters, data : 'Data') -> np.ndarray :
    """Returns the Hessian matrix of the negative log-likelihood wrt the POIs
        
      Output format: 2D np.ndarray of size `npois` x `npois`.
      TODO : update to the new POI scheme, code below is obsolete

      Args:
         pars : parameter values at which to compute the likelihood
         data : observed dataset for which to compute the likelihood
      Returns:
         Hessian matrix of the negative log-likelihood wrt the POIs.
    """    
    sexp = self.s_exp(pars) # This is for a "mu" POI, i.e. d(S_i)/dmu = S_i^exp (true for S_i = mu S_i^exp).
    nexp = sexp + self.b_exp(pars) # This is for a "mu" POI, i.e. d(S_i)/dmu = S_i^exp (true for S_i = mu S_i^exp).
    s = 0
    for i in range(0, sexp.size) : s += sexp[i]*data.n[i]/nexp[i]**2
    return s

  def expected_pars(self, pois : dict, minimizer : 'NPMinimizer' = None, data : 'Data' = None) -> 'Parameters' :
    """Assigns NP values to a set of POI values
    
      By default, returns a :class:`Parameters` object with the POI values
      defined by the `pois` arg, and the NPs set to 0. If a minimizer and 
      a dataset are provided, will set the NPs to their profiled values.
      The `pois` arg can also be a class:`Parameters` object, from which
      the POI values will be taken (and the NP values ignored).
        
      Args:
         pois : A dict of POI name : value pairs, or a class:`Parameters` object.
         minimizer (optional) : NP minimizer algorithm used to compute NP profile values
         data (optional) : dataset to use for the profiling
      Returns:
         Object containing the POI values and associated NP values.
    """
    if isinstance(pois, Parameters) :
      pars = pois
    else :
      pars = Parameters(pois, model=self)
    if minimizer and data :
      return minimizer.profile_nps(pars, data)
    else :
      return pars

  def generate_data(self, pars : Parameters) -> 'Data' :
    """Generate a pseudo-dataset for given parameter values
    
      Returns a randomly-generated dataset for the provided
      parameter values. This includes observed bin contents
      for all channels, generated from Poisson distributions,
      and aux. obs. values generated from the NP constraints.
    
      Args:
         pars : a set of model parameter values
      Returns:
         A randomly-generated dataset
    """    
    return Data(self, np.random.poisson(self.tot_exp(pars)), [ par.generate_aux(pars[par.name]) for par in self.nps.values() ])

  def generate_asimov(self, pars : Parameters) -> 'Data' :
    """Generate an Asimov dataset for given parameter values
    
      Returns an Asimov dataset for the provided parameter
      values, i.e. a dataset in which the obseved bin counts exactly
      match the expected yields, and the aux. obs. match the 
      NP values
      See `arXiv:1007.1727 <https://arxiv.org/abs/1007.1727>`_
    
      Args:
         pars : a set of model parameter values
      Returns:
         An Asimov dataset
    """    
    return Data(self).set_data(self.tot_exp(pars), pars.nps)

  def generate_expected(self, pois, minimizer = None, data = None) :
    """Generate an Asimov dataset for expected parameter values
    
      Same functionality as :meth:`Model.generate_asimov`, but with
      NP values that are obtained from the provided POI values in the
      same way as described for :meth:`Model.expected_pars`.
    
      Args:
         pois : A dict of POI name : value pairs, or a class:`Parameters` object.
         minimizer (optional) : NP minimizer algorithm used to compute NP profile values
         data (optional) : dataset to use for the profiling
      Returns:
         An Asimov dataset
    """    
    return self.generate_asimov(self.expected_pars(pois, minimizer, data))

  @staticmethod
  def create(filename : str) -> 'Model' :
    """Shortcut method to instantiate a model from a JSON file
    
      Same behavior as creating a default model and loading from the file,
      rolled into a single command
        
      Args:
         filename : name of a JSON file containing the model definition
      Returns:
         The created model
    """    
    return Model().load(filename)
  
  def load_jdict(self, jdict : dict) -> 'Model' :
    """Load object information from a dictionary of JSON data
        
      Args:
        jdict: A dictionary containing JSON data

      Returns:
        self
    """    
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
      if par.aux_obs is not None and not par.aux_obs in self.aux_obs :
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
  
  def fill_jdict(self, jdict : dict) :
    """Save information to a dictionary of JSON data
        
      Args:
         jdict: A dictionary containing JSON data
    """    
    jdict['model'] = {}
    jdict['model']['name'] = self.name
    jdict['model']['channels'] = []
    for poi in self.pois    : jdict['model']['POIs']   .append(poi.dump_jdict())
    for aux in self.aux_obs : jdict['model']['aux_obs'].append(aux.dump_jdict())
    for par in self.nps     : jdict['model']['NPs']    .append(par.dump_jdict())
    for channel in self.channels : jdict['model']['channels'].append(channel.dump_jdict())
    
  def __str__(self) -> str :
    """Provides a description string
      
      Returns:
        The object description
    """
    s = 'POIs :'
    for poi in self.pois.values() : s += '\n  - %s' % str(poi)
    s += '\nNPs :'
    for par in self.nps.values() : s += '\n  - %s' % str(par)
    s += '\nChannels :'
    for channel in self.channels.values() : s += '\n  - %s' % str(channel)
    return s


# -------------------------------------------------------------------------
class Data (JSONSerializable) :
  """Class representing a set of observed data
  
  Stores a complete dataset:
  
  * A list of observed bin yields
  
  * Observed values for the auxiliary observables.
  
  Both are stored as a single np.array apiece. The bins yields use the concatenated bin
  list defined in :class:`Model`, while the aux. obs values are given in the same order
  as their appearance in the `aux_obs` attribute of the model.
  
  Only numerical values are stored locally, but the names and properties 
  of the bins and aux. obs. are accessible through the `model` attribute, if it set.

  Attributes:
     counts (np.ndarray): the observed bin yields
     aux_obs (np.ndarray): the observed aux. obs. values
     model (Model): pointer to a :class:`Model` object containing the full model.
  """    
  def __init__(self, model : Model, counts : np.ndarray = None, aux_obs : np.ndarray = None) :
    """Initialize the object
        
      Args:
         counts  : observed bin counts
         aux_obs : aux. obs. values 
    """    
    super().__init__()
    self.model = model
    self.set_counts(counts if counts is not None else [])
    self.set_aux_obs(aux_obs if aux_obs is not None else [])

  def set_counts(self, counts) -> 'Data' :
    """Sets the observed bin counts to the specified values
        
      Args:
         counts  : observed bin counts to set to
      Returns:
         self
    """    
    if isinstance(counts, list) : counts = np.array( counts, dtype=float )
    if not isinstance(counts, np.ndarray) : counts = np.array([ counts ], dtype=float)
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
    """Sets the aux. obs. to the specified values
        
      Args:
         aux_obs : aux. obs. values to set to
      Returns:
         self
    """    
    if isinstance(aux_obs, list) : aux_obs = np.array( aux_obs, dtype=float )
    if not isinstance(aux_obs, np.ndarray) : aux_obs = np.array([ aux_obs ], dtype=float)
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
    """Sets both the observed bin counts and aux. obs. to the specified values
        
      Args:
         counts  : observed bin counts to set to
         aux_obs : aux. obs. values to set to
      Returns:
         self
    """    
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
      if len(channel['bins']) != model_channel.nbins() :
        raise ValueError("Data channel '%s' in specified JSON file has %d bins, but the model channel has %d." % (channel['name'], len(channel['bins']), model_channel.nbins()))
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
      if par.aux_obs is None : 
        aux_obs_values.append(0)
        continue
      if not par.aux_obs in data_aux_obs : raise('Auxiliary observable %s defined in model, but not provided in the data' % par.aux_obs)
      aux_obs_values.append((data_aux_obs[par.aux_obs] - par.nominal_value)/par.variation)
    self.set_aux_obs(np.array(aux_obs_values, dtype=float))
    return self

  def fill_jdict(self, jdict : dict) :
    """Save information to a dictionary of JSON data
        
      Args:
         jdict: A dictionary containing JSON data
    """    
    jdict['data'] = {}
    jdict['data']['counts'] = self.counts.tolist()
    jdict['data']['aux_obs'] = self.aux_obs.tolist()

  def __str__(self) -> str :
    """Provides a description string
      
      Returns:
        The object description
    """
    s = ''
    s += 'counts  = ' + str(self.counts)  + '\n'
    s += 'aux_obs = ' + str(self.aux_obs) + '\n'
    return s
