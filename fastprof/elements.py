"""Module containing the building blocks for fastprof models:

  * :class:`ModelPOI`, representing model *parameters of interest* (POIs).
       
  * :class:`ModelNP`, representing model *nuisance parameters* (NPs)
    
  * :class:`ModelAux`, representing an *auxiliary observable* (aux. obs.) of the model

  * :class:`Channel`, defining a measurement region*

  * :class:`Sample` objects, each defining a contribution to a channel.
  
  All the classes support loading from / saving to JSON files. The basic mechanism for this is implemented in the :class:`JSONSerializable` base class from which they all derive.
"""

import numpy as np
import json
from abc import abstractmethod

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
