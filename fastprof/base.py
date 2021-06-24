"""Module containing the building blocks for fastprof models:

  * :class:`ModelPOI`, representing model *parameters of interest* (POIs).

  * :class:`ModelNP`, representing model *nuisance parameters* (NPs)

  * :class:`ModelAux`, representing an *auxiliary observable* (aux. obs.) of the model

  * :class:`Channel`, defining a measurement region*

  * :class:`Sample` objects, each defining a contribution to a channel.

  All the classes support loading from / saving to markup files. The basic mechanism for this is implemented in the :class:`Serializable` base class from which they all derive.
"""

import numpy as np
import json, yaml
from abc import abstractmethod

# -------------------------------------------------------------------------
class Serializable :
  """An abstract base class for objects that load from / save to a markup filename

  The class implements

  * load() and save() methods with filename arguments,

  * load_json() and save_json() with markup object arguments.

  Both sets are implemented in terms of the abstract methods load_dict() and
  fill_dict(), which operate on dictionaries and should be implemented
  in the derived classes.
  """

  def __init__(self) :
    pass

  def load(self, filename : str, flavor : str = None) :
    """Load the object from a markup file

      Args:
        filename: name of the file to load from
        flavor  : input markup flavor (currently supported: 'json' [default], 'yaml')      
      Returns:
        Serializable: self
    """
    if flavor is None : flavor = self.guess_flavor(filename, 'json')
    with open(filename, 'r') as fd :
      if flavor == 'json' :
        sdict = json.load(fd)
      elif flavor == 'yaml' :
        sdict = yaml.safe_load(fd)
      else :
        raise KeyError("Unknown markup flavor '%s',  so far only 'json' or 'yaml' are supported" % flavor)
    return self.load_dict(sdict)

  def save(self, filename, flavor : str = None) :
    """Save the object to a markup file

      Args:
        filename: name of the file to load from
        flavor  : input markup flavor (currently supported: 'json' [default], 'yaml')      

      Returns:
        Serializable: self
    """
    if flavor is None : flavor = self.guess_flavor(filename, 'json')
    sdict = self.dump_dict()
    with open(filename, 'w') as fd :
      if flavor == 'json' : return json.dump(sdict, fd, ensure_ascii=True, indent=3)
      if flavor == 'yaml' : return yaml.dump(sdict, fd, sort_keys=False, default_flow_style=None, width=10000)
      raise KeyError("Unknown markup flavor '%s',  so far only 'json' or 'yaml' are supported" % flavor)

  def guess_flavor(self, filename, default) :
    """Save the object to a markup file

      Args:
        filename: name of the file to load from
        default : return value if guessing is unsuccessful      

      Returns:
        str: the markup flavor 
    """
    if filename[-4:] == 'json' : return 'json'
    if filename[-4:] == 'yaml' : return 'yaml'
    return default
    
  def load_markup(self, data : str, flavor : str = 'json') :
    """Load the object from a markup string

      Args:
        data: markup data to load, in string format

      Returns:
        Serializable: self
    """
    if flavor == 'json' :
      sdict = json.loads(data)
    elif flavor == 'yaml' :
      sdict = yaml.load(data)
    else :
      raise KeyError("Unknown markup flavor '%s',  so far only 'json' or 'yaml' are supported" % flavor)
    return self.load_dict(sdict)

  def dump_markup(self, flavor : str = 'json') -> str :
    """Dumps the object as a markup string

      Returns:
        str: the markup string encoding the object contents
    """
    sdict = self.dump_dict(sdict)
    if flavor == 'json' : return json.dumps(sdict, ensure_ascii=True, indent=3)
    if flavor == 'yaml' : return yaml.dump(sdict, sort_keys=False, default_flow_style=None, width=10000)
    raise KeyError("Unknown markup flavor '%s',  so far only 'json' or 'yaml' are supported" % flavor)

  def dump_dict(self) :
    """Dumps the object as a dictionary of markup data

      Returns:
        dict: dictionary with the object contents
    """
    sdict = {}
    self.fill_dict(sdict)
    return sdict

  def load_field(self, key : str, dic : dict, default = None, types : list = []) :
    """Load an field from a dictionary of markup data

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
    if not key in dic : return default
    val = dic[key]
    if val is None : return val
    if not isinstance(types, list) : types = [ types ]
    if types != [] and not any([isinstance(val, t) for t in types]) :
      raise TypeError('Object at key %s in markup dictionary has type %s, not the expected %s' %
                      (key, val.__class__.__name__, '|'.join([t.__name__ for t in types])))
    if types == [ list ] : val = np.array(val, dtype=float)
    return val

  def unnumpy(self, obj) :
    if isinstance(obj, np.int64) : return int(obj)
    if isinstance(obj, np.float64) : return float(obj)
    if isinstance(obj, np.ndarray) or isinstance(obj, list) : 
      return [ self.unnumpy(element) for element in obj ]
    if isinstance(obj, dict) :
      return { key : self.unnumpy(value) for key, value in obj.items() }
    return obj

  @abstractmethod
  def load_dict(self, sdict: dict) -> 'Serializable' :
    """Abstract method to load information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    return self

  @abstractmethod
  def fill_dict(self, sdict) :
    """Abstract method to save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data

      Returns:
         None
    """
    pass


# -------------------------------------------------------------------------
class ModelPOI(Serializable) :
  """Class representing a POI of the model

  Stores the information relevant for POIs (see attributes),
  implements markup loading/saving through Serializable
  base class.

  Attributes:
     name          (str)   : the name of the parameter
     min_value     (float) : the lower bound of the allowed range of the parameter
     max_value     (float) : the upper bound of the allowed range of the parameter
     initial_value (float) : the initial value of the parameter when performing fits to data
     unit          (str)   : the unit in which the parameter is expressed
  """

  def __init__(self, name : str = '', min_value : float = None, max_value : float = None,
               initial_value : float = None, unit : str = '', value : float = None, error : float = None) :
    """Initialize object attributes

      Missing arguments are set to None.

      Args:
        name          : the name of the parameter
        min_value     : the lower bound of the allowed range of the parameter
        max_value     : the upper bound of the allowed range of the parameter
        initial_value : the initial value of the parameter when performing fits to data
        unit          : the unit in which the parameter is expressed
    """
    self.name = name
    self.min_value = min_value
    self.max_value = max_value
    self.initial_value = initial_value
    self.unit = unit

  def __str__(self) :
    """Provides a description string

      Returns:
        The object description
    """
    s = "parameter '%s' :" % self.name
    s +=' (min = %g, max = %g)' % (self.min_value, self.max_value)
    if self.initial_value is not None : s += ' init = %g' % self.initial_value
    if self.unit is not None : s += ' %s' % self.unit
    return s

  def load_dict(self, sdict) :
    """load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        ModelPOI: self
    """
    self.name      = self.load_field('name'     , sdict,  self.name, str)
    self.min_value = self.load_field('min_value', sdict,  0, [int, float])
    self.max_value = self.load_field('max_value', sdict,  0, [int, float])
    self.initial_value = self.load_field('initial_value', sdict, 0, [int, float])
    self.unit      = self.load_field('unit'     , sdict,  '', str)
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    sdict['name']      = self.name
    if self.min_value is not None : sdict['min_value'] = self.unnumpy(self.min_value)
    if self.max_value is not None : sdict['max_value'] = self.unnumpy(self.max_value)
    if self.initial_value is not None : sdict['initial_value'] = self.unnumpy(self.initial_value)
    if self.unit != '' : sdict['unit']  = self.unit


# -------------------------------------------------------------------------
class ModelAux(Serializable) :
  """Class representing an auxiliary observable of the model

  Stores the information relevant for the auxiliary observables
  that provide a constraint on some model NPs (see attributes).
  Implements markup loading/saving through Serializable
  base class.

  Attributes:
    name          (str)   : the name of the aux. obs.
    min_value     (float) : the lower bound of the allowed range of the parameter
    max_value     (float) : the upper bound of the allowed range of the parameter
    unit          (str)   : the unit in which the parameter is expressed
 """

  def __init__(self, name = '', min_value : float = None, max_value : float = None, unit : str = None) :
    self.name = name
    self.unit = unit
    self.min_value = min_value
    self.max_value = max_value

  def __str__(self) -> str :
    """Provides a description string
      Returns:
        The object description
    """
    s = "auxiliary observable '%s'" % self.name
    if self.min_value is not None and self.max_value is not None :
      s +=' (min = %g, max = %g)' % (self.min_value, self.max_value)
    return s

  def load_dict(self, sdict) :
    """load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        ModelAux: self
    """
    self.name = self.load_field('name', sdict, '', str)
    self.unit = self.load_field('unit', sdict, '', str)
    self.min_value = self.load_field('min_value', sdict, None, [int, float])
    self.max_value = self.load_field('max_value', sdict, None, [int, float])
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    sdict['name'] = self.name
    sdict['unit'] = self.unit
    sdict['min_value'] = self.unnumpy(self.min_value)
    sdict['max_value'] = self.unnumpy(self.max_value)


# -------------------------------------------------------------------------
class ModelNP(Serializable) :
  """Class representing a NP of the model

  Stores the information relevant for NPs (see attributes),
  implements markup loading/saving through Serializable
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
                             in "unscaled" units (same as `nominal_value` and `variation`)
                             for constrained NPs (otherwise None).
    aux_obs        (:class:`ModelAux`) : pointer to the corresponding aux. obs. object,
                                         for constrained NPs.
    unit           (str)   : the unit in which the parameter is expressed
"""

  def __init__(self, name = '', nominal_value = 0, variation = 1, constraint = None, aux_obs = None, unit : str = None) :
    self.name = name
    self.unit = unit
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

  def scaled_constraint(self) -> float :
    """computes the scaled value of the NP constraint

      See the class description (:class:`ModelNP`) for an explanation of scaled/unscaled.

      Returns:
        The scaled constraint value
    """
    return self.constraint/self.variation

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

  def load_dict(self, sdict) :
    """load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        ModelNP: self
    """
    self.name = self.load_field('name', sdict, '', str)
    self.unit = self.load_field('unit', sdict, '', str)
    self.nominal_value = self.load_field('nominal_value', sdict, 0, [int, float])
    self.variation = self.load_field('variation', sdict, 1, [int, float])
    self.constraint = self.load_field('constraint', sdict, None)
    if self.constraint is not None :
      if self.variation is None : self.variation = float(self.constraint)
      self.aux_obs = self.load_field('aux_obs', sdict, None, str)
    else :
      self.aux_obs = None
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    sdict['name'] = self.name
    if self.unit != '' : sdict['unit'] = self.unit
    sdict['nominal_value'] = self.unnumpy(self.nominal_value)
    sdict['variation']     = self.unnumpy(self.variation)
    if self.constraint is not None : sdict['constraint']    = self.unnumpy(self.constraint)
    if self.aux_obs is not None : sdict['aux_obs'] = self.aux_obs

