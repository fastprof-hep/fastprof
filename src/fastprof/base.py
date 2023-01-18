"""Module containing the building blocks for fastprof models:

  * :class:`ModelPOI`, representing model *parameters of interest* (POIs).

  * :class:`ModelNP`, representing model *nuisance parameters* (NPs)

  * :class:`ModelAux`, representing an *auxiliary observable* (aux. obs.) of the model

  All the classes support loading from / saving to markup files. The basic mechanism for this I/O is implemented in the :class:`Serializable` base class from which they all derive. The JSON and YAML markup formats are supported.
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

  def load(self, filename : str, flavor : str = None) -> 'Serializable' :
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

  def save(self, filename, flavor : str = None, payload : dict = None) -> 'Serializable' :
    """Save the object to a markup file

      Args:
        filename: name of the file to save to
        flavor  : input markup flavor (currently supported: 'json' [default], 'yaml')
        payload : the data that should be saved

      Returns:
        Serializable: self
    """
    if flavor is None : flavor = self.guess_flavor(filename, 'json')
    sdict = self.dump_dict() if payload is None else payload
    with open(filename, 'w') as fd :
      if flavor == 'json' : return json.dump(sdict, fd, ensure_ascii=True, indent=3)
      if flavor == 'yaml' : return yaml.dump(sdict, fd, sort_keys=False, default_flow_style=None, width=10000)
      raise KeyError("Unknown markup flavor '%s',  so far only 'json' or 'yaml' are supported" % flavor)

  def guess_flavor(self, filename, default) -> str :
    """Save the object to a markup file

      Args:
        filename: name of the file to load from
        default : return value if the guessing is unsuccessful

      Returns:
        str: the markup flavor (currently 'json' or 'yaml')
    """
    if filename[-4:] == 'json' : return 'json'
    if filename[-4:] == 'yaml' : return 'yaml'
    return default
    
  def load_markup(self, data : str, flavor : str = 'json') -> 'Serializable' :
    """Load the object from a markup string

      Args:
        data  : markup data to load, in string format
        flavor: input markup flavor (currently supported: 'json' [default], 'yaml')

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

      Args:
        flavor: input markup flavor (currently supported: 'json' [default], 'yaml')

      Returns:
        str: the markup string encoding the object contents
    """
    sdict = self.dump_dict(sdict)
    if flavor == 'json' : return json.dumps(sdict, ensure_ascii=True, indent=3)
    if flavor == 'yaml' : return yaml.dump(sdict, sort_keys=False, default_flow_style=None, width=10000)
    raise KeyError("Unknown markup flavor '%s',  so far only 'json' or 'yaml' are supported" % flavor)

  def dump_dict(self) -> dict :
    """Dumps the object as a dictionary of markup data

      Returns:
        dict: dictionary with the object contents
    """
    sdict = {}
    self.fill_dict(sdict)
    return sdict

  @classmethod
  def load_field(cls, key : str, dic : dict, default = None, types : list = []) :
    """Load information from a dictionary of markup data, using a provided key

      If the key is not present, or if the value type is not among the
      ones listed in `types`, then `default` is returned instead.

      Args:
         key    : key to look up in the dictionary
         dic    : dictionary object in which to look up the key
         default: default value to return if `key` is absent in `dic`, or the return type is not listed (default: None)
         types  : list of strings giving allowed return types (default: [], allows all types)

      Returns:
        (depends): the value indexed by `key` in `dic`, or `default` if not present or not of the expected type.
    """
    if not key in dic : return default
    val = dic[key]
    if val is None : return val
    if not isinstance(types, list) : types = [ types ]
    if types == [ np.ndarray ] : types.append(list) 
    if types != [] and not any([isinstance(val, t) for t in types]) :
      raise TypeError('Object at key %s in markup dictionary has type %s, not the expected %s' %
                      (key, val.__class__.__name__, '|'.join([t.__name__ for t in types])))
    if types == [ np.ndarray, list ] : val = np.array(val, dtype=float)
    return val

  @staticmethod
  def unnumpy(obj : np.ndarray) -> list :
    """Process numpy data to make it serializable

      numpy objects such as np.ndarray cannot be serialized to markup
      as-is and need tobe converted (e.g. arrays to lists). This function
      performs this conversion recursively on arrays and dicts containing
      arrays.

      Args:
         obj : object to convert

      Returns:
        (depends): the same object in serializable form
    """
    if isinstance(obj, np.int64) : return int(obj)
    if isinstance(obj, np.float64) : return float(obj)
    if isinstance(obj, np.ndarray) or isinstance(obj, list) : 
      return [ Serializable.unnumpy(element) for element in obj ]
    if isinstance(obj, dict) :
      return { key : Serializable.unnumpy(value) for key, value in obj.items() }
    return obj
  
  @staticmethod
  def good_float(value : float) -> float :
    """Replace `None` values by a `NaN` to make it serializable

      Args:
         value : object to convert

      Returns:
        float: the same object in serializable form
    """
    return value if value is not None else float('nan')

  @abstractmethod
  def load_dict(self, sdict: dict) -> 'Serializable' :
    """Abstract method to load information from a dictionary of markup data

      Args:
        sdict: a dictionary containing markup data

      Returns:
        self
    """
    return self

  @abstractmethod
  def fill_dict(self, sdict) :
    """Abstract method to save information to a dictionary of markup data

      Args:
         sdict: a dictionary containing markup data
    """
    pass


# -------------------------------------------------------------------------
class ModelPOI(Serializable) :
  """Class representing a POI of the model

  Stores the information relevant for POIs (see attributes),
  implements markup loading/saving through the :class:`Serializable`
  base class.
  This class represents the POI specification and does not
  store the POI value, uncertainties, etc. 

  Attributes:
     name          (str)   : the name of the parameter
     min_value     (float) : the lower bound of the allowed range of the parameter
     max_value     (float) : the upper bound of the allowed range of the parameter
     initial_value (float) : the initial value of the parameter when performing fits to data
     unit          (str)   : the unit in which the parameter is expressed
  """

  def __init__(self, name : str = '', min_value : float = None, max_value : float = None,
               initial_value : float = None, unit : str = '') :
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

  def __str__(self) -> str :
    """Return a description string

      Returns:
        str: the object description
    """
    return 'POI ' + self.string_repr(verbosity = 1)

  def string_repr(self, verbosity : int = 1, pre_indent : str = '', indent : str = '   ') -> str :
    """Return a string representation of the object

      Same as __str__ but with options

      Args:
        verbosity : verbosity of the output
        pre_indent: number of indentation spaces to add to all lines
        indent    : number of indentation spaces to add to fields of this object

      Returns:
        the description string
    """
    unit = ' %s' % self.unit if self.unit is not None and self.unit != '' else ''
    rep = '%s%s' % (pre_indent,  self.name)
    if verbosity == 1 :
      rep += ' = %g%s (min = %g%s, max = %g%s)' % (self.initial_value, unit, self.min_value, unit, self.max_value, unit)
    elif verbosity >= 2 :
      rep += '\n%sinitial_value = %g%s' % (pre_indent + indent, self.initial_value, unit)
      rep += '\n%smin_value = %g%s' % (pre_indent + indent, self.min_value, unit)
      rep += '\n%smax_value = %g%s' % (pre_indent + indent, self.max_value, unit)
    return rep


  def load_dict(self, sdict) -> 'ModelPOI' :
    """load object information from a dictionary of markup data

      Args:
        sdict: a dictionary containing markup data

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
         sdict: a dictionary containing markup data
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
  Implements markup loading/saving through the :class:`Serializable`
  base class.
  This class represents the aux. obs specification and does not
  store the its value.

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
    """Return a description string
      Returns:
        the object description
    """
    return 'aux_obs ' + self.string_repr(verbosity = 1)

  def string_repr(self, verbosity : int = 1, pre_indent : str = '', indent : str = '   ') -> str :
    """Return a string representation of the object

      Same as __str__ but with options

      Args:
        verbosity : verbosity of the output
        pre_indent: number of indentation spaces to add to all lines
        indent    : number of indentation spaces to add to fields of this object

      Returns:
        the description string
    """
    unit = ' %s' % self.unit if self.unit is not None and self.unit != '' else ''
    s = '%s%s' % (pre_indent,  self.name)
    if verbosity == 1 :
      s += ' (min = %g%s, max = %g%s)' % (self.good_float(self.min_value), unit, self.good_float(self.max_value), unit)
    elif verbosity >= 2 :
      s += '\n%smin_value = %5g%s' % (pre_indent + indent, self.good_float(self.min_value), unit)
      s += '\n%smax_value = %5g%s' % (pre_indent + indent, self.good_float(self.max_value), unit)
    return s


  def load_dict(self, sdict) -> 'ModelAux' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: a dictionary containing markup data

      Returns:
        self
    """
    self.name = self.load_field('name', sdict, '', str)
    self.unit = self.load_field('unit', sdict, '', str)
    self.min_value = self.load_field('min_value', sdict, None, [int, float])
    self.max_value = self.load_field('max_value', sdict, None, [int, float])
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: a dictionary containing markup data
    """
    sdict['name'] = self.name
    sdict['unit'] = self.unit
    sdict['min_value'] = self.unnumpy(self.min_value)
    sdict['max_value'] = self.unnumpy(self.max_value)


# -------------------------------------------------------------------------
class ModelNP(Serializable) :
  """Class representing a NP of the model

  Stores the information relevant for NPs (see attributes),
  implements markup loading/saving through the :class:`Serializable`
  base class.
  This class represents the NP specification and does not
  store its value, uncertainties, etc. 

  NPs have two representations:

  * Their original representation in the full model, with a nominal value
    given by the `nominal_value` attribute and a variations given by the
    `variations` attribute

  * A *scaled* representation in the linearized model (see :class:`Model`),
    where their values represent pulls : the nominal is 0, and 1 represents
    a :math:`1\sigma` deviation from the nominal

  The scaled representation is obtained from the original as
  `scaled` = (`original` - `nominal_value`)/`variation`.

  Attributes:
    name           (str)   : the name of the parameter
    nominal_value  (float) : the reference value of the parameter used to compute nominal sample yields
    variation      (float) : the uncertainty on the parameter value
    constraint     (float) : the value of the width of the constraint Gaussian,
                             for constrained NPs (otherwise None). The value is
                             in *unscaled* units (same as `nominal_value` and `variation`)
    aux_obs        (:class:`ModelAux`) : pointer to the corresponding aux. obs. object,
                                         for constrained NPs.
    unit           (str)   : the unit in which the parameter is expressed
"""

  def __init__(self, name = '', nominal_value : float = 0, variation : float = 1,
               constraint : float = None, aux_obs : float = None, unit : str = None) :
    self.name = name
    self.unit = unit
    self.nominal_value = nominal_value
    self.variation = variation
    self.constraint = constraint
    self.aux_obs = aux_obs

  def is_free(self) -> bool :
    """Returns if an NP is free (unconstrained) or constrained by an aux. obs.

      Returns:
         bool: True for a free parameter, False for a constrained one
    """
    return self.constraint is None

  def unscaled_value(self, scaled_value : float) -> float :
    """Returns the unscaled value of a NP, for a given scaled value

      See the class description (:class:`ModelNP`) for an explanation of scaled vs. unscaled.

      Args:
        scaled_value: the scaled value of the NP
      Returns:
        the corresponding unscaled value
    """
    return self.nominal_value + scaled_value*self.variation

  def scaled_value(self, unscaled_value : float) -> float :
    """Returns the scaled value of a NP, for a given unscaled value

      See the class description (:class:`ModelNP`) for an explanation of scaled vs. unscaled.

      Args:
        unscaled_value: the unscaled value of the NP
      Returns:
        the corresponding scaled value
    """
    return (unscaled_value - self.nominal_value)/self.variation

  def scaled_constraint(self) -> float :
    """Returns the scaled value of the NP constraint

      See the class description (:class:`ModelNP`) for an explanation of scaled/unscaled.

      Returns:
        the scaled constraint value
    """
    return self.constraint/self.variation

  def generate_aux(self, value : float) -> float :
    """Randomly generate an aux. obs value for this NP

      Constrained NPs represent the mean of a Gaussian of width `constraint`.
      The function generates a random value in this distribution.

      Args:
        value : the unscaled value of the NP
      Returns:
        a random value for the aux. obs.
    """
    if self.constraint is None : return 0
    return np.random.normal(value, self.constraint)

  def __str__(self) -> str :
    """Return a description string

      Returns:
        The object description
    """
    return 'NP ' + self.string_repr(verbosity = 1)

  def string_repr(self, verbosity : int = 1, pre_indent : str = '', indent : str = '   ') -> str :
    """Return a string representation of the object

      Same as __str__ but with options

      Args:
        verbosity : verbosity of the output
        pre_indent: number of indentation spaces to add to all lines
        indent    : number of indentation spaces to add to fields of this object

      Returns:
        the description string
    """
    unit = ' %s' % self.unit if self.unit is not None and self.unit != '' else ''
    constraint = ' constrained to %s with σ = %g%s' % (self.aux_obs, self.constraint, unit) if self.aux_obs is not None else ' free parameter'
    s = '%s%s' % (pre_indent,  self.name)
    if verbosity == 1 :
      s += ' = %g +/- %g%s%s' % (self.nominal_value, self.variation, unit, constraint)
    elif verbosity >= 2 :
      s += '\n%snominal_value = %g%s' % (pre_indent + indent, self.nominal_value, self.unit)
      s += '\n%svariation = %g%s' % (pre_indent + indent, self.variation, self.unit)
      s += '\n%sconstraint = %s' % (pre_indent + indent, constraint)
    return s

  def load_dict(self, sdict) :
    """Load object information from a dictionary of markup data

      Args:
        sdict: a dictionary containing markup data

      Returns:
        self
    """
    self.name = self.load_field('name', sdict, '', str)
    self.unit = self.load_field('unit', sdict, '', str)
    self.nominal_value = self.load_field('nominal_value', sdict, 0, [int, float])
    self.variation = self.load_field('variation', sdict, None, [int, float])
    self.constraint = self.load_field('constraint', sdict, None)
    if self.variation is None :
      self.variation = self.constraint if self.constraint is not None else 1
    if self.constraint is not None :
      self.aux_obs = self.load_field('aux_obs', sdict, None, str)
    else :
      self.aux_obs = None
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: a dictionary containing markup data
    """
    sdict['name'] = self.name
    if self.unit != '' : sdict['unit'] = self.unit
    sdict['nominal_value'] = self.unnumpy(self.nominal_value)
    sdict['variation']     = self.unnumpy(self.variation)
    if self.constraint is not None : sdict['constraint']    = self.unnumpy(self.constraint)
    if self.aux_obs is not None : sdict['aux_obs'] = self.aux_obs

