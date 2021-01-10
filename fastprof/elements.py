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
  fill_jdict(), which operate on dictionaries and should be implemented
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
    if not key in dic : return default
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


# -------------------------------------------------------------------------
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
     unit          (str)   : the unit in which the parameter is expressed
  """

  def __init__(self, name : str = '', value : float = None, error : float = None, min_value : float = None, max_value : float = None,
               initial_value : float = None, unit : str = None) :
    """Initialize object attributes

      Missing arguments are set to None.

      Args:
        name          : the name of the parameter
        value         : the value of the parameter (either a best-fit value or a fixed hypothesis value)
        error         : the uncertainty on the parameter value
        min_value     : the lower bound of the allowed range of the parameter
        max_value     : the upper bound of the allowed range of the parameter
        initial_value : the initial value of the parameter when performing fits to data
        unit          : the unit in which the parameter is expressed
    """
    self.name = name
    self.unit = unit
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
    self.unit      = self.load_field('unit'     , jdict,  '', str)
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
    jdict['unit']      = self.unit
    jdict['value']     = self.value
    jdict['error']     = self.error
    jdict['min_value'] = self.min_value
    jdict['max_value'] = self.max_value
    jdict['initial_value'] = self.max_value


# -------------------------------------------------------------------------
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

  def load_jdict(self, jdict) :
    """load object information from a dictionary of JSON data

      Args:
        jdict: A dictionary containing JSON data

      Returns:
        ModelAux: self
    """
    self.name = self.load_field('name', jdict, '', str)
    self.unit = self.load_field('unit', jdict, '', str)
    self.min_value = self.load_field('min_value', jdict, None, [int, float])
    self.max_value = self.load_field('max_value', jdict, None, [int, float])
    return self

  def fill_jdict(self, jdict) :
    """Save information to a dictionary of JSON data

      Args:
         jdict: A dictionary containing JSON data
    """
    jdict['name'] = self.name
    jdict['unit'] = self.unit
    jdict['min_value'] = self.min_value
    jdict['max_value'] = self.max_value


# -------------------------------------------------------------------------
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

  def load_jdict(self, jdict) :
    """load object information from a dictionary of JSON data

      Args:
        jdict: A dictionary containing JSON data

      Returns:
        ModelNP: self
    """
    self.name = self.load_field('name', jdict, '', str)
    self.unit = self.load_field('unit', jdict, '', str)
    self.nominal_value = self.load_field('nominal_value', jdict, None, [int, float])
    self.variation = self.load_field('variation', jdict, None, [int, float])
    self.constraint = self.load_field('constraint', jdict, None)
    if self.constraint is not None :
      if self.variation is None : self.variation = float(self.constraint)
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
    jdict['unit'] = self.unit
    jdict['nominal_value'] = self.nominal_value
    jdict['variation'] = self.variation
    jdict['constraint'] = self.constraint
    jdict['aux_obs'] = self.aux_obs


# -------------------------------------------------------------------------
class Norm(JSONSerializable) :
  """Class representing the normalization term of a sample
  
  This is the base class for other types of normalization, and
  also defines a constant normalization 
  """

  def __init__(self) :
    """Create a new Norm object
    """
    pass
  
  def implicit_impact(self, par : ModelNP, variation : float = +1) -> list :
    """provides the NP variations that are implicit in the norm

      Args:
         par       : the nuisance parameter for which to get variations
         variation : magnitude of the variation, in numbers of sigmas        
    """
    return None

  @abstractmethod
  def value(self, pars_dict : dict) -> float :
    """Computes the overall normalization factor

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         normalization factor value
    """
    pass

  
  def gradients(self, pars_dict : dict) -> dict :
    """Computes gradients of the normalization wrt parameters

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         known gradients, in the form { par_name: dN/dpar },
         or `None` if gradients are not defined
    """
    return None

  @abstractmethod
  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    pass


# -------------------------------------------------------------------------
class NumberNorm(JSONSerializable) :
  """Class representing the normalization term of a sample as a float

  Attributes:
     norm_value (float) : value of the normalization term
  """

  type_str = 'number'

  def __init__(self, norm_value : float = None) :
    """Create a new Norm object
    
    Args:
      norm_value : normalization value
    """
    self.norm_value = norm_value
  
  def value(self, pars_dict : dict) -> float :
    """Computes the overall normalization factor

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         normalization factor value
    """
    return self.norm_value

  def gradients(self, pars_dict : dict) -> dict :
    """Computes gradients of the normalization wrt parameters
      
      Here the norm is constant so the gradients are defined,
      but all are 0. In this case, return an empty dict
      
      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         known gradients, in the form { par_name: dN/dpar }
    """
    return {}

  def load_jdict(self, jdict) -> 'NumberNorm' :
    """Load object information from a dictionary of JSON data

      Args:
        jdict: A dictionary containing JSON data

      Returns:
        self
    """
    if 'norm_type' in jdict and jdict['norm_type'] != NumberNorm.type_str :
      raise ValueError("Cannot load normalization data defined for type '%s' into object of type '%s'" % (jdict['norm_type'], NumberNorm.type_str))
    self.norm_value = self.load_field('norm', jdict, None, float)
    return self

  def fill_jdict(self, jdict) :
    """Save information to a dictionary of JSON data

      Args:
         jdict: A dictionary containing JSON data
    """
    jdict['norm_type'] = NumberNorm.type_str
    jdict['norm'] = self.norm_value

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    return '%s' % self.norm_value


# -------------------------------------------------------------------------
class ParameterNorm(JSONSerializable) :
  """Class representing the normalization term of a sample as the
  value of a single parameter

  Attributes:
     par_name     (str) : name of the parameter giving the normalization term
  """

  type_str = 'parameter'

  def __init__(self, par_name : str = '') :
    """Create a new Norm object
    
    Args:
      par_name : name of the parameter that defines the norm value
    """
    self.par_name = par_name
  
  def implicit_impact(self, par : ModelNP, variation : float = +1) -> list :
    """provides the NP variations that are implicit in the norm

      Args:
         par       : the nuisance parameter for which to get variations
         variation : magnitude of the NP variation, in numbers of sigmas
         normalize : if True, return the variation scaled to a +1sigma effect

      Returns:
        the relative yield variation for the specified NP variation
    """
    if par.name != self.par_name or par.nominal_value <= 0 : 
      rel_var = 0
    else :
      rel_var = variation*par.variation/par.nominal_value
    return { '%+g' % variation : rel_var, '%+g' % -variation : -rel_var }

  def value(self, pars_dict : dict) -> float :
    """Computes the overall normalization factor

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         normalization factor value
    """
    if not self.par_name in pars_dict :
      raise KeyError("Cannot to compute normalization as the value of an unknown parameter '%s'. Known parameters are as follows: %s" % (self.par_name, str(pars_dict)))
    return pars_dict[self.par_name]

  def gradients(self, pars_dict : dict) -> dict :
    """Computes gradients of the normalization wrt parameters

     Only one gradient is non-zero, return this one

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         known gradients, in the form { par_name: dN/dpar }
    """
    return { self.par_name : 1 }

  def load_jdict(self, jdict) -> 'ParameterNorm' :
    """Load object information from a dictionary of JSON data

      Args:
        jdict: A dictionary containing JSON data

      Returns:
        self
    """
    if 'norm_type' in jdict and jdict['norm_type'] != ParameterNorm.type_str :
      raise ValueError("Cannot load normalization data defined for type '%s' into object of type '%s'" % (jdict['norm_type'], ParameterNorm.type_str))
    self.par_name = self.load_field('norm', jdict, '', str)
    return self

  def fill_jdict(self, jdict) :
    """Save information to a dictionary of JSON data

      Args:
         jdict: A dictionary containing JSON data
    """
    jdict['norm_type'] = ParameterNorm.type_str
    jdict['norm'] = self.par_name

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    return '%s' % self.par_name


# -------------------------------------------------------------------------
class LinearCombNorm(JSONSerializable) :
  """Class representing the normalization term of a sample as the
  linear combination of parameter values

  Attributes:
     pars_coeffs (dict) : dict with the format { par_name : par_coeff } 
                          mapping parameter names to their coefficients
                          in the linear combination
  """

  type_str = 'linear_combination'

  def __init__(self, pars_coeffs : dict = {}) :
    """Create a new Norm object
    
    Args:
      par_name : name of the parameter that defines the norm value
    """
    self.pars_coeffs = pars_coeffs
  
  def implicit_impact(self, par : ModelNP, variation : float = +1) -> list :
    """provides the NP variations that are implicit in the norm

      Args:
         par       : the nuisance parameter for which to get variations
         variation : magnitude of the NP variation, in numbers of sigmas
         normalize : if True, return the variation scaled to a +1sigma effect

      Returns:
        the relative yield variation for the specified NP variation
    """
    if not par.name in self.pars_coeffs or par.nominal_value == 0 : 
      rel_var = 0
    else :
      rel_var = variation*par.variation/par.nominal_value*self.pars_coeffs[par.name]
    return { '%+g' % variation : rel_var, '%+g' % -variation : -rel_var }

  def value(self, pars_dict : dict) -> float :
    """Computes the overall normalization factor

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         normalization factor value
    """
    val = 0
    for par_name in self.pars_coeffs :
      if not par_name in pars_dict :
        raise KeyError("Cannot to compute normalization as linear combination of the unknowm parameter '%s'. Known parameters are as follows: %s" % (par_name, str(pars_dict)))
      val += self.pars_coeffs[par_name]*pars_dict[par_name]
    return val

  def gradients(self, pars_dict : dict) -> dict :
    """Computes gradients of the normalization wrt parameters

     Non-zero gradients are given by the linear coefficients

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         known gradients, in the form { par_name: dN/dpar }
    """
    return { par_name : self.pars_coeffs[par_name] for par_name in self.pars_coeffs }

  def load_jdict(self, jdict) -> 'LinearCombNorm' :
    """Load object information from a dictionary of JSON data

      Args:
        jdict: A dictionary containing JSON data

      Returns:
        self
    """
    if 'norm_type' in jdict and jdict['norm_type'] != LinearCombNorm.type_str :
      raise ValueError("Cannot load normalization data defined for type '%s' into object of type '%s'" % (jdict['norm_type'], LinearCombNorm.type_str))
    self.pars_coeffs = self.load_field('norm', jdict, {}, dict)
    return self

  def fill_jdict(self, jdict) :
    """Save information to a dictionary of JSON data

      Args:
         jdict: A dictionary containing JSON data
    """
    jdict['norm_type'] = LinearCombNorm.type_str
    jdict['norm'] = self.pars_coeffs

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    return ''.join([ '%+g*%s' % (self.pars_coeffs[par_name], par_name) for par_name in self.pars_coeffs ])


# -------------------------------------------------------------------------
class FormulaNorm(JSONSerializable) :
  """Class representing the normalization term of a sample
     as a formula expression.

  Attributes:
     formula  (str) : a string defining the normalization term
                      as a function of the POIs
  """

  type_str = 'formula'

  def __init__(self, formula : str = '') :
    """Create a new FormulaNorm object
    
    Args:
       formula : the formula expression for the norm
    """
    self.formula = formula

  def implicit_impact(self, par : ModelNP, variation : float = +1) -> list :
    """Provides the NP variations that are implicit in the norm
    
    This cannot be done by default for a generic formula, so return nothing
    
    Args:
         par       : the nuisance parameter for which to get variations
         variation : magnitude of the variation, in numbers of sigmas        
    """
    return None

  def value(self, pars_dict : dict) -> float :
    """Computes the overall normalization factor

      Args:
         pars_dict : a dictionary of parameter { name: value } pairs
      Returns:
         normalization factor value
    """
    try:
      return eval(self.formula, pars_dict)/self.nominal_norm
    except Exception as inst:
      print("Error while evaluating the normalization '%s' of sample '%s'." % (self.formula, self.name))
      raise(inst)

  def load_jdict(self, jdict) -> 'FormulaNorm' :
    """Load object information from a dictionary of JSON data

      Args:
        jdict: A dictionary containing JSON data

      Returns:
        self
    """
    if jdict['norm_type'] != FormulaNorm.type_str :
      raise ValueError("Cannot load normalization data defined for type '%s' into object of type '%s'" % (jdict['norm_type'], FormulaNorm.type_str))
    self.formula = self.load_field('norm', jdict, '', str)
    return self

  def fill_jdict(self, jdict) :
    """Save information to a dictionary of JSON data

      Args:
         jdict: A dictionary containing JSON data
    """
    jdict['norm_type'] = FormulaNorm.type_str
    jdict['norm'] = self.formula

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    return 'formula[%s]' % self.formula


# -------------------------------------------------------------------------
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
          corresponding to :math:`n\sigma` variations
          in the value of each NP; provided as a python dict
          with format { np_name: np_impacts } with the impacts provided
          as a list (over bins) of small dicts { nsigmas : relative_variation }
          for all the known variations.
     pos_vars : internal storage of variations used to define the impact
                coefficients for each NP (positive values only)
     neg_vars : internal storage of variations used to define the impact
                coefficients for each NP (negative values only)
     pos_imps : internal storage of variations used to define the impact
                coefficients for each NP (positive values only)
     neg_imps : internal storage of variations used to define the impact
                coefficients for each NP (negative values only)
  """

  def __init__(self, name : str = '', norm : Norm = None, nominal_norm : float = None,
               nominal_yields : np.ndarray = None, impacts : dict = {}) :
    """Create a new Sample object

      Args:
         name: sample name
         norm: object describing the sample normalization
         nominal_norm: value of the normalization factor for which the nominal yields are provided
         nominal_yields: nominal event yields for all channel bins (np. array of size `nbins`)
         impacts: relative change of the nominal event yields for each NP (see class description for format)
    """
    self.name = name
    self.norm = norm
    self.nominal_norm = nominal_norm
    self.nominal_yields = nominal_yields
    self.impacts = impacts
    self.pos_vars = {}
    self.neg_vars = {}
    self.pos_imps = {}
    self.neg_imps = {}

  def set_np_data(self, nps : list, variation : float = 1) :
    """Post-initialization update based on model NPs

    The function performs a couple of updates that require passing
    the model NPs, which isn't available at initialization time.
    It is called by the model just before the sample data is loaded,
    in :meth:`fastprof.Model.set_internal_vars`.
    
    The updates are

    * Update the nominal_norm field: this can be set in the JSON or constructor,
      but it can be natural to take it as the value obtained from the nominal
      model parameter values. 
    
    * Update the nominal yields : if `None`, we assume there is just one bin,
      and that the content matches the nominal norm (as would be the case if
      the normalization is a raw number of events)
    
    * Update the `self.impacts` array, which stores the impact of all known NPs.
      This is first loaded from JSON or specified in the constructor, and should
      list all NPs. However the impact of NPs which enter only the normalization
      can be deduced directly, if the normalization has a simple form. The
      function sets these impacts if they are not present (if already defined,
      they are not overwritten).
    
    Args:
      nps       : list of all model NPs
      variation : the +/- variation to store for norm NPs
    """
    if self.nominal_norm is None :
      nominal_pars = { par.name : par.nominal_value for par in nps }
      try :
        self.nominal_norm = self.norm.value(nominal_pars)
      except Exception as inst :
        #print("Cannot fill in empty nominal_norm field for sample '%s' using the following parameter values: %s" % (self.name, str(nominal_pars)))
        #raise(inst)
        print("Using normalization = 1 for sample '%s'." % self.name)
        self.nominal_norm = 1
    if self.nominal_yields is None :
      self.nominal_yields = np.array([ self.nominal_norm ])
    for par in nps :
      if par.name in self.impacts : continue
      impacts = [ self.norm.implicit_impact(par, +variation) ]
      if impacts is None : continue
      self.impacts[par.name] =  impacts * len(self.nominal_yields)

  def available_variations(self, par : str = None) -> list :
    """provides the available variations of the per-bin event yields for a given NP

      Args:
         par       : name of the NP
      Returns:
        list of available variations
    """
    if par is None and len(self.impacts) > 0 : par = list(self.impacts.keys())[0]
    if not par in self.impacts : raise KeyError("No impact defined in sample '%s' for parameter '%s'." % (self.name, par))
    if len(self.impacts[par]) == 0 : return []
    return [ +1 if var == 'pos' else -1 if var == 'neg' else float(var) for var in self.impacts[par][0].keys() ]

  def impact(self, par : str, variation : float = +1, normalize : bool = False) -> np.array :
    """provides the relative variations of the per-bin event yields for a given NP

      Args:
         par       : name of the NP
         variation : magnitude of the variation, in numbers of sigmas
         normalize : if True, return the variation scaled to a +1sigma effect
      Returns:
         per-bin relative variations (shape : nbins)
    """
    if not par in self.impacts : 
      raise KeyError('No impact defined in sample %s for parameters %s.' % (self.name, par))
    imp = None
    try:
      imp = np.array([ imp['%+g' % variation] for imp in self.impacts[par] ], dtype=float)
    except Exception as inst:
      # Legacy naming scheme
      if (variation == 1 or variation == -1) :
        which = 'pos' if variation == 1 else 'neg'
        try:
          imp = np.array([ imp[which] for imp in self.impacts[par] ], dtype=float)
        except Exception as inst:
          print('Impact lookup failed for sample %s, parameter %s, variation %+g' % (self.name, par, variation))
          raise(inst)
    if imp is None : return imp
    return (1 + imp)**(1/variation) - 1 if normalize else imp

  def impact_coefficients(self, par : str, variations = None, is_log : bool = False) -> list :
    """Returns a parameterization of the impact values for an NP

      Impacts are computed for fixed variations (+1, -1, etc.). In the model,
      an interpolation must be used based on these values. The simplest case
      is for a single variation, where a linear interpolation is used. For
      cases with more interpolation points, a polynomial is used.

      The interpolation can be in linear scale (suitable for interpolating
      yields as N0*(1 + pol(NP))) or log scale (as N0*exp(pol(NP))).

      Args:
         par       : name of the NP
         variation : list of variations to consider (default: None, uses what is available)
         is_bool   : interpolate in log (True) or linear (False) scale
      Returns:
         Polynomial coefficients of the interpolation (pair of np.arrays of shape (nbins, len(variations))
    """
    if variations is not None :
      self.pos_vars[par] = [+v for v in variations ]
      self.neg_vars[par] = [-v for v in variations ]
    else :
      available = self.available_variations(par)
      self.pos_vars[par] = sorted([ v for v in available if v > 0 ])
      self.neg_vars[par] = sorted([ v for v in available if v < 0 ])
    self.pos_imps[par] = np.array([ self.impact(par, var) for var in self.pos_vars[par] ])
    self.neg_imps[par] = np.array([ self.impact(par, var) for var in self.neg_vars[par] ])
    try:
      max_impact = 100
      pos_imp_vals = [ pos_imp if not is_log else np.log(np.maximum(np.minimum(1 + pos_imp, max_impact), 1/max_impact)) for pos_imp in self.pos_imps[par].T ]
      neg_imp_vals = [ neg_imp if not is_log else np.log(np.maximum(np.minimum(1 + neg_imp, max_impact), 1/max_impact)) for neg_imp in self.neg_imps[par].T ]
      pos_params = np.array([ self.interpolate_impact(self.pos_vars[par], pos_imp) for pos_imp in pos_imp_vals ]).T
      neg_params = np.array([ self.interpolate_impact(self.neg_vars[par], neg_imp) for neg_imp in neg_imp_vals ]).T
    except Exception as inst:
      print("ERROR: impact computation failed for parameter '%s'" % par)
      raise(inst)
    return pos_params, neg_params

  def interpolate_impact(self, pos : list, impacts : np.array) -> np.array :
    """Returns polynomial approximant to the impact valust
      Args:
         pos : list of parameter variation values
         impacts : list of corresponding impacts
      Returns:
         Polynomial coefficients of the interpolation
    """
    if len(pos) != len(impacts) : raise ValueError("Cannot interpolate, number of x values (%d) doesn't match y values (%d)." % (len(pos), len(impacts)))
    vdm = np.array( [ [ p**(n+1) for n in range(0, len(pos)) ] for p in pos ] )
    #print(pos, impacts, vdm)
    return np.linalg.inv(vdm).dot(impacts)

  def sym_impact(self, par : str) -> np.array :
    """Provides the symmetrized relative variations of the per-bin event yields for a given NP

      Args:
         par : name of the NP
      Returns:
         symmetrized per-bin relative variations
    """
    try:
      #return np.sqrt((1 + self.impact(par, +1))*(1 + self.impact(par, -1, normalize=True))) - 1
      return 0.5*(self.impact(par, +1) - self.impact(par, -1))
    except Exception as inst:
      print('Symmetric impact computation failed, returning the positive impacts instead')
      print(inst)
      return self.impact(par, +1)

  def normalization(self, pars_dict : dict) -> float :
    """Computes the overall normalization factor

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         normalization factor value
    """
    return self.norm.value(pars_dict)/self.nominal_norm

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    s = "Sample '%s', norm = %s (nominal = %s)" % (self.name, str(self.norm), str(self.nominal_norm))
    return s

  def load_jdict(self, jdict) -> 'Sample' :
    """Load object information from a dictionary of JSON data

      Args:
        jdict: A dictionary containing JSON data

      Returns:
        self
    """
    self.name = self.load_field('name', jdict, '', str)
    norm_type = self.load_field('norm_type', jdict, '', str)
    # Automatic typecasting: either NumberNorm or ParameterNorm,
    # depending if the 'norm' parameter represents a float
    if norm_type == '' :
      try:        
        norm = self.load_field('norm', jdict, None, float)
      except:
        norm = None
      norm_type = NumberNorm.type_str if norm is not None else ParameterNorm.type_str
    if norm_type == NumberNorm.type_str :
      self.norm = NumberNorm()
    elif norm_type == ParameterNorm.type_str :
      self.norm = ParameterNorm()
    elif norm_type == FormulaNorm.type_str :
      self.norm = FormulaNorm()
    self.norm.load_jdict(jdict)
    self.nominal_norm = self.load_field('nominal_norm', jdict, None, [float, int])
    self.nominal_yields = self.load_field('nominal_yields', jdict, None, list)
    self.impacts = self.load_field('impacts', jdict, {}, dict)
    return self

  def fill_jdict(self, jdict) :
    """Save information to a dictionary of JSON data

      Args:
         jdict: A dictionary containing JSON data
    """
    jdict['name'] = self.name
    self.norm.fill_jdict(jdict)
    jdict['nominal_norm'] = self.nominal_norm
    jdict['nominal_yields'] = self.nominal_yields
    jdict['impacts'] = self.impacts


# -------------------------------------------------------------------------
class Channel(JSONSerializable) :
  """Class representing a model channel

  Provides the functionality for HistFactory channel structures,
  representing a set of measurement bins. Two types of channels
  are currently implemented, differing in how the
  bin list is handled:

  * `bin` : a channel with a single measurement bin

  * `binned_range` : a channel with multiple bins spanning
    a range of a continuous observable.

  Each type corresponds to a different class derived from
  this one: :class:`SingleBinChannel` and :class:`BinnedRangeChannel`
  respectively.

  This class is the common base, defining :
  * The channel name

  * A list of :class:`Sample` objects representing the processes
    contributing to the event yield in each bin.

  Attributes:
     name (str) : the name of the channel
     bins (list) : a list of python dict objects defining each bin
     samples (dict) : the channel samples, as a dict mapping the sample names
        to the sample objects (see :class:`Sample`).
  """

  def __init__(self, name : str = '', bins : list = []) :
    """Initializes the Channel class

      Args:
         name : channel name
         bins : list of bin defiinitions
    """
    self.name = name
    self.samples = {}

  @abstractmethod
  def nbins(self) -> int :
    """Returns the number of bins in the channel

      Returns:
        the number of bins
    """
    pass

  def sample(self, name : str) :
    """Access a sample by name

    Args:
      name : the sample name

    Returns:
      the named sample, or `None` if not defined
    """
    return self.samples[name] if name in self.samples else None

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    s = "Channel '%s'" % self.name
    for sample in self.samples.values() : s += '\n    o ' + str(sample)
    return s

  def load_jdict(self, jdict : dict) -> 'Channel' :
    """Load object information from a dictionary of JSON data

      Args:
        jdict: A dictionary containing JSON data

      Returns:
        self
    """
    self.name = jdict['name']
    for json_sample in jdict['samples'] :
      sample = Sample()
      sample.load_jdict(json_sample)
      self.samples[sample.name] = sample
    return self

  def fill_jdict(self, jdict : dict) :
    """Save information to a dictionary of JSON data

      Args:
         jdict: A dictionary containing JSON data
    """
    jdict['name'] = self.name
    jdict['samples'] = []
    for sample in self.samples : jdict['samples'].append(sample.dump_jdict())

  def load_data_jdict(self, jdict : dict, counts : np.array) :
    """Load observed data information from JSON
          raise KeyError("Data channel definition must contain a 'type' field")

    Utility function called from :class:`fastprof.Data` to parse
    an observed data specification for this channel and fill an
    array of event counts

      Args:
        jdict: A dictionary containing JSON data
        counts : the array of data counts to fill
    """
    if not 'name' in jdict :
      raise KeyError("Data channel definition must contain a 'name' field")
    if jdict['name'] != self.name :
      raise ValueError("Cannot load channel data defined for channel '%s' into channel '%s'" % (jdict['name'], self.name))
    
  def save_data_jdict(self, jdict : dict, counts : np.array) :
    """Save observed data information from JSON
    
    Utility function called from :class:`fastprof.Data` to write
    out an observed data specification for this channel into the
    appropriate format

    Args:
      jdict  : A dictionary to fill with JSON data
      counts : an array of data counts to read from
    """
    jdict['name'] = self.name
    if len(counts) != self.nbins() :
      raise ValueError("Cannot save channel data counts of length %d for channel '%s' with %d bins" % (len(counts), self.name, self.nbins()))


# -------------------------------------------------------------------------
class BinnedRangeChannel(Channel) :
  """Class representing a model channel consisting of a binned observable range

  Class derived from :class:`Channel` to handle a bin channel consisting
  of a list of bins for a continuous observable.
 
  In this case, each bin is stored as a dict with the format
  { 'lo_edge' : <float value>, 'hi_edge': <float value> }
  providing the lower and upper range of the bin

    * `count` type: dict of the form { 'name' : <bin name> }
  """
  type_str = 'binned_range'

  def __init__(self, name : str = '', bins : list = []) :
    """Initializes the BinnedRangeChannel class

      Args:
         name : channel name
         bins : list of bin definitions, each in the form of a dict
                with format { 'lo_edge' : <float value>, 'hi_edge': <float value> }
    """
    super().__init__(name)
    self.bins = bins

  def nbins(self) -> int :
    """Returns the number of bins in the channel

      Returns:
        the number of bins
    """
    return len(self.bins)

  def load_jdict(self, jdict : dict) :
    """Load object information from a dictionary of JSON data

      Args:
        jdict: A dictionary containing JSON data

      Returns:
        Channel: self
    """
    super().load_jdict(jdict)
    if jdict['type'] != BinnedRangeChannel.type_str :
      raise ValueError("Trying to load a BinnedRangeChannel from channel data of type '%s'" % jdict['type'])
    self.bins = jdict['bins']
    self.obs_name = self.load_field('obs_name', jdict, '', str)
    self.obs_unit = self.load_field('obs_unit', jdict, '', str)
    return self

  def fill_jdict(self, jdict) :
    """Save information to a dictionary of JSON data

      Args:
         jdict: A dictionary containing JSON data
    """
    super().fill_jdict(jdict)
    jdict['bins'] = self.bins
    jdict['obs_name'] = self.obs_name
    jdict['obs_unit'] = self.obs_unit

  def load_data_jdict(self, jdict : dict, counts : np.array) :
    """Load observed data information from JSON
    
    Utility function called from :class:`fastprof.Data` to parse
    an observed data specification for this channel and fill an
    array of event counts

      Args:
        jdict: A dictionary containing JSON data
        counts : the array of data counts to fill
    """
    super().load_data_jdict(jdict, counts)
    if not 'name' in jdict :
      raise KeyError("Data channel definition must contain a 'name' field")
    if not 'bins' in jdict :
      raise KeyError("No 'bins' section defined for data channel '%s' in specified JSON file." % self.name)
    if len(jdict['bins']) != self.nbins() :
      raise ValueError("Binned range channel '%s' in specified JSON file has %d bins, but the model channel has %d." % (channel['name'], len(channel['bins']), self.nbins()))
    for b, bin_data in enumerate(jdict['bins']) :
      if bin_data['lo_edge'] != self.bins[b]['lo_edge'] or bin_data['hi_edge'] != self.bins[b]['hi_edge'] :
        raise ValueError("Bin %d in data channel '%s' spans [%g,%g], but the model bin spans [%g,%g]." %
                         (b, self.name, bin_data['lo_edge'], bin_data['hi_edge'], self.bins[b]['lo_edge'], self.bins[b]['hi_edge']))
      counts[b] = bin_data['counts']

  def save_data_jdict(self, jdict : dict, counts : np.array) :
    """Save observed data information from JSON
    
    Utility function called from :class:`fastprof.Data` to write
    out an observed data specification for this channel into the
    appropriate format

    Args:
      jdict  : A dictionary to fill with JSON data
      counts : an array of data counts to read from
    """
    super().save_data_jdict(jdict, counts)
    jdict['type'] = BinnedRangeChannel.type_str
    jdict['obs_name'] = self.obs_name
    jdict['obs_unit'] = self.obs_unit
    jdict['bins'] = []
    for b, bin_spec in enumerate(self.bins) :
      bin_data = {}
      bin_data['lo_edge'] = bin_spec['lo_edge']
      bin_data['hi_edge'] = bin_spec['hi_edge']
      bin_data['counts'] = int(counts[b])
      jdict['bins'].append(bin_data)


# -------------------------------------------------------------------------
class SingleBinChannel(Channel) :
  """Class representing a model channel consisting of a single bin

  Class derived from :class:`Channel` to handle the case of a single 
  counting bin
  
  In this case, each bin is stored as a dict with the format
  { 'name' : <name> }.
  """

  type_str = 'bin'

  def __init__(self, name : str = '') :
    """Initializes the BinnedRangeChannel class

      Args:
         name : channel name (and bin name)
    """
    super().__init__(name)

  def nbins(self) -> int :
    """Returns the number of bins in the channel

      Returns:
        the number of bins
    """
    return 1

  def load_data_jdict(self, jdict : dict, counts : np.array) :
    """Load observed data information from JSON
    
    Utility function called from :class:`fastprof.Data` to parse
    an observed data specification for this channel and fill an
    array of event counts

      Args:
        jdict: A dictionary containing JSON data
        counts : the array of data counts to fill
    """
    super().load_data_jdict(jdict, counts)
    if 'type' in jdict and jdict['type'] != SingleBinChannel.type_str :
      raise ValueError("Cannot load channel data defined for type '%s' into channel of type '%s'" % (jdict['type'], SingleBinChannel.type_str))
    if not 'counts' in jdict :
      raise KeyError("No 'counts' section defined for data channel '%s' in specified JSON file." % self.name)
    counts[0] = jdict['counts']

  def save_data_jdict(self, jdict : dict, counts : np.array) :
    """Save observed data information from JSON
    
    Utility function called from :class:`fastprof.Data` to write
    out an observed data specification for this channel into the
    appropriate format

    Args:
      jdict  : A dictionary to fill with JSON data
      counts : an array of data counts to read from
    """
    super().save_data_jdict(jdict, counts)
    jdict['type'] = SingleBinChannel.type_str
    jdict['counts'] = int(counts[0])
