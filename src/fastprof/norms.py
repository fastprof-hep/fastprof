"""Module containing the building blocks for fastprof models:

  * :class:`Norm` objects, defining the normalization of a sample

"""

from .base import Serializable, ModelNP
import numpy as np
from abc import abstractmethod


# -------------------------------------------------------------------------
class Norm(Serializable) :
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

  
  def gradient(self, pars_dict : dict) -> dict :
    """Computes gradient of the normalization wrt parameters

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         known gradient, in the form { par_name: dN/dpar },
         or `None` if gradient are not defined
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
class NumberNorm(Serializable) :
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
  
  def implicit_impact(self, par : ModelNP, variation : float = +1) -> list :
    """provides the NP variations that are implicit in the norm

      This is called only for NPs. If the normalization parameter is an NP,
      then this function will automatically provide the corresponding
      impact value on the bin contents. Here there are no variations since the
      norm is just a number, so always return None

      Args:
         par       : the nuisance parameter for which to get variations
         variation : magnitude of the NP variation, in numbers of sigmas
         normalize : if True, return the variation scaled to a +1sigma effect

      Returns:
        the relative yield variation for the specified NP variation
    """
    return None

  def value(self, pars_dict : dict) -> float :
    """Computes the overall normalization factor

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         normalization factor value
    """
    return self.norm_value

  def gradient(self, pars_dict : dict) -> dict :
    """Computes gradient of the normalization wrt parameters
      
      Here the norm is constant so the gradient are defined,
      but all are 0. In this case, return an empty dict
      
      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         known gradient, in the form { par_name: dN/dpar }
    """
    return {}

  def load_dict(self, sdict) -> 'NumberNorm' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    if 'norm_type' in sdict and sdict['norm_type'] != NumberNorm.type_str :
      raise ValueError("Cannot load normalization data defined for type '%s' into object of type '%s'" % (sdict['norm_type'], NumberNorm.type_str))
    self.norm_value = self.load_field('norm', sdict, '', [int, float, str])
    if self.norm_value == '' : self.norm_value = 1
    if isinstance(self.norm_value, str) :
      raise ValueError("Normalization data for type '%s' is the string '%s', not supported for numerical norms." % (NumberNorm.type_str, self.norm_value))
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    sdict['norm_type'] = NumberNorm.type_str
    sdict['norm'] = self.unnumpy(self.norm_value)

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    return '%s' % self.norm_value


# -------------------------------------------------------------------------
class ParameterNorm(Serializable) :
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

      This is called only for NPs. If the normalization parameter is an NP,
      then this function will automatically provide the corresponding
      impact value on the bin contents. Only +/- 1 sigma variations are
      returned, since NP impacts are anyway always linear.

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

  def gradient(self, pars_dict : dict) -> dict :
    """Computes gradient of the normalization wrt parameters

     Only one gradient is non-zero, return this one

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         known gradient, in the form { par_name: dN/dpar }
    """
    return { self.par_name : 1 }

  def load_dict(self, sdict) -> 'ParameterNorm' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    if 'norm_type' in sdict and sdict['norm_type'] != ParameterNorm.type_str :
      raise ValueError("Cannot load normalization data defined for type '%s' into object of type '%s'" % (sdict['norm_type'], ParameterNorm.type_str))
    self.par_name = self.load_field('norm', sdict, '', str)
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    sdict['norm_type'] = ParameterNorm.type_str
    sdict['norm'] = self.par_name

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    return '%s' % self.par_name


# -------------------------------------------------------------------------
class LinearCombNorm(Serializable) :
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

  def gradient(self, pars_dict : dict) -> dict :
    """Computes gradient of the normalization wrt parameters

     Non-zero gradient is given by the linear coefficients

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         known gradient, in the form { par_name: dN/dpar }
    """
    return { par_name : self.pars_coeffs[par_name] for par_name in self.pars_coeffs }

  def load_dict(self, sdict) -> 'LinearCombNorm' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    if 'norm_type' in sdict and sdict['norm_type'] != LinearCombNorm.type_str :
      raise ValueError("Cannot load normalization data defined for type '%s' into object of type '%s'" % (sdict['norm_type'], LinearCombNorm.type_str))
    self.pars_coeffs = self.load_field('norm', sdict, {}, dict)
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    sdict['norm_type'] = LinearCombNorm.type_str
    sdict['norm'] = self.pars_coeffs

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    return ''.join([ '%+g*%s' % (self.pars_coeffs[par_name], par_name) for par_name in self.pars_coeffs ])


# -------------------------------------------------------------------------
class FormulaNorm(Serializable) :
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
    
    This cannot be done by default for a generic formula, so return nothing.
    The NP impacts (if any) should be explicitly listed in the 'impacts' section
    of the model definition.
    
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
      return eval(self.formula, pars_dict)
    except Exception as inst:
      print("Error while evaluating normalization formula '%s'." % self.formula)
      raise(inst)

  def load_dict(self, sdict) -> 'FormulaNorm' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    if sdict['norm_type'] != FormulaNorm.type_str :
      raise ValueError("Cannot load normalization data defined for type '%s' into object of type '%s'" % (sdict['norm_type'], FormulaNorm.type_str))
    self.formula = self.load_field('norm', sdict, '', str)
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    sdict['norm_type'] = FormulaNorm.type_str
    sdict['norm'] = self.formula

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    return 'formula[%s]' % self.formula

