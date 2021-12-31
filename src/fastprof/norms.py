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
class ExpressionNorm(Serializable) :
  """Class representing the normalization term of a sample as the
     value of a single parameter

  Attributes:
     expr_name     (str) : name of the parameter or expression used to 
                          define the normalization term
  """

  type_str = 'expression'

  def __init__(self, expr_name : str = '') :
    """Create a new Norm object
    
    Args:
      expr_name : name of the parameter that defines the norm value
    """
    self.expr_name = expr_name
  
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
    if par.name != self.expr_name or par.nominal_value == 0 : 
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
    if not self.expr_name in pars_dict :
      raise KeyError("Cannot compute normalization as the value of an unknown expression '%s'. Known parameters are as follows: %s" % (self.expr_name, str(pars_dict)))
    return pars_dict[self.expr_name]

  def gradient(self, pars_dict : dict) -> dict :
    """Computes gradient of the normalization wrt parameters

     Only one gradient is non-zero, return this one

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         known gradient, in the form { expr_name: dN/dpar }
    """
    return { self.expr_name : 1 }

  def load_dict(self, sdict) -> 'ExpressionNorm' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    if 'norm_type' in sdict and sdict['norm_type'] != ExpressionNorm.type_str :
      raise ValueError("Cannot load normalization data defined for type '%s' into object of type '%s'" % (sdict['norm_type'], ExpressionNorm.type_str))
    self.expr_name = self.load_field('norm', sdict, '', str)
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    sdict['norm_type'] = ExpressionNorm.type_str
    sdict['norm'] = self.expr_name

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    return '%s' % self.expr_name
