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
  
  def implicit_impact(self, par : ModelNP, reals : dict, real_vals : dict, variation : float = +1) -> list :
    """provides the NP variations that are implicit in the norm

      Args:
         par       : the nuisance parameter for which to get variations
         variation : magnitude of the variation, in numbers of sigmas        
    """
    return None

  @abstractmethod
  def value(self, real_vals : dict) -> float :
    """Computes the overall normalization factor

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         normalization factor value
    """
    pass

  
  def gradient(self, pois : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes gradient of the normalization wrt parameters

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         known gradient, in the form { par_name: dN/dpar },
         or `None` if gradient are not defined
    """
    return None

  def hessian(self, pois : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes gradient of the normalization wrt parameters

      Args:
         real_vals : a dictionary of parameter name: value pairs
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

  @classmethod
  def instantiate(cls, sdict, load_data : bool = True) :
    """Instantiate an Expression object from one of the possibilities below.

      Args:
         cls: this class
         sdict: A dictionary containing markup data
    """
    norm_type = cls.load_field('norm_type', sdict, '', str)
    if norm_type == '' :
      norm = cls.load_field('norm', sdict, None, [int, float, str])
      if isinstance(norm, (int, float)) or norm is None :
        norm_type = NumberNorm.type_str
      else :
        try:
          float(norm)
          norm_type = NumberNorm.type_str
        except Exception as inst:
          norm_type = ExpressionNorm.type_str
    if norm_type == NumberNorm.type_str :
      norm = NumberNorm()
    elif norm_type == ExpressionNorm.type_str :
      norm = ExpressionNorm()
    else :
      raise KeyError("Unknown normalisation type '%s'." % norm_type)
    norm.load_dict(sdict)
    return norm


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
  
  def implicit_impact(self, par : ModelNP, reals : dict, real_vals : dict, variation : float = +1) -> list :
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

  def value(self, real_vals : dict) -> float :
    """Computes the overall normalization factor

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         normalization factor value
    """
    return self.norm_value

  def gradient(self, pois : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes gradient of the normalization wrt parameters
      
      Here the norm is constant so the gradient are all 0.
      
      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
        gradient vector, with size len(pois)
    """
    return np.zeros(len(pois))

  def hessian(self, pois : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes gradient of the normalization wrt parameters
      
      Here the norm is constant so the hessian are all 0.
      
      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
        Hessian matrix
    """
    return np.zeros((len(pois), len(pois)))

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
  
  def implicit_impact(self, par : ModelNP, reals : dict, real_vals : dict, variation : float = +1) -> list :
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
    rel_var = variation*par.variation*self.gradient({ par.name : par }, reals, real_vals)[0]/par.nominal_value if par.nominal_value != 0 else 0
    return { '%+g' % variation : rel_var, '%+g' % -variation : -rel_var }

  def value(self, real_vals : dict) -> float :
    """Computes the overall normalization factor

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         normalization factor value
    """
    if self.expr_name == '' : return 1
    if not self.expr_name in real_vals :
      raise KeyError("Cannot compute normalization as the value of the unknown expression '%s'. Known parameters are as follows: %s" % (self.expr_name, str(real_vals)))
    return real_vals[self.expr_name]

  def gradient(self, pois : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes gradient of the normalization wrt parameters

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         known gradient, in the form { expr_name: dN/dpar }
    """
    return reals[self.expr_name].gradient(pois, reals, real_vals)

  def hessian(self, pois : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes hessian of the normalization wrt parameters

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         known gradient, in the form { expr_name: dN/dpar }
    """
    return reals[self.expr_name].hessian(pois, reals, real_vals)

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
