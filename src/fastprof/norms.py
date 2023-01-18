"""Module containing the building blocks for fastprof models:

  * :class:`Norm` objects, defining the normalization of a sample
  
  At the moment two types of norms are defined: :class:`NumberNorm` for normalizations
  define by a simple real value, and :class:`ExpressionNorm` for a parameter expression
  (either a single POI or NP, or an expression of these -- see 
  :class:`fastprof.expressions.Expression`).

"""

from .base import Serializable, ModelNP
import numpy as np
from abc import abstractmethod


# -------------------------------------------------------------------------
class Norm(Serializable) :
  """Base class for the normalization term of a sample
  
  The base class for other types of normalization.
  """

  def __init__(self) :
    """Create a new Norm object
    """
    pass
  
  def implicit_impact(self, par : ModelNP, reals : dict, real_vals : dict, variation : float = +1) -> list :
    """Provide the NP variations that are implicit in the norm

      Args:
        par       : the nuisance parameter for which to get variations
        reals     : expressions as a dictionary of { name: object } pairs
        real_vals : values of all reals and pars, as { name: value} pairs
        variation : magnitude of the variation, in numbers of sigmas        

      Returns:
        the relative yield variation for the specified NP variation
    """
    return None

  @abstractmethod
  def value(self, real_vals : dict) -> float :
    """Compute the overall normalization factor

      Args:
        real_vals : values of all reals and pars, as { name: value} pairs
      Returns:
         normalization factor value
    """
    pass

  
  def gradient(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the gradient of the normalization wrt parameters

      Returns the gradient of the expression as a 1D array of size len(pars).
      The order of the gradients is the same as that of the `pars` arg.
    
      Args:
        pars : expression parameters as a dictionary of { name: object } pairs
        reals: sub-expressions as a dictionary of { name: object } pairs
        real_vals: values of all reals and pars, as { name: value} pairs
      Returns:
        gradients wrt parameters
    """
    return None

  def hessian(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the Hessian of the expression wrt parameters

      Returns the Hessian of the expression as a 2D array of size len(pars) x len(pars).

      Args:
        pars : expression parameters as a dictionary of { name: object } pairs
        reals: sub-expressions as a dictionary of { name: object } pairs
        real_vals: values of all reals and pars, as { name: value} pairs
      Returns:
        Hessian wrt parameter pairs
    """
    return None

  @abstractmethod
  def __str__(self) -> str :
    """Provide a description string

      Returns:
        the description string
    """
    pass

  @classmethod
  def instantiate(cls, sdict : dict, load_data : bool = True) -> 'Norm' :
    """Instantiate a Norm object from one of the possibilities below.

      Class method used when loading the norm from markup, to
      instantiate an object based type strings specified in the markup.

      Args:
         cls: this class
         sdict: dictionary containing markup data
         load_data: if `True`, also initialize the norm from markup
    """
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
  """Class representing the normalization term of a sample as a simple number

  Attributes:
     norm_value (float) : value of the normalization
  """

  type_str = 'number'

  def __init__(self, norm_value : float = None) :
    """Create a new NumberNorm object
    
    Args:
      norm_value : normalization value
    """
    self.norm_value = norm_value
  
  def implicit_impact(self, par : ModelNP, reals : dict, real_vals : dict, variation : float = +1) -> list :
    """Provides the NP variations that are implicit in the norm

      This is called only for NPs. If the normalization parameter is an NP,
      then this function will automatically provide the corresponding
      impact value on the bin contents. Here there are no variations since the
      norm is just a number, so always return None

      Args:
        par       : the nuisance parameter for which to get variations
        reals     : expressions as a dictionary of { name: object } pairs
        real_vals : values of all reals and pars, as { name: value} pairs
        variation : magnitude of the variation, in numbers of sigmas        

      Returns:
        the relative yield variation for the specified NP variation
    """
    return None

  def value(self, real_vals : dict) -> float :
    """Compute the overall normalization factor

      Args:
         real_vals : a dictionary of parameter name: value pairs
      
      Returns:
         normalization factor value
    """
    return self.norm_value

  def gradient(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the gradient of the normalization wrt parameters

      Returns the gradient of the expression as a 1D array of size len(pars).
      Here just return zeros since the norm is a constant number.
    
      Args:
        pars : expression parameters as a dictionary of { name: object } pairs
        reals: sub-expressions as a dictionary of { name: object } pairs
        real_vals: values of all reals and pars, as { name: value} pairs
      
      Returns:
        gradients wrt parameters
    """
    return np.zeros(len(pars))

  def hessian(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the Hessian of the expression wrt parameters

      Returns the Hessian of the expression as a 2D array of size len(pars) x len(pars).
      Here just return zeros since the norm is a constant number.

      Args:
        pars : expression parameters as a dictionary of { name: object } pairs
        reals: sub-expressions as a dictionary of { name: object } pairs
        real_vals: values of all reals and pars, as { name: value} pairs
      
      Returns:
        Hessian wrt parameter pairs
    """
    return np.zeros((len(pars), len(pars)))

  def load_dict(self, sdict : dict) -> 'NumberNorm' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    self.norm_value = self.load_field('norm', sdict, '', [int, float, str])
    if self.norm_value == '' : self.norm_value = 1
    if isinstance(self.norm_value, str) :
      raise ValueError("Normalization data for type '%s' is the string '%s', not supported for numerical norms." % (NumberNorm.type_str, self.norm_value))
    return self

  def fill_dict(self, sdict : dict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    sdict['norm'] = self.unnumpy(self.norm_value)

  def __str__(self) -> str :
    """Provide a description string

      Returns:
        the description string
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
    """Create a new ExpressionNorm object
    
    Args:
      expr_name : name of the model expression that defines the norm value
    """
    self.expr_name = expr_name
  
  def implicit_impact(self, par : ModelNP, reals : dict, real_vals : dict, variation : float = +1) -> list :
    """Provide the NP variations that are implicit in the norm

      This is called only for NPs. If the normalization parameter is an NP,
      then this function will automatically provide the corresponding
      impact value on the bin contents. Only +/- 1 sigma variations are
      returned, since NP impacts are anyway always linear.

      Args:
        par       : the nuisance parameter for which to get variations
        reals     : expressions as a dictionary of { name: object } pairs
        real_vals : values of all reals and pars, as { name: value} pairs
        variation : magnitude of the variation, in numbers of sigmas        

      Returns:
        the relative yield variation for the specified NP variation
    """
    rel_var = variation*par.variation*self.gradient({ par.name : par }, reals, real_vals)[0]/par.nominal_value if par.nominal_value != 0 else 0
    return { '%+g' % variation : rel_var, '%+g' % -variation : -rel_var }

  def value(self, real_vals : dict) -> float :
    """Compute the overall normalization factor

      Args:
         real_vals : a dictionary of parameter name: value pairs

      Returns:
         normalization factor value
    """
    if self.expr_name == '' : return 1
    if not self.expr_name in real_vals :
      raise KeyError("Cannot compute normalization as the value of the unknown expression '%s'. Known parameters are as follows: %s" % (self.expr_name, str(real_vals)))
    return real_vals[self.expr_name]

  def gradient(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the gradient of the normalization wrt parameters

      Returns the gradient of the expression as a 1D array of size len(pars).
      The order of the gradients is the same as that of the `pars` arg.
      The computation is delegated to the Expression object.
    
      Args:
        pars : expression parameters as a dictionary of { name: object } pairs
        reals: sub-expressions as a dictionary of { name: object } pairs
        real_vals: values of all reals and pars, as { name: value} pairs

      Returns:
        gradients wrt parameters
    """
    return reals[self.expr_name].gradient(pars, reals, real_vals)

  def hessian(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the Hessian of the expression wrt parameters

      Returns the Hessian of the expression as a 2D array of size len(pars) x len(pars).
      The order of the gradients is the same as that of the `pars` arg.
      The computation is delegated to the Expression object.

      Args:
        pars : expression parameters as a dictionary of { name: object } pairs
        reals: sub-expressions as a dictionary of { name: object } pairs
        real_vals: values of all reals and pars, as { name: value} pairs

      Returns:
        Hessian wrt parameter pairs
    """
    return reals[self.expr_name].hessian(pars, reals, real_vals)

  def load_dict(self, sdict : dict) -> 'ExpressionNorm' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    self.expr_name = self.load_field('norm', sdict, '', str)
    return self

  def fill_dict(self, sdict : dict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    sdict['norm'] = self.expr_name

  def __str__(self) -> str :
    """Provide a description string

      Returns:
        the description string
    """
    return '%s' % self.expr_name
