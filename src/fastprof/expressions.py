from .base import Serializable
import numpy as np
from abc import abstractmethod


# -------------------------------------------------------------------------
class Expression(Serializable) :
  """Class representing an expression involving POIs and/or NPs

  Attributes:
     pars_coeffs (dict) : dict with the format { par_name : par_coeff } 
                          mapping parameter names to their coefficients
                          in the linear combination
  """

  def __init__(self, name : str = '') :
    """Create a new Expression object
    """
    self.name = name
    super().__init__()

  def value(self, pars_dict : dict) -> float :
    """Computes the expression value

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         expression value
    """
    pass

  def gradient(self, pars_dict : dict) -> dict :
    """Computes gradient of the expression wrt parameters

     Non-zero gradient is given by the linear coefficients

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         known gradients, in the form { par_name: dN/dpar }
    """
    pass # TODO: add recursive calls to clients to define gradients of complex functions directly

  def replace(self, name, value) -> float :
    """Replaces parameter 'name' by a numberical value

       If this is not possible, None is returned. If this makes the expression
       trivial, the resulting numerical value of the expression is returned.

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         self for sucess, None for failure, and a number if the expression is now trivial
    """
    return None # TODO: add recursive calls to clients to make this safer

  def load_dict(self, sdict) -> 'Expression' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    self.name  = sdict['name']
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    sdict['name'] = self.name

  @classmethod
  def instantiate(cls, sdict, load_data : bool = True) :
    """Instantiate an Expression object from one of the possibilities below.

      Args:
         cls: this class
         sdict: A dictionary containing markup data
    """
    if not 'type' in sdict or sdict['type'] == Formula.type_str :
      expression = Formula()
    elif sdict['type'] == LinearCombination.type_str :
      expression = LinearCombination()
    else :
      raise ValueError("ERROR: unsupported expression type '%s'" % sdict['type'])
    if load_data : expression.load_dict(sdict)
    return expression

# -------------------------------------------------------------------------
class LinearCombination(Expression) :
  """Class representing a linear combination of parameters

  Attributes:
     pars_coeffs (dict) : dict with the format { par_name : par_coeff } 
                          mapping parameter names to their coefficients
                          in the linear combination
  """

  type_str = 'linear_combination'

  def __init__(self, name : str = '', nominal_value : float = 0, pars_coeffs : dict = {}) :
    """Create a new LinearCombination object
    
    Args:
      par_coeffs : dict specifying the parameters and coefficients of the 
                   linear combination, with entries of the form par_name: coeff 
    """
    super().__init__(name)
    self.nominal_value = nominal_value
    self.pars_coeffs = pars_coeffs

  def value(self, pars_dict : dict) -> float :
    """Computes the expression value

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         expression value
    """
    val = self.nominal_value
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

  def replace(self, name, value) -> float :
    """Replaces parameter 'name' by a numberical value
    
       If this is not possible, None is returned. If this makes the expression
       trivial, the resulting numerical value of the expression is returned.

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         self for sucess, None for failure, and a number if the expression is now trivial
    """
    if not name in self.pars_coeffs :
      raise KeyError("Parameter '%s' was not found in expression '%s'." % (name, self.name))
    self.nominal_value += self.pars_coeffs[name]*value
    del self.pars_coeffs[name]
    if len(self.pars_coeffs) > 0 : return self
    return self.nominal_value

  def load_dict(self, sdict) -> 'LinearCombination' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    super().load_dict(sdict)
    if 'type' in sdict and sdict['type'] != LinearCombination.type_str :
      raise ValueError("Cannot load normalization data defined for type '%s' into object of type '%s'" % (sdict['type'], LinearCombination.type_str))
    self.nominal_value = self.load_field('nominal_value', sdict, 0,  [float, int])
    self.pars_coeffs = self.load_field('pars_coeffs', sdict, {}, dict)
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    super().fill_dict(sdict)
    sdict['type'] = LinearCombination.type_str
    sdict['nominal_value'] = self.nominal_value
    sdict['pars_coeffs'] = self.pars_coeffs

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    return ''.join([ self.nominal_value ] + [ '%+g*%s' % (self.pars_coeffs[par_name], par_name) for par_name in self.pars_coeffs ])


# -------------------------------------------------------------------------
class Formula(Expression) :
  """Class representing the normalization term of a sample
     as a formula expression.

  Attributes:
     formula  (str) : a string defining the normalization term
                      as a function of the POIs
  """

  type_str = 'formula'

  def __init__(self, name : str = '', formula : str = '') :
    """Create a new Formula object
    
    Args:
       formula : the formula expression
    """
    super().__init__(name)
    self.formula = formula

  def value(self, pars_dict : dict) -> float :
    """Computes the expression value

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         expression value
    """
    try:
      return eval(self.formula, pars_dict)
    except Exception as inst:
      print("Error while evaluating normalization formula '%s'." % self.formula)
      raise(inst)

  def load_dict(self, sdict) -> 'FormulaFunction' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    super().load_dict(sdict)
    if sdict['type'] != FormulaFunction.type_str :
      raise ValueError("Cannot load normalization data defined for type '%s' into object of type '%s'" % (sdict['type'], FormulaFunction.type_str))
    self.formula = self.load_field('formula', sdict, '', str)
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    super().fill_dict(sdict)
    sdict['type'] = Formula.type_str
    sdict['formula'] = self.formula

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    return 'formula[%s]' % self.formula

