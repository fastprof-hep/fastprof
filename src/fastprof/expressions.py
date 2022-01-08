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

  @abstractmethod
  def value(self, pars_dict : dict) -> float :
    """Computes the expression value

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         expression value
    """
    pass

  def gradient(self, pois : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes gradient of the expression wrt parameters

     Non-zero gradient is given by the linear coefficients

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         known gradients, in the form { par_name: dN/dpar }
    """
    return None

  def replace(self, name : str, value : float, reals : dict) -> float :
    """Replaces parameter 'name' by a numberical value

       All instances of the parameter 'name'
       should be replaced by 'value'. This expression should 
       update itself to remove the dependency on 'name', and
       adjust its own numerical value according to 'value'.
       If this makes the expression trivial, the resulting
       numerical value of the expression is returned. Otherwise,
       the expession itself is returned.

      Args:
         name : the name of the parameter to replace
         value : the value to use as replacement
         reals : a dict containing all the reals in the model,
                 to recursively propagate the replace call to
                 sub-expressions.
      Returns:
         self, or just a number if the expression is now trivial
    """
    raise KeyError("ERROR: cannot replace value of parameter '%s' in expression '%s'." % (name, self.name))

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
class SingleParameter(Expression) :
  """Class representing a single parameter

  Attributes:
     pars_coeffs (dict) : dict with the format { par_name : par_coeff } 
                          mapping parameter names to their coefficients
                          in the linear combination
  """

  type_str = 'single_parameter'

  def __init__(self, par_name : str = '') :
    """Create a new LinearCombination object
    
    Args:
      par_coeffs : dict specifying the parameters and coefficients of the 
                   linear combination, with entries of the form par_name: coeff 
    """
    super().__init__(par_name)

  def value(self, real_vals : dict) -> float :
    """Computes the expression value

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         expression value
    """
    if not self.name in real_vals :
        raise KeyError("Cannot evaluate unknown parameter '%s'. Known parameters are as follows: %s" % (self.name, str(real_vals)))
    return real_vals[self.name]

  def gradient(self, pois : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes gradient of the normalization wrt parameters

     Non-zero gradient is given by the linear coefficients

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         known gradient, in the form { par_name: dN/dpar }
    """
    if not self.name in pois :
        raise KeyError("Invalid single-parameter expression '%s', not formed from a POI. Known POIs are as follows: %s" % (self.name, str(pois.keys())))
    return np.array([ 1 if poi == self.name else 0 for poi in pois ])

  def replace(self, name : str, value : float, reals : dict) -> float :
    """Replaces parameter 'name' by a numberical value

       All instances of the parameter 'name'
       should be replaced by 'value'. This expression should 
       update itself to remove the dependency on 'name', and
       adjust its own numerical value according to 'value'.
       If this makes the expression trivial, the resulting
       numerical value of the expression is returned. Otherwise,
       the expession itself is returned.

      Args:
         name : the name of the parameter to replace
         value : the value to use as replacement
         reals : a dict containing all the reals in the model,
                 to recursively propagate the replace call to
                 sub-expressions.
      Returns:
         self, or just a number if the expression is now trivial
    """
    if name == self.name : return value
    return self

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
        raise KeyError("Cannot evaluate linear combination of the unknowm parameter '%s'. Known parameters are as follows: %s" % (par_name, str(pars_dict)))
      val += self.pars_coeffs[par_name]*pars_dict[par_name]
    return val

  def gradient(self, pois : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes gradient of the normalization wrt parameters

     Non-zero gradient is given by the linear coefficients

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         known gradient, in the form { par_name: dN/dpar }
    """
    val = np.zeros(len(pois))
    for par_name in self.par_coeffs :
      val += self.pars_coeffs[par_name]*reals[par_name].gradient(pois, reals, real_vals)
    return val

  def replace(self, name : str, value : float, reals : dict) -> float :
    """Replaces parameter 'name' by a numberical value

       All instances of the parameter 'name'
       should be replaced by 'value'. This expression should 
       update itself to remove the dependency on 'name', and
       adjust its own numerical value according to 'value'.
       If this makes the expression trivial, the resulting
       numerical value of the expression is returned. Otherwise,
       the expession itself is returned.

      Args:
         name : the name of the parameter to replace
         value : the value to use as replacement
         reals : a dict containing all the reals in the model,
                 to recursively propagate the replace call to
                 sub-expressions.
      Returns:
         self, or just a number if the expression is now trivial
    """
    for par in self.pars_coeffs :
      new_real = reals[par].replace(name, value, reals)
      if new_real != reals[par] : # i.e. it is now a trivial number
        self.nominal_value += self.pars_coeffs[par]*new_real
        del self.pars_coeffs[par]
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

