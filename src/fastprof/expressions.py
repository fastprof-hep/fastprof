from .base import Serializable
import numpy as np
from abc import abstractmethod


# -------------------------------------------------------------------------
class Expression(Serializable) :
  """Class representing an expression involving POIs

  Attributes:
     name (str) : the name of the expression
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

  def gradient(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes the gradient of the expression wrt parameters

     Non-zero gradient is given by the linear coefficients

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         known gradients, in the form { par_name: dN/dpar }
    """
    return None

  def hessian(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes the Hessian of the expression wrt parameters

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
    raise KeyError("Cannot replace value of parameter '%s' in expression '%s'." % (name, self.name))

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
    elif sdict['type'] == Number.type_str :
      expression = Number()
    elif sdict['type'] == LinearCombination.type_str :
      expression = LinearCombination()
    elif sdict['type'] == ProductRatio.type_str :
      expression = ProductRatio()
    elif sdict['type'] == Exponential.type_str :
      expression = Exponential()
    else :
      raise ValueError("Unsupported expression type '%s'" % sdict['type'])
    if load_data : expression.load_dict(sdict)
    return expression

# -------------------------------------------------------------------------
class Number(Expression) :
  """Class representing a single number
  """

  type_str = 'number'

  def __init__(self, par_name : str = '', value : float = 0) :
    """Create a new LinearCombination object

    #"""
    super().__init__(par_name)
    self.val = value

  def value(self, real_vals : dict) -> float :
    """Computes the expression value

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         expression value
    """
    return self.val

  def gradient(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes gradient of the normalization wrt parameters

     Non-zero gradient is given by the linear coefficients

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         known gradient, in the form { par_name: dN/dpar }
    """
    return np.zeros(len(pars))

  def hessian(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes Hessian of the normalization wrt parameters

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
        Hessian matrix
    """
    return np.zeros((len(pars), len(pars)))

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
    return self

  def load_dict(self, sdict) -> 'Number' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    super().load_dict(sdict)
    if 'type' in sdict and sdict['type'] != Number.type_str :
      raise ValueError("Cannot load normalization data defined for type '%s' into object of type '%s'" % (sdict['type'], Number.type_str))
    self.val = self.load_field('value', sdict, 0, [int, float])
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    super().fill_dict(sdict)
    sdict['type'] = Number.type_str
    sdict['value'] = self.val

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    return '%s=%g' % (self.name, self.val)


# -------------------------------------------------------------------------
class SingleParameter(Expression) :
  """Class representing a single parameter
  """

  type_str = 'single_parameter'

  def __init__(self, par_name : str = '') :
    """Create a new LinearCombination object
    
    #"""
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

  def gradient(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes gradient of the normalization wrt parameters

     Non-zero gradient is given by the linear coefficients

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         known gradient, in the form { par_name: dN/dpar }
    """
    if not self.name in pars : return np.zeros(len(pars))
    return np.array([ 1 if par == self.name else 0 for par in pars ])

  def hessian(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes Hessian of the normalization wrt parameters

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
        Hessian matrix
    """
    return np.zeros((len(pars), len(pars)))

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
    if name == self.name :
      if value is None : raise KeyError("Cannot remove POI '%s' since it must be removed from expression '%s' and no replacement value was provided." % (name, self.name))
      return value
    return self

  def load_dict(self, sdict) -> 'SingleParameter' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    super().load_dict(sdict)
    if 'type' in sdict and sdict['type'] != SingleParameter.type_str :
      raise ValueError("Cannot load normalization data defined for type '%s' into object of type '%s'" % (sdict['type'], SingleParameter.type_str))
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    super().fill_dict(sdict)
    sdict['type'] = SingleParameter.type_str

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    return self.name


# -------------------------------------------------------------------------
class LinearCombination(Expression) :
  """Class representing a linear combination of parameters

  Attributes:
     coeffs (dict) : dict with the format { par_name : par_coeff } 
                          mapping parameter names to their coefficients
                          in the linear combination
  """

  type_str = 'linear_combination'

  def __init__(self, name : str = '', nominal_value : float = 0, coeffs : dict = {}) :
    """Create a new LinearCombination object
    
    Args:
      coeffs : dict specifying the parameters and coefficients of the 
               linear combination, with entries of the form expr: coeff 
    """
    super().__init__(name)
    self.nominal_value = nominal_value
    self.coeffs = coeffs

  def value(self, real_vals : dict) -> float :
    """Computes the expression value

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         expression value
    """
    val = self.nominal_value
    for expr in self.coeffs :
      if not expr in real_vals :
        raise KeyError("Cannot evaluate linear combination of the unknown parameter '%s'. Known parameters are as follows: %s" % (expr, str(real_vals)))
      val += self.coeffs[expr]*real_vals[expr]
    return val

  def gradient(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes gradient of the normalization wrt parameters

     Non-zero gradient is given by the linear coefficients

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         known gradient, in the form { par_name: dN/dpar }
    """
    val = np.zeros(len(pars))
    for par_name in self.coeffs :
      val += self.coeffs[par_name]*reals[par_name].gradient(pars, reals, real_vals)
    return val

  def hessian(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes Hessian of the normalization wrt parameters

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
        Hessian matrix
    """
    return np.zeros((len(pars), len(pars)))

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
    to_remove = []
    for expr in self.coeffs :
      new_expr = reals[expr].replace(name, value, reals)
      if new_expr != reals[expr] : # i.e. it is now a trivial number
        self.nominal_value += self.coeffs[expr]*new_expr
        to_remove.append(expr)
    for expr in to_remove : del self.coeffs[expr]
    if len(self.coeffs) > 0 : return self
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
      raise ValueError("Cannot load expression data defined for type '%s' into object of type '%s'" % (sdict['type'], LinearCombination.type_str))
    self.nominal_value = self.load_field('nominal_value', sdict, 0,  [float, int])
    self.coeffs = self.load_field('coeffs', sdict, {}, dict)
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    super().fill_dict(sdict)
    sdict['type'] = LinearCombination.type_str
    sdict['nominal_value'] = self.nominal_value
    sdict['coeffs'] = self.coeffs

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    return ''.join([ str(self.nominal_value) ] + [ '%+g*%s' % (self.coeffs[par_name], par_name) for par_name in self.coeffs ])


# -------------------------------------------------------------------------
class ProductRatio(Expression) :
  """Class representing a linear combination of parameters

  Attributes:
     numerator (list) : list of expressions in the numerator
     denominator (list) : list of expressions in the denominator
  """

  type_str = 'product_ratio'

  def __init__(self, name : str = '', prefactor : float = 1, numerator : list = [], denominator : list = []) :
    """Create a new LinearCombination object
    
    Args:
      numerator   : list of expressions in the numerator
      denominator : list of expressions in the denominator
    """
    super().__init__(name)
    self.prefactor = prefactor
    self.numerator = numerator
    self.denominator = denominator

  def value(self, real_vals : dict) -> float :
    """Computes the expression value

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         expression value
    """
    val = self.prefactor
    for expr in self.numerator :
      if not expr in real_vals :
        raise KeyError("Cannot evaluate product of the unknown parameter '%s'. Known parameters are as follows: %s" % (expr, str(pars_dict)))
      val *= real_vals[expr]
    for expr in self.denominator :
      if not expr in real_vals :
        raise KeyError("Cannot evaluate ratio of the unknown parameter '%s'. Known parameters are as follows: %s" % (expr, str(pars_dict)))
      expr_val = real_vals[expr]
      if expr_val == 0 : raise ValueError("Attempting to divide by '%s' == 0 when computing the value of '%s'." % (expr, self.name))
      val /= expr_val
    return val

  def prod_relative_gradient(self, pars : dict, reals : dict, real_vals : dict, product : list, raise_div_by_zero : bool= False) -> np.array :
    val = np.zeros(len(pars))
    for expr in product :
      expr_val = real_vals[expr]
      if expr_val == 0 : 
        if raise_div_by_zero :
          if expr_val == 0 : raise ZeroDivisionError("Division by '%s' == 0." % expr)
        else :
          continue
      val += reals[expr].gradient(pars, reals, real_vals)/expr_val
    return val
    

  def gradient(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes gradient of the expression wrt parameters

     Non-zero gradient is given by the linear coefficients

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         known gradient, in the form { par_name: dN/dpar }
    """
    rel_grad_nom = self.prod_relative_gradient(pars, reals, real_vals, self.numerator  , raise_div_by_zero=False)
    rel_grad_den = self.prod_relative_gradient(pars, reals, real_vals, self.denominator, raise_div_by_zero=True)
    return self.value(real_vals)*(rel_grad_nom - rel_grad_den)
    
  def prod_relative_hessian(self, pars : dict, reals : dict, real_vals : dict, product : list, raise_div_by_zero : bool= False) -> np.array :
    val = np.zeros((len(pars), len(pars)))
    for expr in product :
      expr_val = real_vals[expr]
      if expr_val == 0 :
        if raise_div_by_zero :
          raise ZeroDivisionError("Division by '%s' == 0." % expr)
        else :
          continue
      rel_hess = reals[expr].hessian (pars, reals, real_vals)/expr_val
      rel_grad = reals[expr].gradient(pars, reals, real_vals)/expr_val
      val += rel_hess - rel_grad[:,None]*rel_grad[None,:]
    return val

  def hessian(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes Hessian of the expression wrt parameters

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
        Hessian matrix
    """
    rel_hess_nom = self.prod_relative_hessian (pars, reals, real_vals, self.numerator  , raise_div_by_zero=False)
    rel_hess_den = self.prod_relative_hessian (pars, reals, real_vals, self.denominator, raise_div_by_zero=True)
    rel_grad_nom = self.prod_relative_gradient(pars, reals, real_vals, self.numerator  , raise_div_by_zero=False)
    rel_grad_den = self.prod_relative_gradient(pars, reals, real_vals, self.denominator, raise_div_by_zero=True)
    rel_grad_ratio = rel_grad_nom - rel_grad_den
    return self.value(real_vals)*(rel_hess_nom - rel_hess_den + rel_grad_ratio[:,None]*rel_grad_ratio[None,:])

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
    to_remove = []
    for expr in self.numerator :
      new_expr = reals[expr].replace(name, value, reals)
      if new_real != reals[expr] : # i.e. it is now a trivial number
        self.prefactor *= new_real
        to_remove.append(expr)
    for expr in to_remove : del self.numerator[expr]
    to_remove = []
    for expr in self.denominator :
      new_expr = reals[expr].replace(name, value, reals)
      if new_real != reals[expr] : # i.e. it is now a trivial number
        if expr_val == 0 : raise ValueError("Attempting to divide by '%s' == 0 when replacing '%s'=%g in '%s'." % (expr, name, value, self.name)) 
        self.prefactor /= new_real
        to_remove.append(expr)
    for expr in to_remove : del self.denominator[expr]
    if len(self.numerator) + len(self.denominator) > 0 : return self
    return self.prefactor

  def load_dict(self, sdict) -> 'ProductRatio' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    super().load_dict(sdict)
    if 'type' in sdict and sdict['type'] != ProductRatio.type_str :
      raise ValueError("Cannot load normalization data defined for type '%s' into object of type '%s'" % (sdict['type'], ProductRatio.type_str))
    self.prefactor   = self.load_field('prefactor'  , sdict,  1,  [int, float])
    self.numerator   = self.load_field('numerator'  , sdict, [],  list)
    self.denominator = self.load_field('denominator', sdict, [],  list)
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    super().fill_dict(sdict)
    sdict['type'] = ProductRatio.type_str
    sdict['prefactor'] = self.prefactor
    sdict['numerator'] = self.numerator
    sdict['denominator'] = self.denominator

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    return '*'.join([ str(self.prefactor) ] + self.numerator) + '(' +  '*'.join(self.denominator) + ')'


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


# -------------------------------------------------------------------------
class Exponential(Expression) :
  """Class representing a linear combination of parameters

  Attributes:
     coeffs (dict) : dict with the format { par_name : par_coeff }
                          mapping parameter names to their coefficients
                          in the linear combination
  """

  type_str = 'exp'

  def __init__(self, name : str = '', exponent : str = '') :
    """Create a new Exponential object

    Args:
      exponent : name of the expression specifying the exponent
    """
    super().__init__(name)
    self.exponent = exponent

  def value(self, real_vals : dict) -> float :
    """Computes the expression value

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         expression value
    """
    if not self.exponent in real_vals :
      raise KeyError("Cannot evaluate exponential of the unknown expression '%s'. Known expressions are as follows: %s" % (self.exponent, str(real_vals.keys())))
    return np.exp(real_vals[self.exponent])

  def gradient(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes gradient of the normalization wrt parameters

     Non-zero gradient is given by the linear coefficients

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         known gradient, in the form { par_name: dN/dpar }
    """
    if not self.exponent in reals :
      raise KeyError("Cannot compute gradient of the exponential of the unknown expression '%s'. Known expressions are as follows: %s" % (self.exponent, str(reals.keys())))
    return self.value(real_vals)*reals[self.exponent].gradient(pars, reals, real_vals)

  def hessian(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes Hessian of the normalization wrt parameters

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
        Hessian matrix
    """
    if not self.exponent in reals :
      raise KeyError("Cannot compute Hessian of the exponential of the unknown expression '%s'. Known expressions are as follows: %s" % (self.exponent, str(reals.keys())))
    grad = reals[self.exponent].gradient(pars, reals, real_vals)
    return self.value(real_vals)*(reals[self.exponent].hessian(pars, reals, real_vals) + np.outer(grad, grad))

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
    return reals[self.exponent].replace(name, value, reals)

  def load_dict(self, sdict) -> 'Exponential' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    super().load_dict(sdict)
    if 'type' in sdict and sdict['type'] != Exponential.type_str :
      raise ValueError("Cannot load expression data defined for type '%s' into object of type '%s'" % (sdict['type'], Exponential.type_str))
    self.exponent = self.load_field('exponent', sdict, '', str)
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    super().fill_dict(sdict)
    sdict['type'] = Exponential.type_str
    sdict['exponent'] = self.exponent

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    return 'exp(%s)' % self.exponent
