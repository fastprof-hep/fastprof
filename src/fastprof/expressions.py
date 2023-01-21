"""Module containing the expression classes

Expressions are functions of model POIs that are used to define
the normalization factors of the model samples. These factors can be
single POIs, or model complex expressions. In each case, the expression
must be evaluated and als define closed-form gradients and hessians that
can be used to help with minimization and error propagation.

The following classes are defined:

  * :class:`Expression` : the base class defining the overall structure. It derives from
    :class:`fastprof.base.Serializable` in order to benefit from markup I/O functionality
    
  * :class:`Number` : an expression defined as a single real number.
  
  * :class:`SingleParameter` : an expression defined as a single POI.
  
  * :class:`LinearCombination` : the linear combination of  multiple
    POIs or expressions.

  * :class:`ProductRatio` : the product and ratio of  multiple POIs or
    expressions, in the form `(p1*p2*p3*...)/(q1*q2*q3*...)`.

  * :class:`Exponential` : the the exponential of a POI or expression.

  * :class:`Formula` : an expression defined by a generic formula, currently
    not fully implemented.
    
The evaluation functions make use of 3 main arguments: a list of parameters (`pars`), a
list of sub-expressions (`reals`) and a dict of { name : value } for all 
the known expressions, including parameters (`real_vals`). Evaluation is recursive, and
values are entered in `real_vals` as they are computed. The `pars` and `reals` arguments
are expected as { name: object } dicts since this is how it naturally comes from the
model, but only the keys are used for now.
"""

from .base import Serializable
import numpy as np
from abc import abstractmethod


# -------------------------------------------------------------------------
class Expression(Serializable) :
  """Class representing an expression involving POIs

  Attributes:
     name (str) : the name of the expression

    Args:
        name : the expression name
  """
  def __init__(self, name : str = '') :
    """Create a new Expression object
    """
    self.name = name
    super().__init__()

  @abstractmethod
  def value(self, pars : dict) -> float :
    """Compute the expression value
    
      Returns the expression value as a single float 

      Args:
         pars : the expression parameters as a dictionary of { name: object } pairs
      Returns:
         the expression value
    """
    pass

  def gradient(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the gradient of the expression wrt parameters

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
      The order of the gradients is the same as that of the `pars` arg.

      Args:
        pars : expression parameters as a dictionary of { name: object } pairs
        reals: sub-expressions as a dictionary of { name: object } pairs
        real_vals: values of all reals and pars, as { name: value} pairs
      Returns:
        Hessian wrt parameter pairs
    """
    return None

  def replace(self, name : str, value : float, reals : dict) -> float :
    """Replace parameter 'name' by a numberical value

       Replaces all instances of the parameter 'name'
       by 'value', as needed for instance if a parameter is set 
       constant to this value. This expression should 
       update itself to remove the dependency on 'name', and
       adjust its own numerical value according to 'value'.

      Args:
         name : the name of the parameter to replace
         value : the value to use as replacement
         reals : a dict of all sub-expressions in the model,
                 to recursively propagate the replace call to
                 sub-expressions.
      Returns:
         the new expression
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

  def fill_dict(self, sdict : dict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    sdict['name'] = self.name

  @classmethod
  def instantiate(cls, sdict, load_data : bool = True) :
    """Instantiate an Expression object from one of the possibilities below.

      Class method used when loading the expressions from markup, to
      instantiate an object from type strings specified in the markup.

      Args:
         cls: this class
         sdict: dictionary containing markup data
         load_data: if `True`, also initialize the expression from markup
    """
    if not 'type' in sdict or sdict['type'] == Formula.type_str :
      expression = Formula()
    elif sdict['type'] == Number.type_str :
      expression = Number()
    elif sdict['type'] == SingleParameter.type_str :
      expression = SingleParameter()
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
  """Expression class representing a (fixed) numerical value
  
  Attributes:
    val (float) : the numerical value
  """

  type_str = 'number'

  def __init__(self, par_name : str = '', value : float = 0) :
    """Initialize a new object
    
    Args:
      par_name: the object name
      val : the numerical value to assign
    """
    super().__init__(par_name)
    self.val = value

  def value(self, real_vals : dict) -> float :
    """Compute the expression value
    
      Returns the expression value, in this case the stored float.

      Args:
         pars : the expression parameters as a dictionary of { name: object } pairs
      Returns:
         the expression value
    """
    return self.val

  def gradient(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Computes gradient of the normalization wrt parameters

      Returns zeros in this case, since the expression is constant.

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         known gradient, in the form { par_name: dN/dpar }
    """
    return np.zeros(len(pars))

  def hessian(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the Hessian of the expression wrt parameters

      Returns the Hessian of the expression as a 2D array of size len(pars) x len(pars).
      The order of the gradients is the same as that of the `pars` arg.

      Returns zeros in this case, since the expression is constant.

      Args:
        pars : expression parameters as a dictionary of { name: object } pairs
        reals: sub-expressions as a dictionary of { name: object } pairs
        real_vals: values of all reals and pars, as { name: value} pairs
      Returns:
        Hessian wrt parameter pairs
    """
    return np.zeros((len(pars), len(pars)))

  def replace(self, name : str, value : float, reals : dict) -> float :
    """Replace parameter 'name' by a numberical value

       Replaces all instances of the parameter 'name'
       by 'value', as needed for instance if a parameter is set 
       constant to this value. This expression should 
       update itself to remove the dependency on 'name', and
       adjust its own numerical value according to 'value'.
       
       Here just return itself no matter what, since there are
       no parameters.

      Args:
         name : the name of the parameter to replace
         value : the value to use as replacement
         reals : a dict of all sub-expressions in the model,
                 to recursively propagate the replace call to
                 sub-expressions.
      Returns:
         self
    """
    return self

  def load_dict(self, sdict : dict) -> 'Number' :
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

  def fill_dict(self, sdict : dict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    super().fill_dict(sdict)
    sdict['type'] = Number.type_str
    sdict['value'] = self.val

  def __str__(self) -> str :
    """Provide a description string

      Returns:
        the object description
    """
    return '%s=%g' % (self.name, self.val)


# -------------------------------------------------------------------------
class SingleParameter(Expression) :
  """Class representing a single parameter
  """

  type_str = 'single_parameter'

  def __init__(self, par_name : str = '') :
    """Create a new SingleParameter object
    
    """
    super().__init__(par_name)

  def value(self, real_vals : dict) -> float :
    """Compute the expression value
    
      Returns the expression value as a single float, here
      just the parameter value.

      Args:
         pars : the expression parameters as a dictionary of { name: object } pairs
      Returns:
         the expression value
    """
    if not self.name in real_vals :
        raise KeyError("Cannot evaluate unknown parameter '%s'. Known parameters are as follows: %s" % (self.name, str(real_vals)))
    return real_vals[self.name]

  def gradient(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the gradient of the expression wrt parameters

      Returns the gradient of the expression as a 1D array of size len(pars).
      The order of the gradients is the same as that of the `pars` arg.
      
      Here the gradients are just 1 for this parameter and
      0 otherwise.
    
      Args:
        pars : expression parameters as a dictionary of { name: object } pairs
        reals: sub-expressions as a dictionary of { name: object } pairs
        real_vals: values of all reals and pars, as { name: value} pairs
      Returns:
        gradients wrt parameters
    """
    if not self.name in pars : return np.zeros(len(pars))
    return np.array([ 1 if par == self.name else 0 for par in pars ])

  def hessian(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the Hessian of the expression wrt parameters

      Returns the Hessian of the expression as a 2D array of size len(pars) x len(pars).
      The order of the gradients is the same as that of the `pars` arg.
      
      Here return just zeros, since second derivatives of a single parameter are
      always 0.

      Args:
        pars : expression parameters as a dictionary of { name: object } pairs
        reals: sub-expressions as a dictionary of { name: object } pairs
        real_vals: values of all reals and pars, as { name: value} pairs
      Returns:
        Hessian wrt parameter pairs
    """
    return np.zeros((len(pars), len(pars)))

  def replace(self, name : str, value : float, reals : dict) -> float :
    """Replaces parameter 'name' by a numberical value

       All instances of the parameter 'name'
       should be replaced by 'value'. This expression should 
       update itself to remove the dependency on 'name', and
       adjust its own numerical value according to 'value'.

      Args:
         name : the name of the parameter to replace
         value : the value to use as replacement
         reals : a dict containing all the reals in the model,
                 to recursively propagate the replace call to
                 sub-expressions.
      Returns:
         self
    """
    if name == self.name :
      if value is None : raise KeyError("Cannot remove POI '%s' since it must be removed from expression '%s' and no replacement value was provided." % (name, self.name))
      return Number(self.name, value)
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

  def fill_dict(self, sdict : dict) :
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

  The linear combination is represented as
  (nominal value) + coeff1*par1 + coeff2*par2 + ...

  Attributes:
    nominal_value (float) : the nominal term in the expression
    coeffs (dict) : dict with the format { par_name : par_coeff } 
                    mapping parameter names to their coefficients
                    in the linear combination
  """

  type_str = 'linear_combination'

  def __init__(self, name : str = '', nominal_value : float = 0, coeffs : dict = {}) :
    """Create a new LinearCombination object
    
    Args:
      nominal_value: the nominal term added to the linear combination
      coeffs : dict specifying the parameters and coefficients of the 
               linear combination, with entries of the form expr: coeff 
    """
    super().__init__(name)
    self.nominal_value = nominal_value
    self.coeffs = coeffs

  def value(self, real_vals : dict) -> float :
    """Compute the gradient of the expression wrt parameters

      Returns the gradient of the expression as a 1D array of size len(pars).
      The order of the gradients is the same as that of the `pars` arg.
      
      For a linear combination, the gradients are just given by the
      linear coefficients.
    
      Args:
        real_vals: values of all reals and pars, as { name: value} pairs
      Returns:
        gradients wrt parameters
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
        pars : expression parameters as a dictionary of { name: object } pairs
        reals: sub-expressions as a dictionary of { name: object } pairs
        real_vals: values of all reals and pars, as { name: value} pairs
      Returns:
         known gradient, in the form { par_name: dN/dpar }
    """
    val = np.zeros(len(pars))
    for par_name in self.coeffs :
      val += self.coeffs[par_name]*reals[par_name].gradient(pars, reals, real_vals)
    return val

  def hessian(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the Hessian of the expression wrt parameters

      Returns the Hessian of the expression as a 2D array of size len(pars) x len(pars).
      The order of the gradients is the same as that of the `pars` arg.

      Args:
        pars : expression parameters as a dictionary of { name: object } pairs
        reals: sub-expressions as a dictionary of { name: object } pairs
        real_vals: values of all reals and pars, as { name: value} pairs
      Returns:
        Hessian wrt parameter pairs
    """
    hess = np.zeros((len(pars), len(pars)))
    for par_name in self.coeffs :
      hess += self.coeffs[par_name]*reals[par_name].hessian(pars, reals, real_vals)
    return hess


  def replace(self, name : str, value : float, reals : dict) -> 'LinearCombination' :
    """Replace parameter 'name' by a numerical value

       Replaces all instances of the parameter 'name'
       by 'value', as needed for instance if a parameter is set 
       constant to this value. This expression should 
       update itself to remove the dependency on 'name', and
       adjust its own numerical value according to 'value'.

      Args:
         name : the name of the parameter to replace
         value : the value to use as replacement
         reals : a dict of all sub-expressions in the model,
                 to recursively propagate the replace call to
                 sub-expressions.
      Returns:
         self
    """
    to_remove = []
    for expr in self.coeffs :
      new_expr = reals[expr].replace(name, value, reals)
      if new_expr != reals[expr] : # i.e. it is now a trivial number
        self.nominal_value += self.coeffs[expr]*new_expr.val
        to_remove.append(expr)
    for expr in to_remove : del self.coeffs[expr]
    if len(self.coeffs) > 0 : return self
    return Number(self.name, self.nominal_value)

  def load_dict(self, sdict : dict) -> 'LinearCombination' :
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

  def fill_dict(self, sdict : dict) :
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

  def numerator_value(self, real_vals : dict) -> float :
    for expr in self.numerator :
      if not expr in real_vals :
        raise KeyError("Cannot evaluate product of the unknown parameter '%s'. Known parameters are as follows: %s" % (expr, str(real_vals)))
    return np.prod([ real_vals[expr] for expr in self.numerator ])

  def denominator_value(self, real_vals : dict) -> float :
    for expr in self.denominator :
      if not expr in real_vals :
        raise KeyError("Cannot evaluate ratio of the unknown parameter '%s'. Known parameters are as follows: %s" % (expr, str(real_vals)))
    return np.prod([ real_vals[expr] for expr in self.denominator ])

  def value(self, real_vals : dict) -> float :
    """Compute the expression value
    
      Returns the expression value as a single float 

      Args:
         pars : the expression parameters as a dictionary of { name: object } pairs
      Returns:
         the expression value
    """
    try :
      val = self.prefactor*self.numerator_value(real_vals)/self.denominator_value(real_vals)
    except ZeroDivisionError :
      raise ValueError("Attempting to divide by 0 in the denominator of '%s'." % self.name)
    return val

  # gradient of a product -- remove each term in turn and replace by its gradient
  # allows to never divide by terms, which may be zero, so useful for the numerator.
  def prod_gradient(self, pars : dict, reals : dict, real_vals : dict, product : list) -> np.array :
    if len(product) == 0 : return np.zeros(len(pars))
    expr_vals = np.array( [ real_vals[expr] for expr in product ] )
    expr_mat = np.tile(expr_vals, (len(expr_vals), 1))
    np.fill_diagonal(expr_mat, 1)
    grad_vals = np.stack([ reals[expr].gradient(pars, reals, real_vals) for expr in product ], axis=1)
    return np.sum(np.prod(expr_mat, axis=1)*grad_vals, axis=1)

  # normalized gradient -- sum of T'/T for each term, i.e. the above divided by the product value
  def norm_gradient(self, pars : dict, reals : dict, real_vals : dict, product : list) -> np.array :
    if len(product) == 0 : return np.zeros(len(pars))
    expr_vals = np.array( [ real_vals[expr] for expr in product ] )
    grad_vals = np.stack([ reals[expr].gradient(pars, reals, real_vals) for expr in product ], axis=1)
    return np.sum(grad_vals/expr_vals, axis=1)

  def gradient(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the gradient of the expression wrt parameters

      Returns the gradient of the expression as a 1D array of size len(pars).
      The order of the gradients is the same as that of the `pars` arg.
    
      Args:
        pars : expression parameters as a dictionary of { name: object } pairs
        reals: sub-expressions as a dictionary of { name: object } pairs
        real_vals: values of all reals and pars, as { name: value} pairs
      Returns:
        gradients wrt parameters
    """
    grad_num = self.prod_gradient(pars, reals, real_vals, self.numerator)
    grad_den = self.norm_gradient(pars, reals, real_vals, self.denominator)
    return self.prefactor*grad_num/self.denominator_value(real_vals) - self.value(real_vals)*grad_den

  def norm_hessian(self, pars : dict, reals : dict, real_vals : dict, product : list) -> np.array :
    if len(product) == 0 : return np.zeros((len(pars), len(pars)))
    expr_vals = np.array( [ real_vals[expr] for expr in product ] )
    grad_vals = np.stack([ reals[expr].gradient(pars, reals, real_vals) for expr in product ], axis=1)
    hess_vals = np.stack( [ reals[expr].hessian(pars, reals, real_vals) for expr in product ], axis=2 )
    return np.sum((hess_vals - grad_vals[None,:,:]*grad_vals[:,None,:]/expr_vals)/expr_vals, axis=2)

  def hessian(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the Hessian of the expression wrt parameters

      Returns the Hessian of the expression as a 2D array of size len(pars) x len(pars).
      The order of the gradients is the same as that of the `pars` arg.

      Args:
        pars : expression parameters as a dictionary of { name: object } pairs
        reals: sub-expressions as a dictionary of { name: object } pairs
        real_vals: values of all reals and pars, as { name: value} pairs
      Returns:
        Hessian wrt parameter pairs
    """
    hess_num = self.norm_hessian (pars, reals, real_vals, self.numerator  )
    hess_den = self.norm_hessian (pars, reals, real_vals, self.denominator)
    grad_num = self.norm_gradient(pars, reals, real_vals, self.numerator  )
    grad_den = self.norm_gradient(pars, reals, real_vals, self.denominator)
    grad_dif = grad_num - grad_den
    return self.value(real_vals)*(hess_num - hess_den + grad_dif[:,None]*grad_dif[None,:])

  def replace(self, name : str, value : float, reals : dict) -> 'ProductRatio' :
    """Replace parameter 'name' by a numberical value

       Replaces all instances of the parameter 'name'
       by 'value', as needed for instance if a parameter is set 
       constant to this value. This expression should 
       update itself to remove the dependency on 'name', and
       adjust its own numerical value according to 'value'.

      Args:
         name : the name of the parameter to replace
         value : the value to use as replacement
         reals : a dict of all sub-expressions in the model,
                 to recursively propagate the replace call to
                 sub-expressions.
      Returns:
         self
    """
    to_remove = []
    for expr in self.numerator :
      new_expr = reals[expr].replace(name, value, reals)
      if new_expr != reals[expr] : # i.e. it is now a trivial number
        self.prefactor *= new_expr.val
        to_remove.append(expr)
    for expr in to_remove : del self.numerator[expr]
    to_remove = []
    for expr in self.denominator :
      new_expr = reals[expr].replace(name, value, reals)
      if new_expr != reals[expr] : # i.e. it is now a trivial number
        if new_expr.val == 0 :
          raise ValueError("Attempting to divide by '%s' == 0 when replacing '%s'=%g in '%s'." % (expr, name, value, self.name)) 
        self.prefactor /= new_expr.val
        to_remove.append(expr)
    for expr in to_remove : del self.denominator[expr]
    if len(self.numerator) + len(self.denominator) > 0 : return self
    return Number(self.name, self.prefactor)

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

  def fill_dict(self, sdict : dict) :
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
    num = '*'.join([ str(self.prefactor) ] + self.numerator)
    den = '*'.join(self.denominator)
    if den == '' : return num
    if num == '' : return '1/(%s)' % den
    return '(%s)/(%s)' % (num, den)



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
    """Compute the expression value
    
      Returns the expression value as a single float 

      Args:
         pars : the expression parameters as a dictionary of { name: object } pairs
      Returns:
         the expression value
    """
    if not self.exponent in real_vals :
      raise KeyError("Cannot evaluate exponential of the unknown expression '%s'. Known expressions are as follows: %s" % (self.exponent, str(real_vals.keys())))
    return np.exp(real_vals[self.exponent])

  def gradient(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the gradient of the expression wrt parameters

      Returns the gradient of the expression as a 1D array of size len(pars).
      The order of the gradients is the same as that of the `pars` arg.
      
      For a linear combination, the gradients are just given by the
      linear coefficients.
    
      Args:
        pars : expression parameters as a dictionary of { name: object } pairs
        reals: sub-expressions as a dictionary of { name: object } pairs
        real_vals: values of all reals and pars, as { name: value} pairs
      Returns:
        gradients wrt parameters
    """
    if not self.exponent in reals :
      raise KeyError("Cannot compute gradient of the exponential of the unknown expression '%s'. Known expressions are as follows: %s" % (self.exponent, str(reals.keys())))
    return self.value(real_vals)*reals[self.exponent].gradient(pars, reals, real_vals)

  def hessian(self, pars : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the Hessian of the expression wrt parameters

      Returns the Hessian of the expression as a 2D array of size len(pars) x len(pars).
      The order of the gradients is the same as that of the `pars` arg.

      Args:
        pars : expression parameters as a dictionary of { name: object } pairs
        reals: sub-expressions as a dictionary of { name: object } pairs
        real_vals: values of all reals and pars, as { name: value} pairs
      Returns:
        Hessian wrt parameter pairs
    """
    if not self.exponent in reals :
      raise KeyError("Cannot compute Hessian of the exponential of the unknown expression '%s'. Known expressions are as follows: %s" % (self.exponent, str(reals.keys())))
    grad = reals[self.exponent].gradient(pars, reals, real_vals)
    return self.value(real_vals)*(reals[self.exponent].hessian(pars, reals, real_vals) + np.outer(grad, grad))

  def replace(self, name : str, value : float, reals : dict) -> 'Exponential' :
    """Replace parameter 'name' by a numberical value

       Replaces all instances of the parameter 'name'
       by 'value', as needed for instance if a parameter is set 
       constant to this value. This expression should 
       update itself to remove the dependency on 'name', and
       adjust its own numerical value according to 'value'.

      Args:
         name : the name of the parameter to replace
         value : the value to use as replacement
         reals : a dict of all sub-expressions in the model,
                 to recursively propagate the replace call to
                 sub-expressions.
      Returns:
         self
    """
    new_exp = reals[self.exponent].replace(name, value, reals)
    if new_exp != reals[self.exponent] : # the exponent is now a simple number
      return Number(self.name, np.exp(new_exp.val))
    else :
      return self

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

  def fill_dict(self, sdict : dict) :
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
    """Compute the expression value
    
      Returns the expression value as a single float 

      Args:
         pars : the expression parameters as a dictionary of { name: object } pairs
      Returns:
         the expression value
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

  def fill_dict(self, sdict : dict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    super().fill_dict(sdict)
    sdict['type'] = Formula.type_str
    sdict['formula'] = self.formula

  def __str__(self) -> str :
    """Provide a description string

      Returns:
        The object description
    """
    return 'formula[%s]' % self.formula



