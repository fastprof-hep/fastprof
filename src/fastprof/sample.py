"""Module containing the building blocks for fastprof models:

  * :class:`Sample` objects, each defining a contribution to a channel from an individual physics process.
  
  Each object contains expected event yields and a normalization factor that is an arbitrary
  function of the model POIs and NPs.
  
  The separation between samples is typically based on the need for separate normalization factors
  (samples with identical scaling factors can be merged without changing the model behavior).
"""

from .base import Serializable
from .norms import Norm
import numpy as np
from abc import abstractmethod
import copy


# -------------------------------------------------------------------------
class Sample(Serializable) :
  """Class representing a model sample

  Samples represent the contribution from a given process to a
  :class:`Channel`.

  It provides the following:

  * An overall normalization term, which is a function of the
    model POIs (see :class:`ModelPOI`) and NPs (see :class:`ModelNP`)

  * An expected event yield for each channel bin

  * The relative variation of the per-bin yields with each model NP
    (see :class:`ModelNP`)

  Attributes:
     name          (str) : the name of the sample
     norm_expr     (str) : a string defining the normalization term
                           as a function of the POIs and NPs
     nominal_norm  (float) : the nominal value of the normalization term,
                             for which the expected yields are given.
                             Normally computed at init-time from model ref_pars
                             (init values of POIs + nominal vals of NPs), or can
                             be specified manually.
     nominal_yield (:class:`np.ndarray`) :
          the nominal per-bin yields, provided as an 1D np.array with
          a size equal to the number of channel bins
     impacts (dict) : the relative variations in the per-bin yields
          corresponding to :math:`n\\sigma` variations
          in the value of each NP; provided as a python dict
          with format { np_name: np_impacts } with the impacts provided
          as a list (over bins) of small dicts { nsigmas : relative_variation }
          for all the known variations.
     save_norm (bool) : True if the sample norm is explicitly specified, as opposed
                        to computed on the fly from nominal parameter values.
  """

  def __init__(self, name : str = '', norm : Norm = None, nominal_norm : float = None,
               nominal_yield : np.ndarray = None, impacts : dict = None) :
    """Create a new Sample object

      Args:
         name: sample name
         norm: object describing the sample normalization
         nominal_norm: value of the normalization factor for which the nominal yields are provided
         nominal_yield: nominal event yields for all channel bins (np. array of size `nbins`)
         impacts: relative change of the nominal event yields for each NP (see class description for format)
    """
    self.name = name
    self.norm = norm
    self.nominal_norm = nominal_norm
    self.nominal_yield = nominal_yield
    self.impacts = impacts if impacts is not None else {}
    self.save_norm = True

  def clone(self) :
    """Return a clone of the object

      Performs a deep-copy of the numpy arrays, but just a shallow copy
      of the normalization object.

      Returns:
        the new clone
    """
    return Sample(self.name, self.norm, self.nominal_norm, np.array(self.nominal_yield), copy.deepcopy(self.impacts))

  def set_np_data(self, nps : list, reals : dict, real_vals : dict, verbosity : int = 0) :
    """Post-initialization update based on model NPs

    The function performs a couple of updates that require passing
    the model NPs, which isn't available at initialization time.
    It is called by the model just before the sample data is loaded,
    in :meth:`fastprof.Model.set_internal_vars`.
    
    The updates are

    * Update the nominal_norm field: this can be set in the markup or constructor,
      but it can be natural to take it as the value obtained from the nominal
      model parameter values. 
    
    * Update the nominal yields : if `None`, we assume there is just one bin,
      and that the content matches the nominal norm (as would be the case if
      the normalization is a raw number of events)
    
    * Update the `self.impacts` array, which stores the impact of all known NPs.
      This is first loaded from markup or specified in the constructor, and should
      list all NPs. However the impact of NPs which enter only the normalization
      can be deduced directly, if the normalization has a simple form. The
      function sets these 'implicit' impacts if they are not present (if already defined,
      they are not overwritten).
    
    Args:
      nps : list of all model NPs
      reals : list of all real value objects in the model
      real_vals : list of the current numerical values of each real.
      verbosity : verbosity level of the output
    """
    if self.nominal_norm is None :
      self.save_norm = False
      try :
        self.nominal_norm = self.norm.value(real_vals)
      except Exception as inst :
        if verbosity > 0 : print("Using normalization = 1 for sample '%s' after encountering exception below :" % self.name)
        print(inst)
        self.nominal_norm = 1
    if self.nominal_yield is None :
      self.nominal_yield = np.array([ self.nominal_norm ])
    for par in nps :
      if par.name in self.impacts : continue
      imp_impact = self.norm.implicit_impact(par, reals, real_vals)
      if imp_impact is None : continue
      self.impacts[par.name] = imp_impact

  def impact(self, par : str) -> np.array :
    """Provide the relative variations of the per-bin event yields for a given NP

      Args:
         par       : name of the NP
      Returns:
         per-bin relative variations (shape : nbins)
    """
    if not par in self.impacts : return np.zeros(len(self.nominal_yield)) # no registered impact: return [0...0]
    #  raise KeyError('No impact defined in sample %s for parameters %s.' % (self.name, par))
    impacts = None
    # Case 1 : the impacts are provided as a list, with one entry per bin. Each list entry is the per-bin impact 
    if isinstance(self.impacts[par], list) :
      try:
        impacts = np.array(self.impacts[par], dtype=float)
      except Exception as inst:
        print('Impact lookup failed for sample %s, parameter %s, variation %+g' % (self.name, par, variation))
        print('Expected a list of variations (one for each bin), instead got')
        print(self.impacts[par])
        print('Leading to the following error:')
        raise(inst)
    # Case 2 : the impacts are provided as a single float, which is common for all the bins
    if isinstance(self.impacts[par], (float, int)) :
      impacts = np.array( [ self.impacts[par] ] * len(self.nominal_yield), dtype=float)
    return impacts

  def normalization(self, real_vals : dict) -> float :
    """Compute the overall normalization factor

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         normalization factor value
    """
    #print('njpb', self.name, self.norm.value(real_vals), self.nominal_norm)
    return self.norm.value(real_vals)/self.nominal_norm

  def yields(self, real_vals : dict) -> np.ndarray :
    """Compute the scaled yields for a given value of the POIs

      Args:
         real_vals : a dictionary of parameter name: value pairs
      Returns:
         the array of normalized event yields
    """
    return self.normalization(real_vals)*self.nominal_yield

  def gradient(self, pois : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the gradient of the sample yields. 
    
       If `norm_only` is true, only the normalization part is
       returned -- i.e. this is not multiplied into the 
       expected yields. If `relative` is true, then the result
       is divided by the normalization for the same poi values.

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         normalization factor value
    """
    norm_grad = self.norm.gradient(pois, reals, real_vals)
    if norm_grad is None : return None
    return self.nominal_yield[:,None]*norm_grad/self.nominal_norm

  def hessian(self, pois : dict, reals : dict, real_vals : dict) -> np.array :
    """Compute the Hessian of the sample yields

       If `norm_only` is true, only the normalization part is
       returned -- i.e. this is not multiplied into the 
       expected yields. If `relative` is true, then the result
       is divided by the normalization for the same poi values.

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         normalization factor value
    """
    norm_hess = self.norm.hessian(pois, reals, real_vals)
    if norm_hess is None : return None
    return self.nominal_yield[:, None, None]*norm_hess/self.nominal_norm

  def __str__(self) -> str :
    """Provide a description string

      Returns:
        The object description
    """
    return 'Sample ' + self.string_repr(verbosity = 1)

  def string_repr(self, verbosity : int = 1, pre_indent : str = '', indent : str = '   ') :
    """Return a string representation of the object

      Same as __str__ but with options

      Args:
        verbosity : verbosity of the output
        pre_indent: number of indentation spaces to add to all lines
        indent    : number of indentation spaces to add to fields of this object

      Returns:
        the object description
    """
    rep = self.name
    if verbosity == 1 :
      rep += ', norm = %s (nominal norm = %s), nominal yield = %g' % (str(self.norm), str(self.nominal_norm), np.sum(self.nominal_yield))
    if verbosity >= 2 :
      rep += ', norm = %s (nominal norm = %s), nominal yield : ' % (str(self.norm), str(self.nominal_norm))
      for i, y in enumerate(self.nominal_yield) :
        rep += '\n%s o bin %2d : %g' % (pre_indent, i, y)
    return rep

  def load_dict(self, sdict : dict) -> 'Sample' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    self.name = self.load_field('name', sdict, '', str)
    self.norm = Norm.instantiate(sdict)
    self.nominal_norm = self.load_field('nominal_norm', sdict, None, [float, int])
    self.nominal_yield = self.load_field('nominal_yield', sdict, None, [np.ndarray, list, float, int])
    if isinstance(self.nominal_yield, (float, int)) :
      self.nominal_yield = np.array([ self.nominal_yield ], dtype=float)
    if isinstance(self.nominal_yield, list) :
      self.nominal_yield = np.array(self.nominal_yield, dtype=float)
    self.impacts = self.load_field('impacts', sdict, {}, dict)
    return self

  def fill_dict(self, sdict : dict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    sdict['name'] = self.name
    self.norm.fill_dict(sdict)
    if self.save_norm : sdict['nominal_norm']  = self.unnumpy(self.nominal_norm)
    sdict['nominal_yield'] = self.unnumpy(self.nominal_yield)
    sdict['impacts']       = self.unnumpy(self.impacts)

