"""Module containing the building blocks for fastprof models:

  * :class:`Sample` objects, each defining a contribution to a channel.

"""

from .base import Serializable
from .norms import Norm, NumberNorm, ParameterNorm, FormulaNorm, LinearCombNorm
import numpy as np
from abc import abstractmethod


# -------------------------------------------------------------------------
class Sample(Serializable) :
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
               nominal_yields : np.ndarray = None, impacts : dict = None) :
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
    self.impacts = impacts if impacts is not None else {}
    self.pos_vars = {}
    self.neg_vars = {}
    self.pos_imps = {}
    self.neg_imps = {}

  def set_np_data(self, nps : list, variation : float = 1, verbosity : int = 0) :
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
        if verbosity > 0 : print("Using normalization = 1 for sample '%s'." % self.name)
        self.nominal_norm = 1
    if self.nominal_yields is None :
      self.nominal_yields = np.array([ self.nominal_norm ])
    for par in nps :
      if par.name in self.impacts : continue
      imp_impact = self.norm.implicit_impact(par, +variation)
      if imp_impact is None : continue
      self.impacts[par.name] =  imp_impact

  def available_variations(self, par : str = None) -> list :
    """provides the available variations of the per-bin event yields for a given NP

      Args:
         par       : name of the NP
      Returns:
        list of available variations
    """
    if par is None and len(self.impacts) > 0 : par = list(self.impacts.keys())[0]
    if not par in self.impacts : return [+1, -1] # no registered impact: return dummy +/- sigma variations of 0 (see below)
    #raise KeyError("No impact defined in sample '%s' for parameter '%s'." % (self.name, par))
    if len(self.impacts[par]) == 0 : return []
    prototype = self.impacts[par][0] if isinstance(self.impacts[par], list) else self.impacts[par]
    return [ +1 if var == 'pos' else -1 if var == 'neg' else float(var) for var in prototype.keys() ]

  def impact(self, par : str, variation : float = +1, normalize : bool = False) -> np.array :
    """provides the relative variations of the per-bin event yields for a given NP

      Args:
         par       : name of the NP
         variation : magnitude of the variation, in numbers of sigmas
         normalize : if True, return the variation scaled to a +1sigma effect
      Returns:
         per-bin relative variations (shape : nbins)
    """
    if not par in self.impacts : return np.zeros(len(self.nominal_yields)) # no registered impact: return [0...0]
    #  raise KeyError('No impact defined in sample %s for parameters %s.' % (self.name, par))
    imp = None
    # Case 1 : the impacts are provided as a list, with one entry per bin. Each list entry is a dict with variation:impact pairs 
    if isinstance(self.impacts[par], list) :
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
    # Case 2 : the impacts are provided as a single dict, which is common for all the bins
    if isinstance(self.impacts[par], dict) :
      imp = np.array([ self.impacts[par]['%+g' % variation] ] * len(self.nominal_yields), dtype=float)
    if imp is None : return imp
    return (1 + imp)**(1/variation) - 1 if normalize else imp

  def impact_coefficients(self, par : str, variations = None, is_log : bool = False) -> list :
    """Returns a parameterization of the impact values for an NP

      Impacts are computed for fixed variations (+1, -1, etc.). In the model,
      an interpolation must be used based on these values. The simplest case
      is for a single variation, where a linear interpolation is used. For
      cases with more interpolation points, a polynomial is used.

      The interpolation can be in linear scale (suitable for interpolating
      yields as N0*(1 + pol(NP))) or exponential scale (as N0*exp(pol(NP))).

      Args:
         par       : name of the NP
         variation : list of variations to consider (default: None, uses what is available)
         is_bool   : interpolate in log (True) or linear (False) scale
      Returns:
         Polynomial coefficients of the interpolation (pair of np.arrays of shape (nbins, len(variations))
    """
    # First, determine the list of positive and negative variations for which we need to compute impacts:
    if variations is not None :
      self.pos_vars[par] = [+v for v in variations ]
      self.neg_vars[par] = [-v for v in variations ]
    else :
      available = self.available_variations(par)
      self.pos_vars[par] = sorted([ v for v in available if v > 0 ])
      self.neg_vars[par] = sorted([ v for v in available if v < 0 ])
    # Compute the impacts for each reported variation. These are 2D arrays of size (nvariations, nbins).
    self.pos_imps[par] = np.array([ self.impact(par, var) for var in self.pos_vars[par] ])
    self.neg_imps[par] = np.array([ self.impact(par, var) for var in self.neg_vars[par] ])
    max_impact = 100
    # Shortcut for the case of only 1 variation
    if len(self.pos_vars[par]) == 1 :
      pos_coeffs = [ self.pos_imps[par][0]/self.pos_vars[par][0] if not is_log else np.log(np.maximum(np.minimum(1 + self.pos_imps[par][0], max_impact), 1/max_impact))/self.pos_vars[par][0] ]
      neg_coeffs = [ self.neg_imps[par][0]/self.neg_vars[par][0] if not is_log else np.log(np.maximum(np.minimum(1 + self.neg_imps[par][0], max_impact), 1/max_impact))/self.neg_vars[par][0] ]
      return np.array(pos_coeffs), np.array(neg_coeffs)
    # Otherwise go the longer route of interpolation between multiple variations
    try:
      # The impacts again -- unchanged for the linear case, log'ed for the exponential case
      pos_imp_vals = [ pos_imp if not is_log else np.log(np.maximum(np.minimum(1 + pos_imp, max_impact), 1/max_impact)) for pos_imp in self.pos_imps[par].T ]
      neg_imp_vals = [ neg_imp if not is_log else np.log(np.maximum(np.minimum(1 + neg_imp, max_impact), 1/max_impact)) for neg_imp in self.neg_imps[par].T ]
      # Arrays of per-bin impact coefficients
      pos_coeffs = np.array([ self.interpolate_impact(self.pos_vars[par], pos_imp) for pos_imp in pos_imp_vals ]).T
      neg_coeffs = np.array([ self.interpolate_impact(self.neg_vars[par], neg_imp) for neg_imp in neg_imp_vals ]).T
    except Exception as inst:
      print("ERROR: impact computation failed for parameter '%s'" % par)
      raise(inst)
    return pos_coeffs, neg_coeffs

  def interpolate_impact(self, pos : list, impacts : np.array) -> np.array :
    """Returns polynomial approximant to the impact values
      Args:
         pos : list of parameter variation values
         impacts : list of corresponding impacts
      Returns:
         Polynomial coefficients of the interpolation
    """
    if len(pos) != len(impacts) : raise ValueError("Cannot interpolate, number of x values (%d) doesn't match y values (%d)." % (len(pos), len(impacts)))
    vdm = np.array( [ [ p**(n+1) for n in range(0, len(pos)) ] for p in pos ] )
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
      print("Symmetric impact computation failed for NP '%s' in sample '%s', returning the positive impacts instead." % (par, self.name))
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

  def yields(self, pars_dict : dict) -> np.ndarray :
    """Computes the scaled yields for a given value of the POIs

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         the array of normalized event yields
    """
    return self.normalization(pars_dict)*self.nominal_yields

  def norm_gradient(self, pars_dict : dict) -> float :
    """Computes the overall normalization factor

      Args:
         pars_dict : a dictionary of parameter name: value pairs
      Returns:
         normalization factor value
    """
    grad = self.norm.gradient(pars_dict)
    return grad/self.nominal_norm if grad is not None else None

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    s = "Sample '%s', norm = %s (nominal = %s)" % (self.name, str(self.norm), str(self.nominal_norm))
    return s

  def load_dict(self, sdict) -> 'Sample' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    self.name = self.load_field('name', sdict, '', str)
    norm_type = self.load_field('norm_type', sdict, '', str)
    # Automatic typecasting: either NumberNorm or ParameterNorm,
    # depending if the 'norm' parameter represents a float
    if norm_type == '' :
      norm = self.load_field('norm', sdict, None, [int, float, str])
      if isinstance(norm, (int, float)) or norm is None :
        norm_type = NumberNorm.type_str
      else :
        try:
          float(norm)
          norm_type = NumberNorm.type_str
        except Exception as inst:
          norm_type = ParameterNorm.type_str
    if norm_type == NumberNorm.type_str :
      self.norm = NumberNorm()
    elif norm_type == ParameterNorm.type_str :
      self.norm = ParameterNorm()
    elif norm_type == FormulaNorm.type_str :
      self.norm = FormulaNorm()
    elif norm_type == LinearCombNorm.type_str :
      self.norm = LinearCombNorm()
    elif norm_type == '' :
      raise KeyError("Could not guess normalization type for sample '%s', please specify it explicitly." % self.name)
    else :
      raise KeyError("Unknown normalisation type '%s' in sample '%s'." % (self.norm_type, self.name))
    self.norm.load_dict(sdict)
    self.nominal_norm = self.load_field('nominal_norm', sdict, None, [float, int])
    self.nominal_yields = self.load_field('nominal_yields', sdict, None, list)
    self.impacts = self.load_field('impacts', sdict, {}, dict)
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    sdict['name'] = self.name
    self.norm.fill_dict(sdict)
    sdict['nominal_norm']   = self.unnumpy(self.nominal_norm)
    sdict['nominal_yields'] = self.unnumpy(self.nominal_yields)
    sdict['impacts']        = self.unnumpy(self.impacts)

