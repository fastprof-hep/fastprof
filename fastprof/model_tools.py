"""
Utility classes for model operations

"""
import re
import numpy as np

from .core  import Model, Data, Parameters, ModelPOI
from .norms import ParameterNorm, FormulaNorm
  
# -------------------------------------------------------------------------
class ModelMerger :
  
  def __init__(self, models) :
    self.models = models
    if len(models) == 0 :
      raise ValueError('Merging not possible: no models specified')
    self.target = models[0]
    self.models = models[1:]

  def check(self) :
    for i, model in enumerate(self.models) :
      if model.npois != self.target.npois :
        print("Merging not possible: unexpected number of POIs for model at index %d : got %d, should be %d." % (i+1, model.npois, self.target.npois))
        return False
      if model.nnps != self.target.nnps :
        print("Merging not possible: unexpected number of NPs for model at index %d : got %d, should be %d." % (i+1, model.nnps, self.target.nnps))
        return False
      if model.ncons != self.target.ncons :
        print("Merging not possible: unexpected number of constrained NPs for model at index %d : got %d, should be %d." % (i+1, model.ncons, self.target.ncons))
        return False
      if len(model.samples) != len(self.target.samples) :
        print("Merging not possible: unexpected number of samples for model at index %d : got %d, should be %d." % (i+1, len(model.samples), len(self.target.samples)))
        return False
      for par_mod, par_ref in zip(model.pois, self.target.pois) :
        if par_mod != par_ref :
          print("Merging not possible: unexpected POI at index %d : got %s, should be %s." % (i, par_mod, par_ref))
          return False
      for par_mod, par_ref in zip(model.nps, self.target.nps) :
        if par_mod != par_ref :
          print("Merging not possible: unexpected NP at index %d : got %s, should be %s." % (i, par_mod, par_ref))
          return False
      for sample_ref, sample_mod in zip(model.samples, self.target.samples) :
        if sample_mod != sample_ref :
          print("Merging not possible: unexpected sample at index %d : got %s, should be %s." % (i, sample_mod, sample_ref))
          return False
      for channel_mod in enumerate(model.channels) :
        if channel_mod in self.target.channels :
          print("Merging not possible: model at index %d includes channel %s, which is already present in the target model." % (i, channel_mod))
          return False
    return True

  def merge(self) :
    for model in self.models :
      for channel in model.channels :
        self.target.channels[channel] = model.channels[channel]
        self.target.channel_offsets[channel] = model.channel_offsets[channel] + self.target.nbins
        self.target.nbins += model.nbins
    all_models = [ self.target ] + self.models 
    self.target.nominal_yields = np.concatenate(tuple(model.nominal_yields for model in all_models), axis=1)
    self.target.sym_impact_coeffs = np.concatenate(tuple(model.sym_impact_coeffs for model in all_models), axis=1)
    self.target.pos_impact_coeffs = np.concatenate(tuple(model.pos_impact_coeffs for model in all_models), axis=1)
    self.target.neg_impact_coeffs = np.concatenate(tuple(model.neg_impact_coeffs for model in all_models), axis=1)
    if self.target.use_lognormal_terms :
      self.target.log_sym_impact_coeffs = np.concatenate(tuple(model.log_sym_impact_coeffs for model in all_models), axis=1)
      self.target.log_pos_impact_coeffs = np.concatenate(tuple(model.log_pos_impact_coeffs for model in all_models), axis=1)
      self.target.log_neg_impact_coeffs = np.concatenate(tuple(model.log_neg_impact_coeffs for model in all_models), axis=1)
    return self.target

# -------------------------------------------------------------------------
class ModelReparam :
  
  def __init__(self, model) :
    self.model = model

  def add_pois(self, new_pois : list, verbosity : int = 0) :
    for poi in new_pois :
      if poi.name in self.model.pois : 
        raise KeyError("Cannot add POI '%s' to model, a POI with that name is already defined." % poi.name)
      if verbosity > 0 : print("Adding POI '%s'." % str(poi))
      self.model.pois[poi.name] = poi

  def update_norms(self, new_norms : dict, verbosity : int = 0) :
    for channel in self.model.channels.values() :
      for sample in channel.samples.values() :
        matching_norms = [ norm for (channel_spec, sample_spec), norm in new_norms.items() if re.match(channel_spec, channel.name) and re.match(sample_spec, sample.name) ]
        if len(matching_norms) == 0 : continue
        if len(matching_norms) > 1 :
          raise KeyError("Sample '%s' of channel '%s' matches multiple new normalization entries : \n %s" % (sample.name, channel.name, '\n'.join([str(norm) for norm in matching_norms])))
        if verbosity > 0 : print("Replacing normalization '%s' -> '%s' in sample '%s' of channel '%s'." % (str(sample.norm), str(matching_norms[0]), sample.name, channel.name))
        sample.norm = matching_norms[0]

  def remove_pois(self, poi_names : list, values : dict = {}, verbosity : int = 0) :
    selected_names = []
    for poi_name in poi_names :
      if poi_name in self.model.pois :
        selected_names.append(poi_name)
      else :
        new_names = [ par_name for par_name in self.model.pois if re.match(poi_name, par_name) ]
        if len(new_names) == 0 :
          raise KeyError("Cannot remove POI '%s' from model, no matching POI is defined." % poi_name)
        selected_names.extend(new_names)
    for selected_name in selected_names :
      if verbosity > 0 : print("Removing POI '%s'." % selected_name)
      poi = self.model.pois.pop(selected_name)
      for channel in self.model.channels.values() :
        for sample in channel.samples.values() :
          if isinstance(sample.norm, ParameterNorm) and sample.norm.par_name == selected_name :
            if selected_name not in values :
              raise KeyError("Cannot remove POI '%s' as it is used to normalize sample '%s' of channel '%s'. Please provide a numerical value to use as a replacement." % (selected_name, sample.name, channel.name))
            if verbosity > 0 : print("Using %s=%g replacement in the normalization of sample '%s' of channel '%s'." % (selected_name, values[selected_name], sample.name, channel.name))
            sample.norm = NumberNorm(values[selected_name])
          if isinstance(sample.norm, FormulaNorm) and sample.norm.formula.find(selected_name) != -1 :
            raise KeyError("Cannot remove POI '%s' as it is used to in formula '%s' normalizing sample '%s' of channel '%s'." % (selected_name, sample.norm.formula, sample.name, channel.name))


# -------------------------------------------------------------------------
class ModelPruner :
  
  def __init__(self, model) :
    self.model = model

  def prune(self, min_impact : float = 1E-3) :
    pruned_nps = []
    for i, par_name in enumerate(self.model.nps) :
      if np.amax(self.model.sym_impact_coeffs[:,:,i]) < min_impact and np.amin(self.model.sym_impact_coeffs[:,:,i]) > -min_impact :
        pruned_nps.append(par_name)
        print("Pruning away nuisance parameter '%s'." % par_name)
    for channel in self.model.channels.values() :
      for sample in channel.samples.values() :
        for pruned_np in pruned_nps : sample.impacts.pop(pruned_np)
    self.remove_nps(pruned_nps)

  def remove_nps(self, np_names : list) :
    for np_name in np_names :
      if np_name in self.model.nps :
        selected_names = [ np_name ]
      else :
        selected_names = [ par_name for par_name in self.model.nps if re.match(np_name, par_name) ]
        if len(selected_names) == 0 :
          raise KeyError("Cannot remove NP '%s' from model, no matching NP is defined." % np_name)
      for selected_name in selected_names :
        par = self.model.nps.pop(selected_name)
        if par.aux_obs : self.model.aux_obs.pop(par.aux_obs)
        for channel in self.model.channels.values() :
          for sample in channel.samples.values() :
            if isinstance(sample.norm, ParameterNorm) and sample.norm.par_name == selected_name :
              print("Using %s=%g replacement in normalization of sample '%s' of channel '%s'." % (selected_name, par.nominal_value, sample.name, channel.name))
              sample.norm = NumberNorm(par.nominal_value)
            if isinstance(sample.norm, FormulaNorm) and sample.norm.formula.find(selected_name) != -1 :
              raise KeyError("Cannot remove POI '%s' as it is used to in a formula normalizing sample '%s' of channel '%s'." % (selected_name, sample.name, channel.name))
            if selected_name in sample.impacts : sample.impacts.pop(selected_name)
    self.model.set_internal_vars()


# -------------------------------------------------------------------------
class ParBound :
  """Class to define and enforce parameter bounds

  Defines upper and lower bounds on a model parameter,
  and implements a test method applied to :class:`Parameters`
  objects.

  Atttributes:
    par    (str)   : parameter name
    min_value (float) : parameter lower bound (`None` if no bound)
    max_value (float) : parameter upper bound (`None` if no bound)
  """

  def __init__(self, par : str, min_value : float = None, max_value : float = None) :
    """Initialize the `QMuTildaCalculator` object

    Defines a selection min_value <= par <= max_value.
    Both bounds are optional and can be omitted by passing `None` as
    the corresponding argument (also default).

    Args:
      par    : parameter name
      min_value : parameter lower bound (`None` for no bound, default)
      max_value : parameter upper bound (`None` for no bound, default)
    """
    self.par = par
    self.min_value = min_value
    self.max_value = max_value

  def bounds(self) :
    return (self.min_value, self.max_value)

  def is_fixed(self) :
    return self.min_value == self.max_value

  def is_free(self) :
    return self.min_value < self.max_value

  def __and__(self, other) :
    if self.par != other.par : return None
    new_min_value = self.min_value if other.min_value is None else other.min_value if self.min_value is None else max(self.min_value, other.min_value)
    new_max_value = self.max_value if other.max_value is None else other.max_value if self.max_value is None else min(self.max_value, other.max_value)
    return ParBound(self.par, new_min_value, new_max_value)

  def test_value(self, value) -> bool :
    return (value >= self.min_value if self.min_value is not None else True) and (value <= self.max_value if self.max_value is not None else True)

  def test(self, pars : Parameters) -> bool :
    """Applies the selection to a :class:`Parameters` object

    Args:
      pars : a set of model parameter
    Returns:
     `True` if the parameters pass the selection, `False` if they fail.
    """
    try :
      return self.test_value(pars[self.par])
    except KeyError :
      return True

  def __str__(self) -> str :
    """Provides a description string for the object

    Returns:
      a description string
    """
    if self.is_fixed() : return 'fixed at %g' % self.min_value
    smin = '%s >= %g' % (self.par, self.min_value) if self.min_value != None else ''
    smax = '%s <= %g' % (self.par, self.max_value) if self.max_value != None else ''
    if smin == '' : return smax
    if smax == '' : return smin
    return smin + ' and ' + smax

  def __repr__(self) -> str:
    """Provides a description string for the object

    Needed in addition to :meth:`__str__` to print out correctly
    lists of :class:`ParBound` objects.

    Returns:
      a description string
    """
    return self.__str__()
