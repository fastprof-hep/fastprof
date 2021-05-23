"""
Utility classes for model operations

"""

from .core import Model, Data, Parameters, ModelPOI
import re
  
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

  def add_pois(self, new_pois : list) :
    for poi in new_pois :
      if poi.name in self.model.pois : 
        raise KeyError("Cannot add POI '%s' to model, a POI with that name is already defined." % poi.name)
      model.pois[poi.name] = poi
    
  def update_norms(self, new_norms : dict) :
    for channel in model.channels.values() :
      for sample in channel.sample.values() :
        matching_norms = [ norm for (channel_spec, sample_spec), norm in new_norms.keys() if re.match(channel_spec, channel.name) and re.match(sample_spec, sample.name) ]
        if len(matching_norms) == 0 : continue
        if len(matching_norms) > 1 :
          raise KeyError("Sample '%s' of channel '%s' matches multiple new normalization entries : \n %s" % (sample.name, channel.name, '\n'.join([str(norm) for norm in matching_norms])))
        print("Replacing normalization of  sample '%s' of channel '%s' by %s." % (sample.name, channel.name, str(matching_norms[0])))
        sample.norm = matching_norms[0]

  def remove_pois(self, poi_names : list, values : dict) :
    for poi_name in poi_names :
      if poi_name not in self.model.pois : 
        raise KeyError("Cannot remove POI '%s' from model, no POI with that name is defined." % poi_name)
      poi = model.pois.pop(poi_name)
      for channel in model.channels.values() :
        for sample in channel.sample.values() :
          if isinstance(sample.norm, ParameterNorm) and sample.norm.par_name == poi_name :
            if poi_name not in values :
              raise KeyError("Cannot remove POI '%s' as it is used to normalize sample '%s' of channel '%s'. Please provide a numerical value to use as a replacement." % (poi_name, sample.name, channel.name))
            print("Using %s=%g replacement in normalization of  sample '%s' of channel '%s'." % (poi_name, values[poi_name], sample.name, channel.name))
            sample.norm = NumberNorm(values[poi_name])


# -------------------------------------------------------------------------
class ParBound :
  """Class to define and enforce parameter bounds

  Defines upper and lower bounds on a model parameter,
  and implements a test method applied to :class:`Parameters`
  objects.

  Atttributes:
    par    (str)   : parameter name
    minval (float) : parameter lower bound (`None` if no bound)
    maxval (float) : parameter upper bound (`None` if no bound)
  """

  def __init__(self, par : str, minval : float = None, maxval : float = None) :
    """Initialize the `QMuTildaCalculator` object

    Defines a selection minval <= par <= maxval.
    Both bounds are optional and can be omitted by passing `None` as
    the corresponding argument (also default).

    Args:
      par    : parameter name
      minval : parameter lower bound (`None` for no bound, default)
      maxval : parameter upper bound (`None` for no bound, default)
    """
    self.par = par
    self.minval = minval
    self.maxval = maxval
  def test(self, pars : Parameters) -> bool :
    """Applies the selection to a :class:`Parameters` object

    Args:
      pars : a set of model parameter
    Returns:
     `True` if the parameters pass the selection, `False` if they fail.
    """
    try :
      return (pars[self.par] >= self.minval if self.minval != None else True) and (pars[self.par] <= self.maxval if self.maxval != None else True)
    except KeyError :
      return True
  def __str__(self) -> str :
    """Provides a description string for the object

    Returns:
      a description string
    """
    smin = '%s >= %g' % (self.par, self.minval) if self.minval != None else ''
    smax = '%s <= %g' % (self.par, self.maxval) if self.maxval != None else ''
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
