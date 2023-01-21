"""
Utility classes for model manipulation:

* :class:`ModelMerger`: tool to combine multiple models into a single one.

* :class:`ChannelMerger`: tool to merge groups of model channels into a single
  multi-bin channel (useful for some moels where bins spanning a range where
  defined as single bins, but can be better merged into a single `binned_range` channel)

* :class:`ModelReparam`: tool to reparameterize the model (define new model parameters
  expressions, replace expressions with new ones, etc.).

* :class:`NPPruner`: tool to simplify a model by pruning away nuisance parameters
  with small impacts.

* :class:`SamplePruner`: tool to simplify a model by pruning away samples with negligible
  expected event yields compared to other samples in the same channel.

* :class:`ParBound`: utility class defining the bounds of a fit parameter.
"""

import re
import math
import numpy as np
import itertools

from .core  import Model, Data, Parameters, ModelPOI
from .norms import ExpressionNorm, NumberNorm
from .sample import Sample
from .channels  import SingleBinChannel, MultiBinChannel, BinnedRangeChannel
  
# -------------------------------------------------------------------------
class ModelMerger :
  """Utility class to combine multiple models

  Provides functionality to merge multiple models into a single
  combined model. The class takes as input a list of models, and
  merges everything into the first one, which is modified in place.
  All the models should have the same number of POIs, NPs and samples,
  and distinct channels.
  TODO: make this class a bit more versatile

  Atttributes:
    target (Model) : the model into which the others
                     will be merged.
    models (list) : the list of models to merge
    verbosity (int) : the verbosity of the output
  """
  
  def __init__(self, models, verbosity : int = 0) :
    """Initialize the object

    Args:
      model : the list of models
      verbosity : the verbosity of the output
    """
    if len(models) == 0 :
      raise ValueError('Merging not possible: no models specified')
    self.target = models[0]
    self.models = models[1:]
    self.verbosity = verbosity

  def check(self) -> bool :
    """Perform checks on the merging inputs

    Checks that the structure of POIs, NPs and samples
    is the same for all models (should be relaxed ?...)
    and that no channels overlap.

    Returns:
      True if the check is passed, False otherwise.
    """
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

  def merge(self) -> Model :
    """Perform the model merge

    Merges the models by concatenating the lists of channels,
    nominal yields and impacts.

    Returns:
      the merged Model
    """
    for model in self.models :
      for channel in model.channels :
        self.target.channels[channel] = model.channels[channel]
        self.target.channel_offsets[channel] = model.channel_offsets[channel] + self.target.nbins
        self.target.nbins += model.nbins
    all_models = [ self.target ] + self.models 
    max_nsamples = max([ model.max_nsamples for model in all_models ])
    if self.verbosity > 1 : print('Will resize all models to %d samples' % max_nsamples)
    self.target.sym_impact_coeffs = np.concatenate(tuple(np.pad(model.sym_impact_coeffs, ((0, max_nsamples - model.max_nsamples),(0,0),(0,0))) for model in all_models), axis=1)
    self.target.pos_impact_coeffs = np.concatenate(tuple(np.pad(model.pos_impact_coeffs, ((0, max_nsamples - model.max_nsamples),(0,0),(0,0),(0,0))) for model in all_models), axis=1)
    self.target.neg_impact_coeffs = np.concatenate(tuple(np.pad(model.neg_impact_coeffs, ((0, max_nsamples - model.max_nsamples),(0,0),(0,0),(0,0))) for model in all_models), axis=1)
    if self.target.use_lognormal_terms :
      self.target.log_sym_impact_coeffs = np.concatenate(tuple(np.pad(model.log_sym_impact_coeffs, ((0, max_nsamples - model.max_nsamples),(0,0),(0,0))) for model in all_models), axis=1)
      self.target.log_pos_impact_coeffs = np.concatenate(tuple(np.pad(model.log_pos_impact_coeffs, ((0, max_nsamples - model.max_nsamples),(0,0),(0,0),(0,0))) for model in all_models), axis=1)
      self.target.log_neg_impact_coeffs = np.concatenate(tuple(np.pad(model.log_neg_impact_coeffs, ((0, max_nsamples - model.max_nsamples),(0,0),(0,0),(0,0))) for model in all_models), axis=1)
    self.target.nominal_yields = np.concatenate(tuple(np.pad(model.nominal_yields, ((0, max_nsamples - model.max_nsamples),(0,0))) for model in all_models), axis=1)
    return self.target


# -------------------------------------------------------------------------
class ChannelMerger :
  """Utility class to merge model channels

  Provides functionality to merge a group of channels into a single multi-bin
  channels. This is useful e.g. for channels which are effectively binned ranges
  but have been defined as a set of single-bin channels e.g. in pyhf.
  
  The merged channel can be either a BinnedRangeChannel or a MultiBinChannel,
  depending on cases. For a BinnedRangeChannel, the each bin in the merged channel
  should be associated with a range of an observable. These ranges are specified 
  by `obs_bins`, while `obs_name` and `obs_unit` specify the observable.

  Attributes:
    model (Model) : the model on which to act.
    channels_to_merge (list) : list of names of channels to merge.
    merged_name (str) : name of the merged channel.
    obs_bins (list) : list of (lo, hi) pairs, defining the range of the observable
                      that is associated to each bin in the merged channel, for
                      a merge into a BinnedRangeChannel.
    obs_name (str) : name of the observable, for a merge into a BinnedRangeChannel.
    obs_unit (str) : unit of the observable, for a merge into a BinnedRangeChannel.
    verbosity (int) : the verbosity of the output
    merged_samples (dict) : internal dict of samples for the merged channel.
  """
  
  def __init__(self, model : Model, channels_to_merge : list, merged_name : str,
               obs_bins : list = None, obs_name : str = None, obs_unit : str = '',
               verbosity : int = 0) :
    """Initialize the object

    Args:
      model : the model on which to act.
      channels_to_merge : list of names of channels to merge.
      merged_name : name of the merged channel.
      obs_bins : list of bin (lo, hi) values, for a merge into a BinnedRangeChannel.
      obs_name : name of the observable, for a merge into a BinnedRangeChannel.
      obs_unit : unit of the observable, for a merge into a BinnedRangeChannel.
      verbosity : the verbosity of the output
    """
    self.model = model
    self.channels_to_merge = channels_to_merge
    self.merged_name = merged_name
    self.obs_bins = obs_bins
    self.obs_name = obs_name
    self.obs_unit = obs_unit
    self.verbosity = verbosity
    self.merged_samples = {}

  def add_sample(self, sample) :
    """Internal function to add a new merged sample

    Adds one sample to the lst of samples for the merged channel.
    (The merged channel needs to contain all the samples present
    in at least one of the channels to merge.)

    Args:
      sample : Sample object to add to the list
    """
    self.merged_samples[sample.name] = Sample(sample.name, sample.norm, sample.nominal_norm, sample.nominal_yields, sample.impacts)

  def check(self) -> bool :
    """Perform checks on the merging inputs

    Checks that a valid list of channels has been provided and
    in the process collect the list of 

    Returns:
      True if the check is passed, False otherwise.
    """
    if len(self.channels_to_merge) == 0 : 
      print('ERROR: cannot merge an empty list of channels')
      return False
    for channel in self.channels_to_merge[1:] :
      if channel not in self.model.channels :
        print("ERROR: unknown channel '%s'." % channel)
        return False   
    self.merged_samples = {}
    longest_channel = None
    for channel in self.channels_to_merge :
      chan = self.model.channels[channel]
      if isinstance(chan, BinnedRangeChannel) :
        print("ERROR: cannot cannot merge channel '%s' of binned range type." % chan.name)
        return False
      if longest_channel is None or len(chan.samples) > len(longest_channel.samples) :
        longest_channel = chan
    for sample in longest_channel.samples.values() : self.add_sample(sample)
    for channel in self.channels_to_merge :
      chan = self.model.channels[channel]
      for sample in chan.samples : 
        if sample not in self.merged_samples :
          self.add_sample(chan.samples[sample])
        else :
          merged_sample = self.merged_samples[sample]
          # Make sure all the NPs from all samples are listed, to avoid lookup errors later
          # the nominal yields are updated in merge below
          for par in chan.samples[sample].impacts :
            if par not in merged_sample.impacts : merged_sample.impacts[par] = {"+1" : 0, "-1" : 0}
    return True

  def merge(self) -> Model :
    """Perform the channel merge for the model

    Merges the channels specified in __init__ and returns
    a new model where they are replaced by the merge.
    The new model is a shallow copy of the original, except
    for the merged channel.

    Returns:
      the merged Model
    """
    if not self.check() : return None
    for merged_sample in self.merged_samples.values() :
      all_yields = []
      for channel in self.channels_to_merge :
        chan = self.model.channels[channel]
        if merged_sample.name in chan.samples :
          sample = chan.samples[merged_sample.name]
          all_yields.append(sample.nominal_yields/sample.nominal_norm*merged_sample.nominal_norm)
        else :
          all_yields.append(np.zeros(chan.nbins()))
      merged_sample.nominal_yields = np.concatenate(all_yields)
      for par in merged_sample.impacts :
        all_impacts = []
        for channel in self.channels_to_merge :
          chan = self.model.channels[channel]
          impact = [{"+1" : 0, "-1" : 0}]
          if merged_sample.name in chan.samples :
            sample = self.model.channels[channel].samples[merged_sample.name]
            if par in sample.impacts : impact = sample.impacts[par]
            if isinstance(impact, dict) : impact = [ impact ]
          all_impacts.extend(impact)
        merged_sample.impacts[par] = list(itertools.chain(all_impacts))
    if self.obs_bins is None :
      merged_bins = []
      for channel in self.channels_to_merge :
        chan = self.model.channels[channel]
        if isinstance(chan, SingleBinChannel) :
          merged_bins.append(channel)
        elif isinstance(chan, MultiBinChannel) :
          merged_bins.extend(chan.bins)
    else :
      if len(self.obs_bins) != len(self.channels_to_merge) :
        print("ERROR: specified '%d' bins, was expecting the same as the number of channels, '%d'." % (len(self.obs_bins), len(self.channels_to_merge)))
        return None
      merged_bins = [ { 'lo_edge' : obs_bin[0], 'hi_edge' : obs_bin[1] } for obs_bin in self.obs_bins ]
    merged_channels = {}
    merged_added = False
    for channel in self.model.channels.values() :
      if channel.name not in self.channels_to_merge :
        merged_channels[channel.name] = channel
      elif not merged_added :
        if self.obs_bins is None :
          merged_channels[self.merged_name] = MultiBinChannel(self.merged_name, merged_bins, self.merged_samples)
        else :
          merged_channels[self.merged_name] = BinnedRangeChannel(self.merged_name, merged_bins, self.obs_name, self.obs_unit, self.merged_samples)    
        merged_added = True
    return Model(self.model.name, self.model.pois, self.model.nps, self.model.aux_obs, merged_channels, self.model.expressions,
                 self.model.use_asym_impacts, self.model.use_linear_nps, self.model.use_simple_sym_impacts,
                 self.model.use_lognormal_terms, self.model.variations, self.model.verbosity)
  
  def merge_data(self, data : Data, merged_model : Model) -> Data :
    """Perform the channel merge for the data

    Returns a dataset where the channels are merged
    to match what was done on the model side

    Args:
      data : the original dataset (based on the original Model)
      merged_model : the merged model
    Returns:
      the merged data
    """
    if not self.check() : return None
    merged_counts = []
    merged_added = False
    for channel in self.model.channels :
      if channel not in self.channels_to_merge :
        merged_counts.extend(self.model.channel_n_exp(nexp=data.counts, channel=channel))
      elif not merged_added :
        for merged_channel in self.channels_to_merge : 
          merged_counts.extend(self.model.channel_n_exp(nexp=data.counts, channel=merged_channel))
        merged_added = True
    return Data(merged_model, merged_counts, data.aux_obs)


# -------------------------------------------------------------------------
class ModelReparam :
  """Utility class to reparameterize a model

  Provides functionality to add and remove model POIs and expressions,
  and propagate these changes to the existing expressions and to the
  normalization terms of the model samples.

  Atttributes:
    model (Model) : the model on which to act
    verbosity (int) : the verbosity of the output
  """
  
  def __init__(self, model, verbosity : int = 0) :
    """Initialize the object

    Args:
      model (Model) : the model on which to act
      verbosity (int) : the verbosity of the output
    """
    self.model = model
    self.verbosity = verbosity

  def add_pois(self, new_pois : list) :
    """Add new POIs to the model

    Args:
      new_pois : list of :class:`ModelPOI` objects to add to the model
    """
    for poi in new_pois :
      if poi.name in self.model.pois : 
        raise KeyError("Cannot add POI '%s' to model, a POI with that name is already defined." % poi.name)
      if self.verbosity > 0 : print("Adding POI '%s'." % str(poi))
      self.model.pois[poi.name] = poi

  def add_expressions(self, new_expressions : list) :
    """Add new Expressions to the model

    Args:
      new_expressions : list of :class:`Expression` objects to add to the model
    """
    for expr in new_expressions :
      if expr.name in self.model.expressions : 
        raise KeyError("Cannot add expression '%s' to model, an expression with that name is already defined." % expr.name)
      if self.verbosity > 0 : print("Adding expression '%s = %s'." % (expr.name, str(expr)))
      self.model.expressions[expr.name] = expr

  def update_norms(self, new_norms : dict) :
    """Update normalization terms

    The norms as provided as a dict mapping (channel_spec, sample_spec)
    pairs to :class:`Norm` objects.
    The `channel_spec` and `sample_spec` are strings specifying which channel(s) and sample(s)
    should be assigned the new norm, and can contain wildcards

    Args:
      new_norms : dict mapping (channel_spec, sample_spec) pairs to :class:`Norm` objects.
    """
    for channel in self.model.channels.values() :
      for sample in channel.samples.values() :
        matching_norms = []
        for (channel_spec, sample_spec), norm in new_norms.items() :
          try :
            if re.match(channel_spec + '$', channel.name) and re.match(sample_spec + '$', sample.name) : matching_norms.append(norm)
          except Exception as inst :
            print(inst)
            raise ValueError("ERROR: invalid channel specification '%s' or sample specification '%s'." % (channel_spec, sample_spec)) 
        if len(matching_norms) == 0 : continue
        if len(matching_norms) > 1 :
          raise KeyError("Sample '%s' of channel '%s' matches multiple new normalization entries : \n %s" % (sample.name, channel.name, '\n'.join([str(norm) for norm in matching_norms])))
        if self.verbosity > 0 : print("Replacing normalization '%s' -> '%s' in sample '%s' of channel '%s'." % (str(sample.norm), str(matching_norms[0]), sample.name, channel.name))
        sample.norm = matching_norms[0]

  def remove_pois(self, poi_names : list, values : dict = {}) :
    """Remove POIs from the model

    The POIs are removed from the list of model POIs, and
    also from the model expressions.
    They are replaced with a numerical value in each of these.
    A value must be provided for each POI to remove.

    Args:
      poi_names : list of names of POIs to remove
      values : dict of {name :value } pairs providing the values
               with which to replace each POI.
    """
    selected_names = []
    for poi_name in poi_names :
      if poi_name in self.model.pois :
        selected_names.append(poi_name)
      else :
        new_names = [ par_name for par_name in self.model.pois if re.match(poi_name + '$', par_name) ]
        if len(new_names) == 0 :
          raise KeyError("Cannot remove POIs with specification '%s' from model, as no matching POI is defined." % poi_name)
        selected_names.extend(new_names)
    expressions_to_remove = []
    expression_values = {}
    for selected_name in selected_names :
      if self.verbosity > 0 : print("Removing POI '%s'." % selected_name)
      if not selected_name in self.model.expressions : # if we just replaced this POI by an expression, leave the norms as they are
        self.remove_from_norms(selected_name, values)
      for expr in self.model.expressions.values() :
        value = expr.replace(selected_name, values[selected_name] if selected_name in values else None, self.model.reals())
        if value != expr : # this is a numerical value, for when the expression has become trivial
          expressions_to_remove.append(expr.name)
          expression_values[expr.name] = value.val
      self.model.pois.pop(selected_name)
    self.remove_expressions(expressions_to_remove, expression_values)

  def remove_expressions(self, expr_names : list, values : dict = {}) :
    """Remove Expressions from the model

    The expressions are removed from the list of model expressions, and
    also from other model expressions. A replacement
    value for the expression must be provided in each case

    Args:
      expr_names : list of names of POIs to remove
      values : dict of {name :value } pairs providing the values
               with which to replace each expression.
    """
    selected_names = []
    for expr_name in expr_names :
      if expr_name in self.model.expressions :
        selected_names.append(expr_name)
      else :
        new_names = [ par_name for par_name in self.model.expressions if re.match(expr_name, par_name) ]
        if len(new_names) == 0 :
          raise KeyError("Cannot remove POIs with specification '%s' from model, as no matching POI is defined." % expr_name)
        selected_names.extend(new_names)
    for selected_name in selected_names :
      if self.verbosity > 0 : print("Removing expression '%s'." % selected_name)
      expr = self.model.expressions.pop(selected_name)
      self.remove_from_norms(expr.name, values)

  def remove_from_norms(self, name : str, values : dict = {}) :
    """Remove POIs and Expressions from the normalization terms

    The POIs and expressions are removed from normalization
    terms of each sample where they enter. They are replaced with
    a numerical value that must be provided in each case.
    Args:
      names : list of names of POIs and expressions to remove
      values : dict of {name :value } pairs providing the values with which to replace each POI or expression.
    """
    for channel in self.model.channels.values() :
      for sample in channel.samples.values() :
        if isinstance(sample.norm, ExpressionNorm) and sample.norm.expr_name == name :
          if name in values :
            sample.norm = NumberNorm(values[selected_name])
            if self.verbosity > 0 :
                print("Using %s=%g replacement in the normalization of sample '%s' of channel '%s'." % (name, values[name], sample.name, channel.name))
          else :
            raise KeyError("Cannot remove '%s' as it is used to normalize sample '%s' of channel '%s'. Please provide a numerical value to use as a replacement." % (name, sample.name, channel.name))

# -------------------------------------------------------------------------
class NPPruner :
  """Utility class to prune away model NPs

  Evaluates the impact of individual NPs in the model,
  and removes the ones with max impact below a given
  threshold (taking the maximum over all model bins).
  
  The pruning in principle reduces the size and complexity of the model,
  and therefore loading time, computation time and markup file size, 
  while not changing significantly the results.
  The pruning threshold should however be carefully tuned to ensure
  correct behavior.

  Atttributes:
    model (Model) : the model on which to act
    verbosity (int) : the verbosity of the output
  """
  
  def __init__(self, model, verbosity : int = 0) :
    """Initialize the object

    Args:
      model (Model) : the model on which to act
      verbosity (int) : the verbosity of the output
    """
    self.model = model
    self.verbosity = verbosity

  def prune(self, min_impact : float = 1E-3) :
    """Perform the pruning

    Args:
      min_impact : the impact threshold below which NPs are pruned
    Returns:
      the pruned model
    """
    pruned_nps = []
    for i, par_name in enumerate(self.model.nps) :
      if np.amax(self.model.sym_impact_coeffs[:,:,i]) < min_impact and np.amin(self.model.sym_impact_coeffs[:,:,i]) > -min_impact :
        pruned_nps.append(par_name)
        if self.verbosity > 0 : print("Pruning away nuisance parameter '%s'." % par_name)
    for channel in self.model.channels.values() :
      for sample in channel.samples.values() :
        for pruned_np in pruned_nps : sample.impacts.pop(pruned_np)
    return self.remove_nps({ par : None for par in pruned_nps })

  def remove_nps(self, par_values : dict, clone_model : bool = False) :
    """Remove NPs from the model

    The specified NPs are removed from the list of model NPs.
    The NP are replaced with a user-specified value, so the 
    nominal event yields for all samples are also adjusted to account for
    the difference wrt the nominal NP values for which they were computed.
    The NPs are also removed from normalization terms and model expressions
    (still WIP, see below).

    TODO: handle NPs in expression as well

    Args:
      par_values : NPs to remove, as a { name_spec: value } dict. name_spec can contain wildcards.
      clone_model : if `True`, a cloned model is produced. If `False` (default), the model is pruned in situ
    Returns:
      the pruned model
    """
    model = self.model.clone(set_internal_vars=False, name=self.model.name + '_pruned') if clone_model else self.model
    pars = model.ref_pars.clone()
    removed = []
    for np_name in par_values :
      if np_name in model.nps :
        selected_names = [ np_name ]
      else :
        selected_names = [ par_name for par_name in model.nps if re.match(np_name + '$', par_name) ]
        if len(selected_names) == 0 :
          raise KeyError("Cannot remove NP '%s' from model, no matching NP is defined." % np_name)
      for selected_name in selected_names :
        par = model.nps[selected_name]
        value = par_values[selected_name] if selected_name in par_values and par_values[selected_name] is not None else par.nominal_value
        pars[selected_name] = value
        if self.verbosity > 0 : print("Applying %s=%g" % (selected_name, value))
        removed.append(par)
        for channel in model.channels.values() :
          for s, sample in enumerate(channel.samples.values()) :
            if isinstance(sample.norm, ExpressionNorm) and sample.norm.expr_name == selected_name :
              #unscaled_value = par.unscaled_value(value)
              # We replace by the *nominal* norm, since the effect of the NP will be accounted for
              # in the nominal_yields below. It would be more elegant to do it the other way, but
              # One would need to then correct the nexp sample by sample to remove the normalization
              # effect we apply here, which doesn't seem optimal.
              if self.verbosity > 0 :
                print("Using %s=%g replacement in normalization of sample '%s' of channel '%s'." % 
                      (selected_name, sample.nominal_norm, sample.name, channel.name))
              sample.norm = NumberNorm(sample.nominal_norm)
            else :
              if selected_name in sample.impacts : sample.impacts.pop(selected_name)
        # TODO: should also replace by its value in expressions.
    nexp = model.n_exp(pars)
    real_vals = model.real_vals(pars)
    if self.verbosity > 1 : print('Old pars :\n', model.ref_pars)
    if self.verbosity > 1 : print('New pars :\n', pars)
    for channel in model.channels.values() :
      for s, sample in enumerate(channel.samples.values()) :
        if self.verbosity > 0 : print("Applying changes to the nominal yields of sample '%s' of channel '%s'." % (sample.name, channel.name))
        if self.verbosity > 1 : print('Old yields :\n', sample.nominal_yields)
        sample.nominal_yields = model.channel_n_exp(nexp=nexp, channel=channel.name, sample=sample.name)
        sample.nominal_norm = sample.norm.value(real_vals)
        if self.verbosity > 1 : print('New yields :\n', sample.nominal_yields)
    for par in removed :
      if par.aux_obs : model.aux_obs.pop(par.aux_obs)
      model.nps.pop(par.name)
    model.set_internal_vars()
    return model

      
# -------------------------------------------------------------------------
class SamplePruner :
  """Utility class to prune away model samples

  Evaluates the contribution of individual samples in a channel, and
  prunes away the samples with negligible contributions. 
  The figure of merit is the `total significance` of the sample yields 
  over the rest of the channel samples.
  
  The pruning in principle reduces the size and complexity of the model,
  and therefore loading time, computation time and markup file size, 
  while not changing significantly the results.
  The pruning threshold should however be carefully tuned to ensure
  correct behavior.

  Atttributes:
    model (Model) : the model on which to act
    verbosity (int) : the verbosity of the output
  """
  
  def __init__(self, model, verbosity : int = 0) :
    """Initialize the object

    Args:
      model (Model) : the model on which to act
      verbosity (int) : the verbosity of the output
    """
    self.model = model
    self.verbosity = verbosity

  def prune(self, min_signif : float = 1E-3) -> Model :
    """Perform the pruning

    Args:
      min_signif : the significance threshold below which samples are pruned
    Returns:
      the pruned model
    """
    changed = False
    for channel in self.model.channels.values() :
      total_nominal_yields = np.zeros(channel.nbins())
      for sample in channel.samples.values() : total_nominal_yields += sample.nominal_yields
      to_remove = []
      for sample in channel.samples.values() :
        z = self.total_significance(sample.nominal_yields, total_nominal_yields)
        if z < min_signif :
          to_remove.append(sample.name)
          if self.verbosity > 0 :
            print("Pruning away sample '%s' in channel '%s', significance = %g < %g." % (sample.name, channel.name, z, min_signif))
          changed = True
        elif self.verbosity > 1 :
          print("Keeping sample '%s' in channel '%s', significance = %g >= %g." % (sample.name, channel.name, z, min_signif))
      for sample_name in to_remove : channel.samples.pop(sample_name)
    if changed : self.model.set_internal_vars()
    return self.model
      
  def total_significance(self, nexp_sample : np.ndarray, nexp_total : np.ndarray) -> float :
    """Compute the significance value for a sample

    The function computes the `total significance` of the sample,
    the figure of merit used to decide whether to prune it or not.
    
    The total significance is computed as the sum in quadrature of
    per-bin significances (which is correct for expected significances).
    The per-bin values are in turn computed using the "Asimov" formula
    for significance.

    Args:
      nexp_sample : a 1D array containing the event yields for this sample
      nexp_total  : a 1D array containing the event yields for all the samples in the channe;
    Returns:
      the sample significance.
    """
    if np.min(nexp_total - nexp_sample) <= 0 : return np.Infinity
    try :
      return math.sqrt(np.sum(2*(nexp_total*np.log(nexp_total/(nexp_total - nexp_sample)) - nexp_sample)))
    except :
      return np.Infinity

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
    """Initialize the object

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
    if self.max_value is not None and self.min_value is None : return '%s <= %g' % (self.par, self.max_value)
    if self.min_value is not None and self.max_value is None: return '%s >= %g' % (self.par, self.min_value)
    return '%g <= %s <= %g' % (self.min_value, self.par, self.max_value)

  def __repr__(self) -> str:
    """Provides a description string for the object

    Needed in addition to :meth:`__str__` to print out correctly
    lists of :class:`ParBound` objects.

    Returns:
      a description string
    """
    return self.__str__()
