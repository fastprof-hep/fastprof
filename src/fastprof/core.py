"""Module containing the core fastprof code

The code is organized around the following classes:

  * :class:`Model` : the class implementing the likelihood model. This has the HistFactory structure, with
    stuctural elements that are defined in the :mod:`fastprof.elements` module :

    * The model two types of parameters:

       * *Parameters of interest* (POIs), implemented as :class:`fastprof.elements.ModelPOI` objects

       * *Nuisance parameters* (NPs), implemented as :class:`fastprof.elements.ModelNP` objects.

    * The model is split into a number of channels, represented by :class:`fastprof.elements.Channel` objects, which each define

       * A number of measurement bins

       * A number of samples, represented by :class:`fastprof.elements.Sample` objects, each defining a
         contribution to the expected bin yields.

    * The :class:`fastprof.elements.Sample` objects store their nominal bin yields, an overall normalization and variations as a function of the NPs

  * :class:`Parameters` : a class storing a set of values for model POIs and NPs.

  * :class:`Data` : a class storing the observed data: the observed bin yields for each channel and the auxiliary observables for each NP.

  All the classes support loading from / saving to markup files. The basic mechanism for this is implemented in the :class:`fastprof.elements.Serializable` base class from which they all derive.
"""

import numpy as np
import math, sys
import matplotlib.pyplot as plt

from .base import ModelPOI, ModelNP, ModelAux, Serializable
from .sample import Sample
from .channels import Channel, SingleBinChannel, MultiBinChannel, BinnedRangeChannel

# -------------------------------------------------------------------------
class Parameters :
  """Class representing a set of parameter values

  Stores one full set of model parameter values, both POI and NP.
  Only the numerical values are stored. The parameter names and properties
  can be accessed through the optional `model` attribute, if it set. However
  these are not required for the basic functionality.
  Storage is in 2 np.arrays, one for POIs and one for NPs.

  NPs are stored in their scaled form (see the description of :class:`fastprof.elements.ModelNP`),
  but the unscaled form can be used as well if `model` is set (as this requires
  knowledge of parameter properties not stored locally).

  Attributes:
     pois (np.array): the POI values
     nps (np.array): the NP values
     model (Model): pointer to a :class:`Model` object containing the full
       model information, including parameter names and properties.
  """

  def __init__(self, pois : np.ndarray, nps : np.ndarray = None, model : 'Model' = None) :
    """Initialize a Parameters object from POI and NP values

      The POIs can be provided in a number of formats:

      * A single number, for a model with a single POI

      * An np.ndarray of POI values, with parameter values provided in the
        same order as they appear in the model.

      * A dict of POI name : value pairs with one entry for each model POI.

      NPs are optional; they can be provided as a single number or a np.ndarray
      as for POIS, and will otherwise default to 0.

      Args:
        pois             : float-array of POI values
        nps   (optional) : float-array of NP values
        model (optional  : optional pointer to a :class:`Model` object
    """
    if isinstance(pois, dict) :
      if model is None : raise ValueError('Must provide a model when initializing from a dict') 
      poi_array = np.array([ np.nan ]*model.npois)
      poi_list = list(model.pois)
      for poi, val in pois.items() :
        if poi in poi_list : poi_array[poi_list.index(poi)] = val
      if np.isnan(poi_array).any() : raise ValueError('Input POI dictionary did not contain a valid numerical value for each POI : %s' % str(pois))
      pois = poi_array
    elif isinstance(pois, (float, int)) : 
      if model is not None :
        pois = np.array([pois]*model.npois, dtype=float)
      else :
        pois = np.array([pois], dtype=float)        
    elif isinstance(pois, list) :
      pois = np.array(pois)
    if not isinstance(pois, np.ndarray) or pois.ndim != 1 : raise ValueError('Input POIs should be a 1D np.array, got ' + str(pois))
    if model is not None and pois.size != model.npois :
      raise ValueError('Cannot initialize Parameters with %d POIs, when %d are defined in the model.\nModel POIs:\n%s' % (pois.size, model.npois,'\n'.join(model.pois.keys())))
    self.pois = np.array(pois)
    if nps is None : nps  = np.array([], dtype=float)
    if model is not None and nps.size == 0 : nps = np.zeros(model.nnps)
    if isinstance(nps, (float, int)) : nps  = np.array([ nps ], dtype=float)
    if not isinstance(nps, np.ndarray) or nps.ndim != 1 : raise ValueError('Input NPs should be a 1D np.array, got ' + str(nps))
    if model is not None and nps.size != model.nnps : raise ValueError('Cannot initialize Parameters with %d NPs, when %d are defined in the model'  % (nps .size, model.nnps))
    self.nps  = np.array(nps)
    self.model = model

  def clone(self) -> 'Parameters' :
    """Clone a Parameters object

      Performs a deep-copy operation at the required level: deep-copy
      the np.arrays, but shallow-copy the model pointer

      Returns:
        the new clone
    """
    return Parameters(np.array(self.pois, dtype=float), np.array(self.nps, dtype=float), self.model)

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    s = ''
    if self.model is None :
      s += 'POIs = ' + str(self.pois) + '\n'
      s += 'NPs  = ' + str(self.nps)  + '\n'
    else :
      s += 'POIs : ' + '\n       '.join( [ '%-12s = %8.4f' % (p.name,v) for p, v in zip(self.model.pois.values(), self.pois) ] ) + '\n'
      s += 'NPs  : ' + '\n       ' .join( [ '%-12s = %8.4f (unscaled : %12.4f)' % (p.name,v, self.unscaled(p.name)) for p, v in zip(self.model.nps .values(), self.nps ) ] )
    return s

  def __contains__(self, par : str) -> bool :
    """Tests if a parameter is present

      Args:
        par : name of a parameter (either POI or NP)

      Returns:
        True if a parameter of this name is present, False otherwise
    """
    if self.model is None : raise ValueError('Cannot perform operation without a model.')
    return par in self.model.pois or par in self.model.nps

  def __getitem__(self, par : str) -> float :
    """Implement [] lookup of POI and NP names

      Args:
        par : name of a parameter (either POI or NP)

      Returns:
        The value of the parameter
    """
    if self.model is None : raise ValueError('Cannot perform operation without a model.')
    if par in self.model.pois : return self.pois[list(self.model.pois).index(par)]
    if par in self.model.nps  : return self.nps [list(self.model.nps ).index(par)]
    raise KeyError('Model parameter %s not found' % par)


  def set_pois(self, pois : np.ndarray) -> 'Parameters' :
    """Set the POI array

      Args:
        pois : array of new POI values
      Returns:
        self
    """
    if pois.shape != self.pois.shape : raise ValueError('Cannot set POI array %s to %s.' % (str(self.pois), str(pois)))
    self.pois = np.array(pois)
    return self

  def set_nps(self, nps : np.ndarray) -> 'Parameters' :
    """Set the NP array

      Args:
        nps : array of new NP values
      Returns:
        self
    """
    if nps.shape != self.nps.shape : raise ValueError('Cannot set NP array %s to %s.' % (str(self.nps), str(nps)))
    self.nps = np.array(nps)
    return self

  def set(self, par : str, val : float, unscaled : bool = False) -> 'Parameters' :
    """Set the value of a parameter (POI or NP)

      Args:
        par : name of a parameter (either POI or NP)
        val : parameter value to set
        unscaled : for NPs, interpret `val` as a `scaled` (False) or `unscaled` (True) value,
                   see the description of :class:`fastprof.elements.ModelNP` for details
      Returns:
        self
    """
    if self.model is None : raise ValueError('Cannot perform operation without a model.')
    if par in self.model.pois :
      self.pois[list(self.model.pois).index(par)] = val
      return self
    if par in self.model.nps :
      self.nps[list(self.model.nps).index(par)] = val if not unscaled else self.model.nps[par].scaled_value(val)
      return self
    raise KeyError('Model parameter %s not found' % par)

  def __setitem__(self, par : str, val : float) -> 'Parameters' :
    """Implement setting parameters using pars['parname'] = 3.14 syntax, for both POIs and NPs

      For NPs, `val` is considered to be a `scaled` value (use :meth:`Parameters.set` to
      set `unscaled` values.

      Args:
        par : name of a parameter (either POI or NP)
        val : parameter value to set
      Returns:
        self
    """
    return self.set(par, val)

  def unscaled_nps(self) -> np.array :
    """Returns an np.array of the unscaled values of all NPs

      Returns:
        array of unscaled values of all NPs.
    """
    if self.model is None : raise ValueError('Cannot perform operation without a model.')
    return self.model.np_nominal_values + self.nps*self.model.np_variations

  def unscaled(self, par : str) -> float :
    """Returns the unscaled value of an NP

      Args:
        par : an NP name

      Returns:
        the unscaled value of the NP
    """
    if self.model is None : raise ValueError('Cannot perform operation without a model.')
    if par in self.model.nps : return self.model.nps[par].unscaled_value(self.__getitem__(par))
    raise KeyError('Model nuisance parameter %s not found' % par)

  def dict(self, nominal_nps : bool = False, unscaled_nps : bool = True, pois_only : bool = False) -> dict :
    """Returns a dictionary of parameter name : value pairs

      Args:
        nominal_nps : set NPs to their nominal values
        unscaled_nps : specifies whether to use the `unscaled` (True) or `scaled` (False)
                      value for NPs.
        pois_only : only include POIs

      Returns:
        Dictionary of parameter name : value pairs
    """
    if self.model is None : raise ValueError('Cannot perform operation without a model.')
    if nominal_nps : return Parameters(self.pois, model=self.model).dict(nominal_nps=False, unscaled_nps=unscaled_nps)
    dic = {}
    for poi, val in zip(self.model.pois.keys(), self.pois) : dic[poi] = val
    if pois_only : return dic
    for par, val in zip(self.model.nps .keys(), self.unscaled_nps() if unscaled_nps else self.nps) : dic[par] = val
    return dic

  def partial_dict(self, par_names : list, nominal_nps : bool = False, unscaled_nps : bool = True) :
    if self.model is None : raise ValueError('Cannot perform operation without a model.')
    dic = {}
    for par_name in par_names :
      if par_name in model.poi_indices : dic[par_name] = self.pois[model.poi_indices[par_name]]
      if par_name in model.np_indices : 
        if not nominal_nps :
          dic[par_name] = self.nps[model.poi_indices[par_name]] if not unscaled_nps else self.unscaled()[model.poi_indices[par_name]]
        else :
          dic[par_name] = 0 if not unscaled_nps else self.model.np_nominal_values[model.poi_indices[par_name]]
    return dic

  def set_from_dict(self, dic : dict, unscaled_nps : bool = False) -> 'Parameters' :
    """Set parameter values form a dictionary of parameter name : value pairs

      Args:
        dic : a dictionary containing parameter name : value pairs
        unscaled_nps : specifies whether NP values in `dic` should be
          considered as `unscaled` (True) or `scaled` (False).

      Returns:
        self
    """
    for par, val in dic.items() : self.set(par, val, unscaled_nps)
    return self

  def set_from_aux(self, data : 'Data') -> 'Parameters' :
    """Set NP values to those of auxiliary observables

      Args:
        data : an observed dataset, from which aux. obs. values are taken

      Returns:
        self
    """
    if self.model is None : raise ValueError('Cannot perform operation without a model.')
    self.nps[self.model.ncons:] = data.aux_obs[self.model.ncons:]
    return self


# -------------------------------------------------------------------------
class Model (Serializable) :
  """Class representing the statistical model

  The class provides a description of the full statistical model, consisting
  in

  * Measurement regions, described by :class:`fastprof.elements.Channel` objects

  * Model parameters, split into POIs (:class:`fastprof.elements.ModelPOI` objects)
    and NPs (:class:`fastprof.elements.ModelNP` objects).

  The main purpose of the class is to store the inputs to the fast likelihood
  maximization algorithm of :class:`NPMinimizer`. For this purpose, the
  structures provided by the :class:`fastprof.elements.Channel` and :class:`fastprof.elements.Sample` classes are
  flattened into a number of np.array objects. These use a simplified description
  of measurement bins, in which the bins for all the channels are concatenated into
  a single large collection of size `nbins`. The `channel_offsets` attribute provides
  the indices of the first bin of each channel within this larger bin array.
  Other arrays store the expected event yields for each sample, and their variations
  as a function of the NPs.

  The model functionality is mainly accessed through the :meth:`Model.n_exp` method, which
  returns the expected event yield for a given set of parameter values `pars`, and the
  :meth:`Model.nll` method, which return the negative log-likelihood value.

  Attributes:
     pois (dict): the model POIs (as a dict mapping POI name to :class:`fastprof.elements.ModelPOI` object)
     nps  (dict): the model NPs (as a dict mapping NP name to :class:`fastprof.elements.ModelNP` object)
     aux_obs (dict): the model auxiliary observables that constrain the NPs
        (as a dict mapping aux. obs. name to :class:`fastprof.elements.ModelAux` object)
     npois (int): number of model pois
     nnps (int): number of model NPs
     nauxs (int): number of model auxiliary observables
     channels (dict): the model channels (as a dict mapping channel name to :class:`fastprof.elements.Channel` object)
     samples (dict): the the model samples, compiled over all channels (as a dict mapping sample name
        to :class:`fastprof.elements.Sample` object)
     sample_indices (dict): maps sample name to its index in `samples` (which is an ordered dict)
     nbins (int): total number of measurement bins, compiled over channels
     channel_offsets (dict): maps channel name to the index of the first bin for this sample, among the list
       (of size `nbins`) concatenating the measurement bins of each channel.
     nominal_yields (np.array): expected event yields for each sample, as a 2D array of size
       `nsamples` x `nbins`.
     pos_impacts (np.array): array of the per-sample event yield variations for positive NP values, as
        a 3D array of size `nsamples` x `nbins` x `nnps`.
     neg_impacts (np.array): array of the per-sample event yield variations for negative NP values, as
        a 3D array of size `nsamples` x `nbins` x `nnps`.
     sym_impacts (np.array): array of symmetrized per-sample event yield variations, as
        a 3D array of size `nsamples` x `nbins` x `nnps`. The variations are computed as the average of
        the positive and negative impacts.
     log_pos_impacts (np.array): array of the logs of the per-sample event yield variations for positive NP values, as
        a 3D array of size `nsamples` x `nbins` x `nnps`.
     log_neg_impacts (np.array): array of the logs of the per-sample event yield variations for negative NP values, as
        a 3D array of size `nsamples` x `nbins` x `nnps`.
     log_sym_impacts (np.array): array of the logs of the symmetrized per-sample event yield variations, as
        a 3D array of size `nsamples` x `nbins` x `nnps`.
     constraint_hessian (np.array): inverse of the covariance matrix of the NP constraint Gaussian
     np_nominal_values (np.array): nominal values of the unscaled NPs (see the description of :class:`fastprof.elements.ModelNP` for details)
     np_variations (np.array): variations of the unscaled NPs (see the description of :class:`fastprof.elements.ModelNP` for details)
     use_asym_impacts (bool): use asymmetric impact terms when computing yield variation (True, default), or
        use symmetrized impacts instead (False).
     use_linear_nps (bool): compute yield variations using an exponential form (False, default) or a linear
        form (True).
     use_simple_sym_impacts (bool): option to use `sym_impacts` for the linear impact factors (see :meth:`Model.linear_impacts`,
                                    default: True).
     use_lognormal_terms (bool): include the derivatives of the exponential terms in the NP minimization
        procedure (True) or not (False, default).
     cutoff (float): regularization term that caps the relative variations in event yields
  """

  def __init__(self, name : str = '', pois : dict = None, nps : dict = None, aux_obs : dict = None, channels : dict = None,
               use_asym_impacts : bool = True, use_linear_nps : bool = False, use_simple_sym_impacts : bool = True,
               use_lognormal_terms : bool = False, variations : list = None, verbosity : int = 0) :
    """Initialize Model object

      Args:
        pois     : the model POIs, as a dict mapping POI names to :class:`fastprof.elements.ModelPOI` objects
        nps      : the model NPs, as a dict mapping NP names to :class:`fastprof.elements.ModelNP` objects
        aux_obs  : the model aux. obs., as a dict mapping names to :class:`fastprof.elements.ModelAux` objects
        channels : the model channelsm as a dict mapping channel names to :class:`fastprof.elements.Channel` objects
        use_asym_impacts : option to use symmetric or asymmetric NP impacts (see class description, default: True)
        use_linear_nps   : option to use the linear or exp form of NP impact on yields (see class description, default: False)
        use_simple_sym_impacts : option to use `sym_impacts` for the linear impacts (see :meth:`Model.linear_impacts`, default: True)
        use_lognormal_terms : option to include exp derivatives when minimizing nll (see class description, default: False)
        variations : list of NP variations to consider (default: None -- use 1 and the largest available other one)
    """
    super().__init__()
    self.name = name
    self.pois = { poi.name : poi for poi in pois.values() } if pois is not None else {}
    self.nps = {}
    if nps is not None :
      for par in nps.values() :
        if par.aux_obs is not None and par.aux_obs not in aux_obs :
          raise ValueError("Auxiliary observable '%s' for NP '%s' is not defined." % (par.aux_obs, par.name))
        self.nps[par.name] = par
    self.aux_obs = { par.name : par for par in aux_obs.values() } if aux_obs is not None else {}
    self.channels = { channel.name : channel for channel in channels.values() } if channels is not None else {}
    self.use_asym_impacts = use_asym_impacts
    self.use_linear_nps = use_linear_nps
    self.use_simple_sym_impacts = use_simple_sym_impacts
    self.use_lognormal_terms = use_lognormal_terms
    self.cutoff = 0
    self.variations = variations
    self.verbosity = verbosity
    self.set_internal_vars()

  def set_internal_vars(self) :
    """Private method to initialize internal attributes

      The Model class contains both primary atttributes and secondary
      attributes that are pre-computed from the primary ones to speed
      up computations later. The primary->secondary computation is
      performed by this method, which is called from both
      :meth:`Model.__init__` and :meth:`Model.load_dict`.
    """
    self.npois = len(self.pois)
    self.nnps  = len(self.nps)
    self.nauxs = len(self.aux_obs)
    self.poi_indices = {}
    self.np_indices = {}
    self.constraint_hessian = np.zeros((self.nnps, self.nnps))
    self.np_nominal_values = np.array([ par.nominal_value for par in self.nps.values() ], dtype=float)
    self.np_variations     = np.array([ par.variation     for par in self.nps.values() ], dtype=float)
    for p, par in enumerate(self.nps.values()) :
      if par.constraint is not None :
        self.constraint_hessian[p,p] = 1/par.scaled_constraint()**2
      self.np_indices[par.name] = p
    for p, par in enumerate(self.pois.values()) : self.poi_indices[par.name] = p
    self.samples = {}
    self.nbins = 0
    self.nvariations = 1
    self.max_nsamples = 0
    self.channel_offsets = {}
    if len(self.channels) == 0 : return
    for channel in self.channels.values() :
      if len(channel.samples) > self.max_nsamples : self.max_nsamples = len(channel.samples)
      self.channel_offsets[channel.name] = self.nbins
      self.nbins += channel.nbins()
      for s, sample in enumerate(channel.samples.values()) :
        sample.set_np_data(self.nps.values(), variation=1, verbosity=self.verbosity)
        self.samples[(channel.name, s)] = sample
    if self.verbosity > 1 : print('Initializing nominal event yields')
    self.nominal_yields = np.stack([ np.concatenate([ self.samples[(channel.name, s)].nominal_yields if s < len(channel.samples) else np.zeros(channel.nbins()) for channel in self.channels.values()]) for s in range(0, self.max_nsamples) ])      
    if self.use_asym_impacts :    
      self.pos_impact_coeffs = np.zeros((self.max_nsamples, self.nbins, len(self.nps), self.nvariations))
      self.neg_impact_coeffs = np.zeros((self.max_nsamples, self.nbins, len(self.nps), self.nvariations))
    self.sym_impact_coeffs = np.zeros((self.max_nsamples, self.nbins, len(self.nps)))
    for p, par in enumerate(self.nps) :
      if self.verbosity > 0 : 
        sys.stderr.write('\rInitializing impacts for nuisance parameter %d of %d %-80s' % (p+1, self.nnps, '[ ' + par + ' ]'))
      for s in range(0, self.max_nsamples) :
        sym_list = []
        pos_list = []
        neg_list = []
        for channel in self.channels.values() :
          default_cs = np.zeros((len(self.variations) if self.variations is not None else 1, channel.nbins()))
          sym_list.append(self.samples[(channel.name, s)].sym_impact(par) if s < len(channel.samples) else np.zeros(channel.nbins()))
          if self.use_asym_impacts :
            pos_cs, neg_cs = self.samples[(channel.name, s)].impact_coefficients(par, self.variations, is_log=not self.use_linear_nps) if s < len(channel.samples) else (default_cs, default_cs)
            pos_list.append(pos_cs.T)
            neg_list.append(neg_cs.T)
            if pos_cs.shape[0] > self.nvariations or neg_cs.shape[0] > self.nvariations :
              self.nvariations = max(pos_cs.shape[0], neg_cs.shape[0])
              self.pos_impact_coeffs.resize(self.max_nsamples, self.nbins, len(self.nps), self.nvariations)
              self.neg_impact_coeffs.resize(self.max_nsamples, self.nbins, len(self.nps), self.nvariations)
        if self.use_asym_impacts :
          self.pos_impact_coeffs[s, :, p, :pos_cs.shape[0]] = np.concatenate(pos_list)
          self.neg_impact_coeffs[s, :, p, :pos_cs.shape[0]] = np.concatenate(neg_list)
        self.sym_impact_coeffs[s, :, p] = np.concatenate(sym_list)
    if self.verbosity > 0 : sys.stderr.write('\n')
  def poi(self, index : str) -> ModelPOI :
    """Returns a POI object by index

      Args:
         index : the index of the POI
      Returns:
         a POI object
    """
    pois = list(self.pois.values())
    return pois[index] if index < len(pois) else None

  def channel(self, name : str) -> Channel :
    """Returns a channel object by name

      Args:
         name : a channel name
      Returns:
         The channel object of that name
    """
    return self.channels[name] if name in self.channels else None

  def all_pars(self) -> dict :
    """Returns all model parameters

      Returns:
         A dictionary of parameter name : object pairs containing
         all POIs and NPs.
    """
    pars = {}
    for par in self.pois.values() : pars[par.name] = par
    for par in self.nps.values()  : pars[par.name] = par
    return pars

  def set_constraint(self, par : str, val : float) :
    """Set the value of the constraint on a NP

      If `par` is set to `None`, set the constraint on all NPs.
      See the documentation of :class:`fastprof.elements.ModelNP` for more details
      on constraints

      Args:
         par : a NP name
         val : a constraint value
    """
    for par in self.nps :
      if par is None or par.name == par : par.constraint = val
    self.set_internal_vars()

  def k_exp(self, pars : Parameters) -> np.array :
    """Returns the modifier to event yields due to NPs

      The expected event yield is modified by the NPs in a way
      that depends on the modeling options (see the documentation of
      :class:`Model` for details). This function returns a 2D
      np.array with dimensions `nbins` x `nsamples`,
      where each value is the event yield modifier for each sample
      in each bin.

      Args:
         pars : a set of parameter values (only the NP values are used)
      Returns:
         Event yield modifiers
    """
    if self.use_asym_impacts :
      pos_np = np.maximum(pars.nps, 0)
      neg_np = np.minimum(pars.nps, 0)
      if self.nvariations > 1 :
        pos_vdm = np.vander(pos_np, self.nvariations + 1, True)[:,1:] # remove the first column with only 1s
        neg_vdm = np.vander(neg_np, self.nvariations + 1, True)[:,1:] # remove the first column with only 1s
        delta = np.tensordot(self.pos_impact_coeffs, pos_vdm, axes=2) + np.tensordot(self.neg_impact_coeffs, neg_vdm, axes=2)
      else :
        delta = np.tensordot(self.pos_impact_coeffs[:,:,:,0], pos_np, axes=1) + np.tensordot(self.neg_impact_coeffs[:,:,:,0], neg_np, axes=1)
      if self.use_linear_nps :
        return 1 + delta
      else :
        return np.exp(delta)
    else :
      if self.use_linear_nps :
        return 1 + self.sym_impact_coeffs.dot(pars.nps)
      else :
        return np.exp(np.log(1 + self.sym_impact_coeffs).dot(pars.nps))

    
  def n_exp(self, pars : Parameters) -> np.array :
    """Returns the expected event yields for a given parameter value

    The expected yields correspond to the nominal yields for each sample,
    corrected for the overall normalization terms (function of the POIs)
    and the NP impacts (function of the NPs, see :meth:`Model.k_exp`)
    They provided for each sample in each measurement bin, as a 2D
    np.array with dimensions`nbins` x `nsamples`.

      Args:
         pars: a set of parameter values
      Returns:
         Expected event yields per sample per bin
    """
    nnom = np.stack([ np.concatenate([ self.samples[(channel_name, s)].yields(pars.dict(nominal_nps=True)) if s < len(channel.samples) else np.zeros(channel.nbins()) \
                      for channel_name, channel in self.channels.items()]) for s in range(0, self.max_nsamples) ]) 
    k = self.k_exp(pars)
    if self.cutoff == 0 : return nnom*k
    return nnom*(1 + self.cutoff*np.tanh((k-1)/self.cutoff))

  def tot_bin_exp(self, pars, floor = None) -> np.array :
    """Returns the total expected event yields for a given parameter value

      Same as :meth:`Model.n_exp`, except that the yields are summed over
      all samples. They are provided as a 1D np.array of size `nbins`.

      Args:
         pars: a set of parameter values
      Returns:
         Expected event yields per bin
    """
    ntot = self.n_exp(pars).sum(axis=0)
    return ntot if floor is None else np.maximum(ntot, floor)

  def nll(self, pars : Parameters, data : 'Data', offset : bool = True, floor : bool = None, no_constraints : bool = False) -> float :
    """Returns the negative log-likelihood value for a given parameter set and dataset

      If the `offset` argument is `True` (default), the nll is computed relatively
      to the case where all yields are nominal. This leads to smaller nll values,
      which reduces potential floating-point issues. When computing the difference
      of two nll values as in a profile-likelihood ratio computation, the offset
      cancels out in the difference.

      Args:
         pars   : a set of parameter values
         data   : an observed dataset
         offset : if True, use offsetting to reduce floating-point precision issues
         floor  : if a positive number is provided, will check for negative event yields
                  and replace them with the floor value
         no_constraints : omit the penalty terms from the constraint in the computation.
      Returns:
         The negative log-likelihood value
    """
    delta = data.aux_obs - pars.nps
    ntot = self.tot_bin_exp(pars, floor)
    try :
      if not offset :
        result = np.sum(ntot - data.counts*np.log(ntot))
      else :
        nexp0 = self.nominal_yields.sum(axis=0)
        result = np.sum(ntot - nexp0 - data.counts*(np.log(ntot/nexp0)))
      if not no_constraints :
         result += 0.5*np.linalg.multi_dot((delta, self.constraint_hessian, delta))
      if math.isnan(result) : result = math.inf
      return result
    except Exception as inst:
      print('Fast NLL computation failed with the following exception, returning +Inf')
      print(inst)
      return np.Infinity

  def linear_impacts(self, pars : Parameters) -> np.array :
    """Returns the NP impacts used in linear computations

      The NP minimization for linear models assumes by definition that the impact
      of NPs on all bin contents are linear (see package documentation). This
      method provides an exact computation of this, i.e. the dericative of k_exp
      wrt the NP. Since this computation is expensive, it can be replaced by
      just `sym_impacts`, i.e. the linear impact terms at NP=0.

      Args:
         pars : a set of parameter values
      Returns:
         The impact matrix for all samples, bins and NPs
    """
    if self.use_simple_sym_impacts : return self.sym_impact_coeffs
    if np.array_equal(pars.nps, np.zeros(self.nnps)) : return self.sym_impact_coeffs
    pos_nps = np.maximum(pars.nps, 0)
    neg_nps = np.minimum(pars.nps, 0)
    pos_np1 = np.sign(pos_nps)
    neg_np1 = -np.sign(neg_nps)
    nul_nps = (pars.nps == 0)
    impact = self.sym_impact_coeffs*nul_nps
    for i in range(0, self.pos_impact_coeffs.shape[3]) :
      impact += self.pos_impact_coeffs[:,:,:,i]*((i+1)*pos_der) + \
                self.neg_impact_coeffs[:,:,:,i]*((i+1)*neg_der)
      pos_der *= pos_nps
      neg_der *= neg_nps
    if self.use_linear_nps :
      return impact
    else :
      # For the exp case, mutiply by the exponential. Needs a bit of gymnastics since the
      # exp has one less dimension (nnps) which happens to be the last one, while numpy
      # automatically "broadcasts" only leading dimensions.
      return (impact.T*self.k_exp(pars).T).T

  def plot(self, pars : Parameters, data : 'Data' = None, channel_name : str = None, only : list = None, exclude : list = None,
           variations : list = None, residuals : bool = False, canvas : plt.Figure = None, labels : bool = True, stack : bool = False, figsize=(8,6),
           bin_width : float = None, logy : bool = False) :
    """Plot the expected event yields and optionally data as well

      The plot is performed for a single model, which must be of `binned_range` type.
      The event yields are plotted as a histogram, as a function of the channel
      observable.
      The `variations` arg allows to plot yield variations for selected NP values. The
      format is { ('par1', val1), ... } , which will plot the yields for the case where
      NP par1 is set to val1 (while other NPs remain at nominal), etc.

      Args:
         pars       : parameter values for which to compute the expected yields
         data       : observed dataset to plot alongside the expected yields
         channel    : name of the channel to plot. If `None`, plot the first channel.
         exclude    : list of sample names to exclude from the plot
         variations : list of NP variations to plot, as a list of (str, float) pairs
                      providing the NP name and the value to set.
         residuals  : if True,  plot the data-model differences
         canvas     : a matplotlib Figure on which to plot (if None, plt.gca() is used)
         labels     : if True (default), add labels to the legend 
    """
    if not isinstance(only, list)    and only    is not None : only = [ only ]
    if not isinstance(exclude, list) and exclude is not None : exclude = [ exclude ]
    if channel_name is None : channel_name = list(self.channels.keys())
    if isinstance(channel_name, list) :
      nchan = len(channel_name)
      nrows = int(math.sqrt(nchan))
      ncols = int(nchan/nrows + 0.5)
      fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
      if not isinstance(axs, list) : axs = np.array([ axs ])
      for ax in axs.flatten()[nchan:] : ax.set_visible(False)
      for channel, ax in zip(channel_name, axs.flatten()[:nchan]) :
        self.plot(pars=pars, data=data, channel_name=channel, only=only, exclude=exclude, variations=variations, residuals=residuals, canvas=ax, labels=labels, stack=stack, bin_width=bin_width, logy=logy)
      fig.tight_layout()
      return
      #channel = list(self.channels.values())[0]
      #print("Plotting channel '%s'" % channel.name)
    if not channel_name in self.channels : raise KeyError('ERROR: Channel %s is not defined.' % channel_name)
    channel = self.channels[channel_name]
    if isinstance(channel, BinnedRangeChannel) :
      grid = [ b['lo_edge'] for b in channel.bins ]
      grid.append(channel.bins[-1]['hi_edge'])
    elif isinstance(channel, SingleBinChannel) :
      grid = [0,1]
    elif isinstance(channel, MultiBinChannel) :
      grid = [0, channel.nbins()]
    else :
      raise ValueError("Channel '%s' is of an unsupported type" % channel.name)
    if canvas is None : canvas = plt.figure(figsize=figsize)
    if logy: canvas.set_yscale('log')
    xvals = [ (grid[i] + grid[i+1])/2 for i in range(0, len(grid) - 1) ]
    start = self.channel_offsets[channel.name]
    stop  = start + channel.nbins()
    nexp = self.n_exp(pars)[:, start:stop]
    tot_exp = nexp.sum(axis=0)
    if only is not None :
      samples = []
      for sample_name in only :
        if not sample_name in channel.samples : raise ValueError('Sample %s is not defined.' % sample_name)
        samples.append(list(channel.samples).index(sample_name))
      subtract = nexp[samples,:].sum(axis=0)
      subtract = tot_exp - subtract
      line_style = '--'
      title = ','.join(only)
    elif exclude is not None :
      samples = []
      for sample_name in exclude :
        if not sample_name in channel.samples : raise ValueError('Sample %s is not defined.' % sample_name)
        samples.append(list(channel.samples).index(sample_name))
      subtract = nexp[samples,:].sum(axis=0)
      line_style = '--'
      title = 'Model excluding ' + ','.join(exclude)
    else :
      subtract = np.zeros(nexp.shape[1])
      line_style = '-'
      title = 'Full model'
      samples = range(0, len(channel.samples))
    yvals = tot_exp - subtract if not residuals or data is None else tot_exp - subtract - counts
    if isinstance(channel, BinnedRangeChannel) and bin_width is not None :
      bin_corrs = np.array([ bin_width/(b['hi_edge'] - b['lo_edge']) for b in channel.bins ])
      yvals *= bin_corrs
    canvas.hist(xvals, weights=yvals, bins=grid, histtype='step',color='b', linestyle=line_style, label=title if labels else None)
    if stack :
      for sample in samples :
        stack_exp = nexp[sample:,:].sum(axis=0)
        yvals = stack_exp - subtract if not residuals or data is None else stack_exp - subtract - counts
        if isinstance(channel, BinnedRangeChannel)  and bin_width is not None : yvals *= bin_corrs
        canvas.hist(xvals, weights=yvals, bins=grid, histtype='step', linestyle=line_style, fill=True, label=list(channel.samples)[sample])
    if data is not None :
      counts = np.array(data.counts[start:stop])
      yerrs = [ math.sqrt(n) if n > 0 else 0 for n in counts ]
      yvals = counts if not residuals else np.zeros(channel.nbins())
      if isinstance(channel, BinnedRangeChannel) and bin_width is not None : 
        yvals *= bin_corrs
        yerrs *= bin_corrs
      canvas.errorbar(xvals, yvals, xerr=[0]*channel.nbins(), yerr=yerrs, fmt='ko', label='Data' if labels else None, zorder=99)
    canvas.set_xlim(grid[0], grid[-1])
    if variations is not None :
      for v in variations :
        vpars = pars.clone()
        vpars.set(v[0], v[1])
        col = 'r' if len(v) < 3 else v[2]
        nexp = self.n_exp(vpars)[:, start:stop]
        if only is None and exclude is None :
          subtract = np.zeros(nexp.shape[1])
        else :
          subtract = nexp[samples,:].sum(axis=0)
          if only is not None : subtract = nexp.sum(axis=0) - subtract
        tot_exp = nexp.sum(axis=0) - subtract
        if isinstance(channel, BinnedRangeChannel) and bin_width is not None : tot_exp *= bin_corrs
        canvas.hist(xvals, weights=tot_exp, bins=grid, histtype='step',color=col, linestyle=line_style, label='%s=%+g' %(v[0], v[1]) if labels else None)
    if labels : canvas.legend().set_zorder(100)
    canvas.set_title(channel.name)
    if isinstance(channel, BinnedRangeChannel) :
      canvas.set_xlabel(channel.obs_name + ((' ['  + channel.obs_unit + ']') if channel.obs_unit != '' else ''))
      canvas.set_ylabel('Events / %g %s ' % (bin_width, channel.obs_unit) if bin_width is not None else 'Events')
    elif isinstance(channel, SingleBinChannel) :
      canvas.tick_params(axis='x', which='both', bottom=False, labelbottom=False) # remove x ticks and labels 
      #canvas.set_xlabel(channel.name)
      canvas.set_ylabel('Events')

    #plt.bar(np.linspace(0,self.sig.size - 1,self.sig.size), self.n_exp(pars), width=1, edgecolor='b', color='', linestyle='dashed')

  def gradient(self, pars : Parameters, data : 'Data') -> np.ndarray :
    """Returns the derivatives of the negative log-likelihood wrt the POIs

      Output format: 1D np.ndarray of size `npois`.

      Args:
         pars : parameter values at which to compute the likelihood
         data : observed dataset for which to compute the likelihood
      Returns:
         Values of the derivatives of the negative log-likelihood wrt the POIs.
    """
    try :
      dtot = (self.nominal_yields.T*np.array([ sample.norm_gradient(pars.dict(nominal_nps=True)) for sample in self.samples.values() ], dtype=float)).T.sum(axis=0)
      ntot = self.tot_bin_exp(pars)
      return np.sum(dtot - data.counts*dtot/ntot)
    except:
      return None

  def hessian(self, pars : Parameters, data : 'Data') -> np.ndarray :
    """Returns the Hessian matrix of the negative log-likelihood wrt the POIs

      Output format: 2D np.ndarray of size `npois` x `npois`.
      TODO : update to the new POI scheme, code below is obsolete

      Args:
         pars : parameter values at which to compute the likelihood
         data : observed dataset for which to compute the likelihood
      Returns:
         Hessian matrix of the negative log-likelihood wrt the POIs.
    """
    sexp = self.s_exp(pars) # This is for a "mu" POI, i.e. d(S_i)/dmu = S_i^exp (true for S_i = mu S_i^exp).
    nexp = sexp + self.b_exp(pars) # This is for a "mu" POI, i.e. d(S_i)/dmu = S_i^exp (true for S_i = mu S_i^exp).
    s = 0
    for i in range(0, sexp.size) : s += sexp[i]*data.n[i]/nexp[i]**2
    return s

  def expected_pars(self, pois : dict, minimizer : 'NPMinimizer' = None) -> Parameters :
    """Assigns NP values to a set of POI values

      By default, returns a :class:`Parameters` object with the POI values
      defined by the `pois` arg, and the NPs set to 0. If a dataset is 
      provided, will set the NPs to their profiled values.
      The `pois` arg can also be a class:`Parameters` object, from which
      the POI values will be taken (and the NP values ignored).

      Args:
         pois : A dict of POI name : value pairs, or a class:`Parameters` object.
         data (optional) : dataset to use for the profiling
      Returns:
         Object containing the POI values and associated NP values.
    """
    if isinstance(pois, Parameters) :
      pars = pois
    else :
      pars = Parameters(pois, model=self)
    if minimizer is not None :
      return minimizer.profile(pars)
    else :
      return pars

  def generate_data(self, pars : Parameters) -> 'Data' :
    """Generate a pseudo-dataset for given parameter values

      Returns a randomly-generated dataset for the provided
      parameter values. This includes observed bin contents
      for all channels, generated from Poisson distributions,
      and aux. obs. values generated from the NP constraints.

      Args:
         pars : a set of model parameter values
      Returns:
         A randomly-generated dataset
    """
    return Data(self, np.random.poisson(self.tot_bin_exp(pars)), [ par.generate_aux(pars[par.name]) for par in self.nps.values() ])

  def generate_asimov(self, pars : Parameters) -> 'Data' :
    """Generate an Asimov dataset for given parameter values

      Returns an Asimov dataset for the provided parameter
      values, i.e. a dataset in which the obseved bin counts exactly
      match the expected yields, and the aux. obs. match the
      NP values
      See `arXiv:1007.1727 <https://arxiv.org/abs/1007.1727>`_

      Args:
         pars : a set of model parameter values
      Returns:
         An Asimov dataset
    """
    return Data(self).set_data(self.tot_bin_exp(pars), pars.nps)

  def generate_expected(self, pois, minimizer = None) :
    """Generate an Asimov dataset for expected parameter values

      Same functionality as :meth:`Model.generate_asimov`, but with
      NP values that are obtained from the provided POI values in the
      same way as described for :meth:`Model.expected_pars`.

      Args:
         pois : A dict of POI name : value pairs, or a class:`Parameters` object.
         minimizer (optional) : NP minimizer algorithm used to compute NP profile values
         data (optional) : dataset to use for the profiling
      Returns:
         An Asimov dataset
    """
    return self.generate_asimov(self.expected_pars(pois, minimizer))

  @staticmethod
  def create(filename : str, verbosity : int = 0, flavor : str = None) -> 'Model' :
    """Shortcut method to instantiate a model from a markup file

      Same behavior as creating a default model and loading from the file,
      rolled into a single command

      Args:
         filename : name of a markup file containing the model definition
         verbosity: level of verbosity (0=minimal)
         flavor   : input markup flavor (currently supported: 'json' [default], 'yaml')
      Returns:
         The created model
    """
    return Model(verbosity=verbosity).load(filename, flavor=flavor)

  def load_dict(self, sdict : dict) -> 'Model' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    self.name = self.load_field('name', sdict, '', str)
    if not 'model'    in sdict : raise KeyError("No 'model' section in specified markup file")
    if not 'POIs'     in sdict['model'] : raise KeyError("No 'POIs' section in specified markup file")
    if not 'channels' in sdict['model'] : raise KeyError("No 'channels' section in specified markup file")
    if self.verbosity > 1 : print('Loading parameters')
    self.pois = {}
    for dict_poi in sdict['model']['POIs'] :
      poi = ModelPOI()
      poi.load_dict(dict_poi)
      if poi.name in self.pois :
        raise ValueError('ERROR: multiple POIs defined with the same name (%s)' % poi.name)
      self.pois[poi.name] = poi
    self.aux_obs = {}
    if 'aux_obs' in sdict['model'] :
      for dict_aux in sdict['model']['aux_obs'] :
        par = ModelAux()
        par.load_dict(dict_aux)
        if par.name in self.aux_obs :
          raise ValueError('ERROR: multiple auxiliary observables defined with the same name (%s)' % par.name)
        self.aux_obs[par.name] = par
    self.nps = {}
    if 'NPs' in sdict['model'] :
      for dict_np in sdict['model']['NPs'] :
        par = ModelNP()
        par.load_dict(dict_np)
        if par.aux_obs is not None and not par.aux_obs in self.aux_obs :
          self.aux_obs[par.aux_obs] = ModelAux(name=par.aux_obs, unit=par.unit)
        if par.name in self.nps :
          raise ValueError('ERROR: multiple NPs defined with the same name (%s)' % par.name)
        self.nps[par.name] = par
    self.channels = {}
    if self.verbosity > 1 : print('Loading channels')
    for dict_channel in sdict['model']['channels'] :
      if not 'type' in dict_channel or dict_channel['type'] == SingleBinChannel.type_str :
        channel = SingleBinChannel()
      elif dict_channel['type'] == BinnedRangeChannel.type_str :
        channel = BinnedRangeChannel()
      channel.load_dict(dict_channel)
      if channel.name in self.channels :
        raise ValueError('ERROR: multiple channels defined with the same name (%s)' % channel.name)
      self.channels[channel.name] = channel
    self.set_internal_vars()
    return self

  def fill_dict(self, sdict : dict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    sdict['model'] = {}
    sdict['model']['name'] = self.name
    sdict['model']['POIs'] = []
    sdict['model']['NPs'] = []
    sdict['model']['aux_obs'] = []
    sdict['model']['channels'] = []
    for poi in self.pois.values()    : sdict['model']['POIs']   .append(poi.dump_dict())
    for par in self.nps.values()     : sdict['model']['NPs']    .append(par.dump_dict())
    for aux in self.aux_obs.values() : sdict['model']['aux_obs'].append(aux.dump_dict())
    for channel in self.channels.values() : sdict['model']['channels'].append(channel.dump_dict())

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    s = 'POIs :'
    for poi in self.pois.values() : s += '\n  - %s' % str(poi)
    s += '\nNPs :'
    for par in self.nps.values() : s += '\n  - %s' % str(par)
    s += '\nChannels :'
    for channel in self.channels.values() : s += '\n  - %s' % str(channel)
    return s


# -------------------------------------------------------------------------
class Data (Serializable) :
  """Class representing a set of observed data

  Stores a complete dataset:

  * A list of observed bin yields

  * Observed values for the auxiliary observables.

  Both are stored as a single np.array apiece. The bins yields use the concatenated bin
  list defined in :class:`Model`, while the aux. obs values are given in the same order
  as their appearance in the `aux_obs` attribute of the model.

  Only numerical values are stored locally, but the names and properties
  of the bins and aux. obs. are accessible through the `model` attribute, if it set.

  Attributes:
     counts (np.ndarray): the observed bin yields
     aux_obs (np.ndarray): the observed aux. obs. values
     model (Model): pointer to a :class:`Model` object containing the full model.
  """
  def __init__(self, model : Model, counts : np.ndarray = None, aux_obs : np.ndarray = None) :
    """Initialize the object

      Args:
         counts  : observed bin counts
         aux_obs : aux. obs. values
    """
    super().__init__()
    self.model = model
    self.set_counts(counts if counts is not None else [])
    self.set_aux_obs(aux_obs if aux_obs is not None else [])

  def set_counts(self, counts) -> 'Data' :
    """Sets the observed bin counts to the specified values

      Args:
         counts  : observed bin counts to set to
      Returns:
         self
    """
    if isinstance(counts, list) : counts = np.array( counts, dtype=float )
    if not isinstance(counts, np.ndarray) : counts = np.array([ counts ], dtype=float)
    if counts.size > 0 :
      if counts.ndim != 1 :
        raise ValueError('Input data counts should be a 1D vector, got ' + str(counts))
      if counts.size != self.model.nbins :
        raise ValueError('Input data counts should have a size equal to the number of model bins (%d), got %d.' % (model.nbins, len(counts)))
      self.counts = np.array(counts)
    else :
      self.counts = np.zeros(self.model.nbins)
    return self

  def set_aux_obs(self, aux_obs = []) :
    """Sets the aux. obs. to the specified values

      Args:
         aux_obs : aux. obs. values to set to
      Returns:
         self
    """
    if isinstance(aux_obs, list) : aux_obs = np.array( aux_obs, dtype=float )
    if not isinstance(aux_obs, np.ndarray) : aux_obs = np.array([ aux_obs ], dtype=float) # to ensure we have floats everywhere ?
    if aux_obs.ndim != 1 :
      raise ValueError('Input aux data should be a 1D vector, got ' + str(aux_obs))
    if len(aux_obs) == 0 :
      self.aux_obs = np.array(self.model.np_nominal_values)
    elif len(aux_obs) == self.model.nnps :
      self.aux_obs = np.array(aux_obs)
    elif len(aux_obs) == self.model.nauxs :
      self.aux_obs = np.array(self.model.np_nominal_values)
      a = 0
      for p, par in enumerate(self.model.nps.values()) :
        if par.is_free() : continue
        self.aux_obs[p] = aux_obs[a]
        a += 1
    else :
      raise ValueError('Cannot set aux obs data from an array of size %d, expecting a size of either %d (aux obs only) or %d (all NPs)' % (len(aux_obs), self.model.nauxs, self.model.nnps))
    return self

  def set_data(self, counts, aux_obs) :
    """Sets both the observed bin counts and aux. obs. to the specified values

      Args:
         counts  : observed bin counts to set to
         aux_obs : aux. obs. values to set to
      Returns:
         self
    """
    self.set_counts(counts)
    self.set_aux_obs(aux_obs)
    return self

  def load_dict(self, sdict : dict) -> 'Data' :
    """Load object information from a dictionary of markup data

      Args:
        sdict : A dictionary containing markup data

      Returns:
        self
    """
    if not 'data' in sdict : raise KeyError("No 'data' section in specified markup file")
    if not 'channels' in sdict['data'] : raise KeyError("No 'channels' section in specified markup file")
    offset = 0
    for model_channel in self.model.channels.values() :
      name = model_channel.name
      try :
        channel = next(dict_channel for dict_channel in sdict['data']['channels'] if dict_channel['name'] == name)
      except:
        raise ValueError("Model channel '%s' not found in specified markup file." % name)
      model_channel.load_data_dict(channel, self.counts[offset:offset + model_channel.nbins()])
      offset += model_channel.nbins()
    if 'aux_obs' in sdict['data'] :
      data_aux_obs = { aux_obs['name'] : aux_obs['value'] for aux_obs in sdict['data']['aux_obs'] }
    else :
      data_aux_obs = {}
    aux_obs_values = []
    for par in self.model.nps.values() :
      if par.aux_obs is None :
        aux_obs_values.append(0)
        continue
      if not par.aux_obs in data_aux_obs : raise('Auxiliary observable %s defined in model, but not provided in the data' % par.aux_obs)
      aux_obs_values.append(par.scaled_value(data_aux_obs[par.aux_obs]))
    self.set_aux_obs(np.array(aux_obs_values, dtype=float))
    return self

  def fill_dict(self, sdict : dict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    sdict['data'] = {}
    sdict['data']['channels'] = []
    for channel_name, channel in self.model.channels.items() :
      channel_data = {}
      offset = self.model.channel_offsets[channel_name]
      channel.save_data_dict(channel_data, self.counts[offset:offset + channel.nbins()])
      sdict['data']['channels'].append(channel_data)
    sdict['data']['aux_obs'] = []
    for p, par in enumerate(self.model.nps.values()) :
      if par.aux_obs is None : continue
      aux_data = { 'name' : par.aux_obs, 'value' : par.unscaled_value(self.aux_obs[p]) }
      sdict['data']['aux_obs'].append(aux_data)

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    s = ''
    s += 'counts  = ' + str(self.counts)  + '\n'
    s += 'aux_obs = ' + str(self.aux_obs) + '\n'
    return s
  
