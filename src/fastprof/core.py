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

  * :class:`Parameters` : a class storing a set of values for model POIs and NPs.

  * :class:`Data` : a class storing the observed data: the observed bin yields for each channel and the auxiliary observables for each NP.

  All the classes support loading from / saving to markup files. The basic mechanism for this is implemented in the :class:`fastprof.elements.Serializable` base class from which they all derive.
"""

import numpy as np
import math, sys
import matplotlib.pyplot as plt

from .base import ModelPOI, ModelNP, ModelAux, Serializable
from .sample import Sample
from .channels import Channel, SingleBinChannel, MultiBinChannel, BinnedRangeChannel, GaussianChannel
from .expressions import Expression, SingleParameter

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

      * An np.ndarray of POI values, with parameter values provided in the
        same order as they appear in the model.

      * A dict of POI name : value pairs with one entry for each model POI.

      * A single number, for a model with a single POI

      NPs are optional; they can be provided as a single number or a np.ndarray
      as for POIS, and will otherwise default to 0.
      
      In all cases, the inputs are copied locally.

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
    """Return a clone of a Parameters object

      Performs a deep-copy operation at the required level: deep-copy
      the np.ndarray objects, but shallow-copy the model pointer

      Returns:
        the new clone
    """
    return Parameters(np.array(self.pois, dtype=float), np.array(self.nps, dtype=float), self.model)

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        the object description
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

      Returns *scaled* values of the NPs.

      Args:
        par : name of a parameter (either POI or NP)

      Returns:
        the value of the parameter
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
        unscaled : for NPs, interpret `val` as a *scaled* (False) or *unscaled* (True) value,
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

      For NPs, `val` is considered to be a *scaled* value (use :meth:`Parameters.set` to
      set *unscaled* values.

      Args:
        par : name of a parameter (either POI or NP)
        val : parameter value to set
      Returns:
        self
    """
    return self.set(par, val)

  def unscaled_nps(self) -> np.ndarray :
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
        unscaled_nps : specifies whether to use the *unscaled* (True) or *scaled* (False)
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
          considered as *unscaled* (True) or *scaled* (False).

      Returns:
        self
    """
    if self.model is None : raise ValueError('Cannot perform operation without a model.')
    for par in self.model.pois : self.set(par, dic[par], unscaled_nps)
    for par in self.model.nps : self.set(par, dic[par], unscaled_nps)
    #for par, val in dic.items() : self.set(par, val, unscaled_nps)
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
  in:

  * Measurement regions, described by :class:`fastprof.channels.Channel` objects

  * Model parameters, split into POIs (:class:`fastprof.base.ModelPOI` objects)
    and NPs (:class:`fastprof.base.ModelNP` objects).

  The main purpose of the class is to store the inputs to the fast likelihood
  maximization algorithm of :class:`NPMinimizer`. For this purpose, the
  structures provided by the :class:`fastprof.channels.Channel` and :class:`fastprof.samples.Sample` classes are
  flattened into a number of np.array objects. These use a simplified description
  of measurement bins, in which the bins for all the channels are concatenated into
  a single large collection of size `nbins`. The `channel_offsets` attribute provides
  the indices of the first bin of each channel within this larger bin array.
  Other arrays store the expected event yields for each sample, and their variations
  as a function of the NPs.
  
  The model depends on the POI through the normalization factors of each channel sample.
  These factors can be a single POI, or an `expressions` representing a function of one or more POIs.
  These expressions derive from :class:`fastprof.expressions.Expression`.
  
  The model functionality is mainly accessed through the :meth:`Model.n_exp` method, which
  returns the expected event yield for a given set of parameter values `pars`, and the
  :meth:`Model.nll` method, which return the negative log-likelihood value.

  Attributes:
     name (str) : the name of the model
     pois (dict): the model POIs (as a dict mapping POI name to :class:`fastprof.base.ModelPOI` object)
     nps  (dict): the model NPs (as a dict mapping NP name to :class:`fastprof.base.ModelNP` object)
     aux_obs (dict): the model auxiliary observables that constrain the NPs
        (as a dict mapping aux. obs. name to :class:`fastprof.base.ModelAux` object)
     npois (int): number of model pois
     nnps (int): number of model NPs
     nauxs (int): number of model auxiliary observables
     channels (dict): the model channels (as a dict mapping channel name to :class:`fastprof.channels.Channel` object)
     samples (dict): the the model samples, compiled over all channels (as a dict mapping (sample name, sample index in channel)
        to :class:`fastprof.sample.Sample` object)
     max_nsamples (int) : largest number of samples among the channels. This defines the `nsamples` dimension
       of the objects below, e.g. NP impact matrices.
     expressions (dict) : the POI expressions (as a dict mapping expression name to :class:`fastprof.expressions.Expression` object)
     nbins (int): total number of measurement bins, summing over all channels
     channel_offsets (dict): maps channel name to the index of the first bin of this channel in the overall bin list.
       (of size `nbins`).
     nominal_yields (np.array): expected event yields for each sample, as a 2D array of size
       `nsamples` x `nbins`.
     pos_impact_coeffs (np.array): array of the per-sample event yield variations for positive NP values, as
        a 3D array of size `nsamples` x `nbins` x `nnps`.
     neg_impact_coeffs (np.array): array of the per-sample event yield variations for negative NP values, as
        a 3D array of size `nsamples` x `nbins` x `nnps`.
     sym_impact_coeffs (np.array): array of symmetrized per-sample event yield variations, as
        a 3D array of size `nsamples` x `nbins` x `nnps`. The variations are computed as the average of
        the positive and negative impacts.
     constraint_hessian (np.array): inverse of the covariance matrix of the NP constraint Gaussian
     np_nominal_values (np.array): nominal values of the unscaled NPs (see the description of :class:`fastprof.elements.ModelNP` for details)
     ref_pars (Parameters) : reference values of the parameters : initial value of the POIs and nominal value of the NPs
     ref_yields (np.ndarray) : Bin yields computed at the reference values of the parameters specified in `ref_pars`.
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
     variations (list) : list of NP variations to use (default: [-1, 1])
     verbosity (int) : verbosity level of the output
     poi_indices (dict) : dict mapping POI name to POI index in the np.ndarrays used in the :class:`Parameters` class.
     np_indices (dict) : dict mapping NP name to NP index in the np.ndarrays used in the :class:`Parameters` class.
     poisson_channels (dict) : channels of *Poisson* type (everything but :class:`fastprof.channels.GaussianChannel` classes now)
     gaussian_channels (dict) : channels of *Gaussian* type (of class :class:`fastprof.channels.GaussianChannel`)
     nbins_poisson (int) : number of bins in the Poisson-type channels (coming first in the channel list)
     nbins_gaussian (int) : number of bins in the Gaussian-type channels (coming last in the channel list)
     poi_hessian (np.ndarray) : overall Hessian of the Gaussian-type channels
  """

  def __init__(self, name : str = '', pois : dict = None, nps : dict = None, aux_obs : dict = None, channels : dict = None,
               expressions : dict = None, use_asym_impacts : bool = True, use_linear_nps : bool = False,
               use_simple_sym_impacts : bool = True, use_lognormal_terms : bool = False, variations : list = None,
               verbosity : int = 0) :
    """Initialize a Model object

      Args:
        pois     : the model POIs, as a dict mapping POI names to :class:`fastprof.elements.ModelPOI` objects
        nps      : the model NPs, as a dict mapping NP names to :class:`fastprof.elements.ModelNP` objects
        aux_obs  : the model aux. obs., as a dict mapping names to :class:`fastprof.elements.ModelAux` objects
        channels : the model channels, as a dict mapping channel names to :class:`fastprof.elements.Channel` objects
        expressions      : the model expressions as a dict mapping names to :class:`fastprof.elements.Expression` objects
        use_asym_impacts : option to use assymmetric (True) or symmetric (False) NP impacts (see class description, default: True)
        use_linear_nps   : option to use the linear (True) or exp (False) form of NP impact on yields (see class description, default: False)
        use_simple_sym_impacts : option to use `sym_impacts` for the linear impacts (see :meth:`Model.linear_impacts`, default: True)
        use_lognormal_terms : option to include exp derivatives when minimizing nll (see class description, default: False)
        variations : list of NP variations to consider (default: None -- use 1 and the largest available other one)
        verbosity : verbosity level of the output
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
    self.expressions = { expr.name : expr for expr in expressions.values() } if expressions is not None else {}
    self.use_asym_impacts = use_asym_impacts
    self.use_linear_nps = use_linear_nps
    self.use_simple_sym_impacts = use_simple_sym_impacts
    self.use_lognormal_terms = use_lognormal_terms
    self.cutoff = 0
    self.variations = variations
    self.verbosity = verbosity
    self.set_internal_vars()

  def clone(self, name : str = None, set_internal_vars : bool = True) :
    """Return a clone of the model

      All internal model information (arrays, expressionschannels, samples) are deep-copied
      in the cloning to ensure that the clone is independent of the original.
      Information on the POIs, NPs and Auxs is not clone as this information
      should be the same for all models.
      All model options are set to the same as in the original model.

      Args:
        name : the name of the new model
        set_internal_vars : if True, run the :meth:`Model.set_internal_vars` function
    """
    clone = Model(name=name if name is not None else self.name, pois=self.pois, nps=self.nps, aux_obs=self.aux_obs,
                  channels={ channel.name : channel.clone() for channel in self.channels.values() },
                  expressions=self.expressions, use_asym_impacts=self.use_asym_impacts, use_linear_nps=self.use_linear_nps,
                  use_simple_sym_impacts=self.use_simple_sym_impacts, use_lognormal_terms=self.use_lognormal_terms, variations=self.variations,
                  verbosity=self.verbosity)
    if set_internal_vars : clone.set_internal_vars()
    return clone

  def set_internal_vars(self) :
    """Initialize internal storage

      The Model class contains both primary atttributes set in __init__ 
      but also secondary attributes that are pre-computed from the 
      primary ones to speed up computations later. The primary->secondary
      computation is performed by this method, which is called from both
      :meth:`Model.__init__` and :meth:`Model.load_dict`.
    """
    self.npois = len(self.pois)
    self.nnps  = len(self.nps)
    self.nauxs = len(self.aux_obs)
    self.poi_indices = {}
    self.np_indices = {}
    self.constraint_hessian = np.zeros((self.nnps, self.nnps))
    poi_initial_values = np.array([ poi.initial_value for poi in self.pois.values() ], dtype=float)
    self.np_nominal_values  = np.array([ par.nominal_value for par in self.nps.values() ], dtype=float)
    self.np_variations      = np.array([ par.variation     for par in self.nps.values() ], dtype=float)
    self.ref_pars = Parameters(poi_initial_values, np.zeros(len(self.nps)), self)
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
    poisson_channels  = { channel.name : channel for channel in self.channels.values() if channel.pdf_type() == 'poisson' }
    gaussian_channels = { channel.name : channel for channel in self.channels.values() if channel.pdf_type() == 'gaussian' }
    channels = { **poisson_channels,  **gaussian_channels }
    self.nbins_poisson  = sum([ channel.nbins() for channel in  poisson_channels.values() ])
    self.nbins_gaussian = sum([ channel.nbins() for channel in gaussian_channels.values() ])
    self.poi_hessian = np.zeros((self.nbins_gaussian, self.nbins_gaussian))
    if list(channels.keys()) != list(self.channels.keys()) :
      if self.verbose > 0 : print('Warning: reordering channels to put Gaussians at the end')
      self.channels = channels
    reals = self.reals()
    for channel in self.channels.values() :
      if len(channel.samples) > self.max_nsamples : self.max_nsamples = len(channel.samples)
      self.channel_offsets[channel.name] = self.nbins
      if channel.pdf_type() == 'gaussian' :
        offset = self.nbins - self.nbins_poisson
        self.poi_hessian[offset:offset + channel.nbins(), offset:offset + channel.nbins()] = channel.hessian
      self.nbins += channel.nbins()
      for s, sample in enumerate(channel.samples.values()) :
        sample.set_np_data(self.nps.values(), reals, self.real_vals(self.ref_pars), variation=1, verbosity=self.verbosity)
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
        self.sym_impact_coeffs[s, :, p] = np.concatenate(sym_list) # TODO: check here that there is some non-zero impact, can indicate trouble for later
    if self.verbosity > 0 : sys.stderr.write('\n')
    self.ref_yields = self.n_exp(self.ref_pars)

  def poi(self, index : int) -> ModelPOI :
    """Return a POI object by index

      Args:
         index : the index of the POI
      Returns:
         the POI object, or `None` if `index` is invalid
    """
    pois = list(self.pois.values())
    return pois[index] if index < len(pois) else None

  def channel(self, name : str) -> Channel :
    """Returns a channel object by name

      Args:
         name : a channel name
      Returns:
         The channel object of that name, or `None` if not found
    """
    return self.channels[name] if name in self.channels else None

  def expression(self, name : str) -> Expression :
    """Return an expression object by name

      Args:
         name : an expression name
      Returns:
         The expression object of that name, or `None` if not found
    """
    return self.expressions[name] if name in self.expressions else None

  def all_pars(self) -> dict :
    """Return all model parameters

      Returns:
         A dictionary of parameter name : object pairs containing
         all POIs and NPs.
    """
    pars = {}
    for par in self.pois.values() : pars[par.name] = par
    for par in self.nps.values()  : pars[par.name] = par
    return pars
  
  def reals(self) -> dict :
    """Return all model real values

      Returns all the real values in the model, i.e. POIs, NPs and expressions.
      These are in turns the possible inputs to other expressions.

      Returns:
         A dictionary of parameter name : object pairs containing
         all POIs, NPs and expressions
    """
    return { **{ poi : SingleParameter(poi) for poi in self.pois}, **{ par : SingleParameter(par) for par in self.nps}, **self.expressions }


  def set_constraint(self, par : str, val : float) :
    """Set the value of the constraint on a NP

      If `par` is set to `None`, set the constraint on all NPs.
      See the documentation of :class:`fastprof.base.ModelNP` for more details
      on constraints

      Args:
         par : a NP name
         val : a constraint value
    """
    for par in self.nps :
      if par is None or par.name == par : par.constraint = val
    self.set_internal_vars()

  def k_exp(self, pars : Parameters) -> np.array :
    """Return the modifier to event yields due to NPs

      The expected event yield is modified by the NPs in a way
      that depends on the modeling options (see the documentation of
      :class:`Model` for details). This function returns a 2D
      np.array with dimensions `nbins` x `nsamples`,
      where each value is the event yield modifier for each sample
      in each bin.

      Args:
         pars : a set of parameter values (only the NP values are used)
      Returns:
         The event yield modifiers
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
      if self.verbosity > 2 :
        print('== k_exp eval:')
        print('delta = ', delta)
        if self.verbosity > 3 :
          print('pos_impact_coeffs = ', self.pos_impact_coeffs[:,:,:,0])
          print('neg_impact_coeffs = ', self.neg_impact_coeffs[:,:,:,0])
          print('pos_np = ', pos_np)
          print('neg_np = ', neg_np)
      if self.use_linear_nps :
        return 1 + delta
      else :
        return np.exp(delta)
    else :
      if self.verbosity > 2 :
        print('== k_exp eval [sym impacts]:')
        print('delta = ', self.sym_impact_coeffs.dot(pars.nps))
        if self.verbosity > 3 :
          print('sym_impact_coeffs = ', self.sym_impact_coeffs)
          print('pars.nps = ', pars.nps)
      if self.use_linear_nps :
        return 1 + self.sym_impact_coeffs.dot(pars.nps)
      else :
        return np.exp(np.log(1 + self.sym_impact_coeffs).dot(pars.nps))

  def cut_k_exp(self, pars) :
    k = self.k_exp(pars)
    return k if self.cutoff == 0 else (1 + self.cutoff*np.tanh((k-1)/self.cutoff))

  def real_vals(self, pars : Parameters) -> dict :
    """Return a full dictionary of parameter values, including expressions

       Parameters are internally stored as :class:`Parameters` objects, but
       to evaluate the NLL it is more convenient to have simple name:value
       pairs. This also allows to add to the list the expressions, which
       are essentially functions of the parameters. These are added in order,
       under the assumption that expressions of other expressions are listed
       after their dependents.

      Args:
         pars: a :class:`Parameters` object
      Returns:
         a dictionary of name:value pairs
    """
    vals = pars.dict(nominal_nps=True)
    for real in self.expressions.values() : vals[real.name] = real.value(vals)
    return vals

  def n_exp(self, pars : Parameters) -> np.array :
    """Return the expected event yields for a given parameter value

    The expected yields correspond to the nominal yields for each sample,
    corrected for the overall normalization terms (function of the POIs)
    and the NP impacts (function of the NPs, see :meth:`Model.k_exp`)
    They are provided for each sample in each measurement bin, as a 2D
    np.array with dimensions`nbins` x `nsamples`.

      Args:
         pars: a set of parameter values
      Returns:
         expected event yields per sample per bin
    """
    real_vals = self.real_vals(pars)
    nnom = np.stack([ np.concatenate([ self.samples[(channel_name, s)].yields(real_vals) if s < len(channel.samples) else np.zeros(channel.nbins()) \
                      for channel_name, channel in self.channels.items()]) for s in range(0, self.max_nsamples) ])
    if self.verbosity > 3 : print(nnom)
    return nnom*self.cut_k_exp(pars)

  def channel_n_exp(self, pars : Parameters = None, nexp : np.array = None, channel : str = None, sample : str = None) -> np.array :
    """Return the expected event yields for a given channel

      
      If only `pars` is provided, will return the expected yields for these parameter
      values but only for the bins in the channel specified by `channel` (whereas
      meth:`Model.n_exp` returns the expected yields for all bins.
      
      If `nexp` is provided, ignores `pars` and instead truncates these expected yields
      to only the bins in `channel`.
      
      If `sample` is provided, additionally truncates the output to only the expected
      yield for this sample (`sample` should be the name of a sample in `channel`)
      
      Args:
         pars: a set of parameter values
         nexp: an array of event yields, with dimensions`nbins` x `nsamples`
         channel: the name of a model channel
         sample: the name of a sample in `channel`
      Returns:
         expected event yields per sample per bin
    """
    if nexp is None and pars is None : raise ValueError("ERROR: must specify either 'pars' or 'nexp' for expected yields.")
    nexpval = nexp if nexp is not None else self.n_exp(pars)
    if sample is not None :
      nexpval = nexpval[list(self.channels[channel].samples.keys()).index(sample)]
    if len(nexpval.shape) == 1 :
      return nexpval[self.channel_offsets[channel] : self.channel_offsets[channel] + self.channels[channel].nbins()]
    else :
      return nexpval[:, self.channel_offsets[channel] : self.channel_offsets[channel] + self.channels[channel].nbins()]
      

  def tot_bin_exp(self, pars : Parameters, floor : float = None) -> np.array :
    """Return the total expected event yields for given parameter values

      Same as :meth:`Model.n_exp`, except that the yields are summed over
      all samples. They are provided as a 1D np.array of size `nbins`.

      Args:
         pars: a set of parameter values
         floor: optional minimal value to use for each yield (default: None)
      Returns:
         expected event yields per bin
    """
    ntot = self.n_exp(pars).sum(axis=0)
    if floor is not None : ntot[:self.nbins_poisson] = np.maximum(ntot[:self.nbins_poisson], floor)
    return ntot

  def nll(self, pars : Parameters, data : 'Data', offset : bool = True, floor : bool = None, no_constraints : bool = False) -> float :
    """Return the negative log-likelihood value for a given parameter set and dataset

      If the `offset` argument is `True` (default), the nll is computed relatively
      to the case where all yields are nominal. This leads to smaller nll values,
      which reduces potential floating-point issues. When computing the difference
      of two nll values as in a profile-likelihood ratio computation, the offset
      cancels out in the difference.

      Args:
         pars   : a set of parameter values
         data   : an observed dataset
         offset : if True, use offsetting to reduce floating-point precision issues
         floor  : if a positive number is provided, will replace yields lower than the
                  floor by the floor itsel (to avoid e.g. negative yields in Poisson channels).
         no_constraints : if `True`, omit the penalty terms from the constraint in the computation.
      Returns:
         The negative log-likelihood value
    """
    delta_nps = data.aux_obs - pars.nps
    ntot = self.tot_bin_exp(pars, floor)
    try :
      result = 0
      if not offset :
        if self.nbins_poisson > 0 :
          result += np.sum(ntot[:self.nbins_poisson] - data.counts[:self.nbins_poisson]*np.log(ntot[:self.nbins_poisson]))
        if self.nbins_gaussian > 0 :
          delta_poi = ntot[self.nbins_poisson:] - data.counts[self.nbins_poisson:]
          result += 0.5*np.linalg.multi_dot((delta_poi, self.poi_hessian, delta_poi))
      else :
        if self.nbins_poisson > 0 :
          nexp0 = self.ref_yields.sum(axis=0)[:self.nbins_poisson]
          result += np.sum(ntot[:self.nbins_poisson] - nexp0 - data.counts[:self.nbins_poisson]*(np.log(ntot[:self.nbins_poisson]/nexp0)))
        if self.nbins_gaussian > 0 :
          nexp0 = self.ref_yields.sum(axis=0)[self.nbins_poisson:]
          delta_poi = ntot[self.nbins_poisson:] - data.counts[self.nbins_poisson:]
          result += 0.5*np.linalg.multi_dot((delta_poi + nexp0, self.poi_hessian, delta_poi - nexp0))
      if not no_constraints :
         result += 0.5*np.linalg.multi_dot((delta_nps, self.constraint_hessian, delta_nps))
      if math.isnan(result) : result = np.Infinity
      return result
    except Exception as inst:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      print('Fast NLL computation failed with the following exception in %s at line %g, returning +Inf' % (exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_lineno))
      print(inst)
      return np.Infinity

  def gradient(self, pars : Parameters, data : 'Data') -> np.ndarray :
    """Return the derivatives of the negative log-likelihood wrt the POIs

      Output format: 1D np.ndarray of size `npois`.

      Args:
         pars : parameter values at which to compute the likelihood
         data : observed dataset for which to compute the likelihood
      Returns:
         Values of the derivatives of the negative log-likelihood wrt the POIs.
    """
    try :
      # The NLL is a sum of Poisson terms, each one with an nexp given by a sum over samples
      # This means we can get the gradient using the formula
      # dNLL/dmu_a = sum_i (G^(a)_i - n_i G^(a)_i)
      # where i runs over bins, s over samples, G^(a)_i is the `tot_grads` object,
      # N_i the nexp and n_i the observed counts.
      real_vals = self.real_vals(pars)
      nexps = self.n_exp(pars) # shape = (n_samples, n_bins)
      # grads: shape = (n_samples, n_pois).
      # This is the gradient of the expected yields, modified with the NP effects
      grads = np.stack([ np.concatenate([ self.samples[(channel_name, s)].gradient(self.pois, self.reals(), real_vals)
                                          if s < len(channel.samples) else np.zeros((channel.nbins(), len(self.pois)))
                                          for channel_name, channel in self.channels.items()])
                             for s in range(0, self.max_nsamples) ])*self.cut_k_exp(pars)[:, :, None]
      ntots = np.sum(nexps, axis=0)  # shape = (n_bins)
      gtots = np.sum(grads, axis=0)  # shape = (n_bins, n_pois)
      gdivs = np.divide(gtots, ntots[:, None], out=np.zeros_like(gtots), where=ntots[:, None]!=0)
      return np.sum(gtots - gdivs*data.counts[:, None], axis=0)
    except Exception as inst:
      print('Exception in gradient computation: ', Exception, inst)
      import traceback
      print(traceback.format_exc())
      return None

  def hessian(self, pars : Parameters, data : 'Data') -> np.ndarray :
    """Return the Hessian matrix of the negative log-likelihood wrt the POIs

      Output format: 2D np.ndarray of size `npois` x `npois`.

      Args:
         pars : parameter values at which to compute the likelihood
         data : observed dataset for which to compute the likelihood
      Returns:
         Hessian matrix of the negative log-likelihood wrt the POIs.
    """
    try :
      # The NLL is a sum of Poisson terms, each one with an nexp given by a sum over samples
      # This means we can get the Hessian using the formula
      # dNLL/(dmu_a dmu_b) = sum_i ( H^(a,b)_i - n_i*( H^(a,b)_i - G^(a)_i*G^(a)_i ))
      # where i runs over bins, s over samples, G^(a)_i is the `tot_rel_grads` object,
      # H^(a,b)_i is the `tot_rel_hesse` object,  and n_i the observed counts.
      reals = self.reals()
      real_vals = self.real_vals(pars)
      nexps = self.n_exp(pars) # shape = (n_samples, n_bins)
      # grads: shape = (n_samples, n_pois).
      # This is the gradient of the expected yields, modified with the NP effects
      grads = np.stack([ np.concatenate([ self.samples[(channel_name, s)].gradient(self.pois, reals, real_vals)
                                          if s < len(channel.samples) else np.zeros((channel.nbins(), len(self.pois)))
                                          for channel_name, channel in self.channels.items()])
                             for s in range(0, self.max_nsamples) ])*self.cut_k_exp(pars)[:, :, None]
      # hesse: shape = (n_samples, n_pois, n_pois)
      # This is the Hessian of the expected yields, modified with the NP effects
      hesse = np.stack([ np.concatenate([ self.samples[(channel_name, s)].hessian(self.pois, reals, real_vals)
                                          if s < len(channel.samples) else np.zeros((channel.nbins(), len(self.pois), len(self.pois)))
                                          for channel_name, channel in self.channels.items()])
                         for s in range(0, self.max_nsamples) ])*self.cut_k_exp(pars)[:, :, None, None]
      ntots = np.sum(nexps, axis=0)  # shape = (n_bins)
      gtots = np.sum(grads, axis=0)  # shape = (n_bins, n_pois)
      htots = np.sum(hesse, axis=0)  # shape = (n_bins, n_pois, n_pois)
      gdivs = np.divide(gtots, ntots[:, None]      , out=np.zeros_like(gtots), where=ntots[:, None]!=0)
      hdivs = np.divide(htots, ntots[:, None, None], out=np.zeros_like(htots), where=ntots[:, None, None]!=0)
      return np.sum(htots - (hdivs - gdivs[:, :, None]*gdivs[:, None, :])*data.counts[:, None, None], axis=0)
    except Exception as inst:
      print('Exception in Hessian computation: ', Exception, inst)
      import traceback
      print(traceback.format_exc())
      return None

  def covariance_matrix(self, pars : Parameters, data : 'Data') -> np.ndarray :
    """Return the covariance matrix of the POIs

      Computes the covariance matrix of the POIs at the value specified
      by `pars`, in the fit of the model to `data`.

      Args:
         pars : parameter values at which to compute the covariance
         data : observed dataset for which to compute the likelihood
      Returns:
         the covariance matrix of the POIs
    """
    hess = self.hessian(pars, data)
    try:
      return np.linalg.inv(hess)
    except np.linalg.LinAlgError :
      return None

  def parabolic_errors(self, pars : Parameters = None, data : 'Data' = None, covmat : np.array = None) -> np.ndarray :
    """Return the parabolic uncertaintiees of the POIs

      Extract the parabolic uncertainties of the POIs
      from the diagonal of the covariance matrix. 

      Args:
         pars : parameter values at which to compute the covariance
         data : observed dataset for which to compute the likelihood
      Returns:
         the parabolic uncertaintiees of the POIs
    """
    cov = covmat if covmat is not None else self.covariance_matrix(pars, data)
    errors = np.sqrt(cov.diagonal())
    return { poi : errors[i] for i, poi in enumerate(self.pois) }

  def correlation_matrix(self, pars : Parameters = None, data : 'Data' = None, covmat : np.array = None) -> np.ndarray :
    """Return the correlation matrix of the POIs

      Compute the correlation matrix of the POIs from their
      covariance matrix.

      Args:
         pars : parameter values at which to compute the covariance
         data : observed dataset for which to compute the likelihood
      Returns:
         the correlation matrix of the POIs
    """
    cov = covmat if covmat is not None else self.covariance_matrix(pars, data)
    errors = np.sqrt(cov.diagonal())
    return (cov.T / errors).T / errors

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

  def plot(self, pars : Parameters, data : 'Data' = None, channel_names : str = None, only : list = None, exclude : list = None,
           variations : list = None, residuals : bool = False, canvas : tuple = (None, None), labels : bool = True, stack : bool = False, figsize=(8,6),
           bin_width : float = None, logy : bool = False, legend : bool = True) :
    """Plot the expected event yields, and optionally the data as well

      The plot is performed for the channel(s) specified by `channel_names`, or all channels
      if `None` is passed (default). Each channel is plotted on a separate subplot. If
      a figure is provided by `canvas`, this is used for plotting, otherwise a new figure is 
      created with dimention `figsize`.
      
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
         residuals  : if `True`, plot the data-model differences in a lower panel.
         stack      : if `True`, plot the samples stacked on top of each other.
         canvas     : a matplotlib (fig, axs) pair on which to plot (if None, a new figure is created)
         figsize    : the size of the figure to create, as a (size_x, size_y) tuple (default: (8,6))
         bin_width  : bin width to normalize the yields to, in case of channels with unequal bins
                      if not specified, plot unnormalized bin yields.
         logy       : if `True`, use a logarithmic scale on the Y axis
         legend     : if `True` (default), add a legend to the plot
         labels     : if `True` (default), add labels to the legend 
    """
    if not isinstance(only, list)    and only    is not None : only = [ only ]
    if not isinstance(exclude, list) and exclude is not None : exclude = [ exclude ]
    if channel_names is None : channel_names = list(self.channels.keys())
    if isinstance(channel_names, list) :
      nchan = len(channel_names)
      nrows = int(math.sqrt(nchan))
      ncols = int(nchan/nrows + 0.5)
      if canvas == (None, None) : 
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=100, constrained_layout=True)
      else :
        fig, axs = canvas
      if not isinstance(axs, list) : axs = np.array([ axs ])
      for ax in axs.flatten()[nchan:] : ax.set_visible(False)
      for channel, ax in zip(channel_names, axs.flatten()[:nchan]) :
        self.plot(pars=pars, data=data, channel_names=channel, only=only, exclude=exclude, variations=variations, residuals=residuals, canvas=(fig, ax), labels=labels, stack=stack, bin_width=bin_width, logy=logy, legend=legend)
      fig.tight_layout()
      return fig, axs
    if not channel_names in self.channels : raise KeyError('ERROR: Channel %s is not defined.' % channel_names)
    channel = self.channels[channel_names]
    if isinstance(channel, BinnedRangeChannel) :
      grid = [ b['lo_edge'] for b in channel.bins ]
      grid.append(channel.bins[-1]['hi_edge'])
    elif isinstance(channel, SingleBinChannel) :
      grid = [0,1]
    elif isinstance(channel, MultiBinChannel) :
      grid = np.linspace(0, channel.nbins(), channel.nbins() + 1)
    else :
      raise ValueError("Channel '%s' is of an unsupported type" % channel.name)
    if canvas == (None, None) : 
      fig, axs = plt.subplots(figsize=figsize, dpi=100, constrained_layout=True)
    else :
      fig, axs = canvas
    if logy: axs.set_yscale('log')
    if isinstance(pars, dict) : pars = Parameters(pars, model=self)
    xvals = [ (grid[i] + grid[i+1])/2 for i in range(0, len(grid) - 1) ]
    nexp = self.channel_n_exp(pars=pars, channel=channel.name)
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
    axs.hist(xvals, weights=yvals, bins=grid, histtype='step',color='b', linestyle=line_style, label=title if labels else None)
    if stack :
      for sample in samples :
        stack_exp = nexp[sample:,:].sum(axis=0)
        yvals = stack_exp - subtract if not residuals or data is None else stack_exp - subtract - counts
        if isinstance(channel, BinnedRangeChannel)  and bin_width is not None : yvals *= bin_corrs
        axs.hist(xvals, weights=yvals, bins=grid, histtype='step', linestyle=line_style, fill=True, label=list(channel.samples)[sample])
    if data is not None :
      counts = self.channel_n_exp(nexp=data.counts, channel=channel.name)
      yerrs = [ math.sqrt(n) if n > 0 else 0 for n in counts ]
      yvals = counts if not residuals else np.zeros(channel.nbins())
      if isinstance(channel, BinnedRangeChannel) and bin_width is not None : 
        yvals *= bin_corrs
        yerrs *= bin_corrs
      axs.errorbar(xvals, yvals, xerr=[0]*channel.nbins(), yerr=yerrs, fmt='ko', label='Data' if labels else None, zorder=99)
    axs.set_xlim(grid[0], grid[-1])
    if variations is not None :
      for v in variations :
        vpars = pars.clone()
        vpars.set(v[0], v[1])
        col = 'r' if len(v) < 3 else v[2]
        nexp = self.channel_n_exp(pars=vpars, channel=channel.name)
        if only is None and exclude is None :
          subtract = np.zeros(nexp.shape[1])
        else :
          subtract = nexp[samples,:].sum(axis=0)
          if only is not None : subtract = nexp.sum(axis=0) - subtract
        tot_exp = nexp.sum(axis=0) - subtract
        if isinstance(channel, BinnedRangeChannel) and bin_width is not None : tot_exp *= bin_corrs
        axs.hist(xvals, weights=tot_exp, bins=grid, histtype='step',color=col, linestyle=line_style, label='%s=%+g' %(v[0], v[1]) if labels else None)
    if legend and labels : axs.legend().set_zorder(100)
    axs.set_title(channel.name)
    if isinstance(channel, BinnedRangeChannel) :
      axs.set_xlabel(channel.obs_name + ((' ['  + channel.obs_unit + ']') if channel.obs_unit != '' else ''))
      axs.set_ylabel('Events / %g %s ' % (bin_width, channel.obs_unit) if bin_width is not None else 'Events')
    elif isinstance(channel, SingleBinChannel) :
      axs.tick_params(axis='x', which='both', bottom=False, labelbottom=False) # remove x ticks and labels 
      #axs.set_xlabel(channel.name)
      axs.set_ylabel('Events')
    return fig, axs
    #plt.bar(np.linspace(0,self.sig.size - 1,self.sig.size), self.n_exp(pars), width=1, edgecolor='b', color='', linestyle='dashed')

  def expected_pars(self, pois : dict, minimizer : 'NPMinimizer' = None) -> Parameters :
    """Return a :class:`Parameters` object for a set of POI values

      By default, returns a :class:`Parameters` object with the POI values
      defined by the `pois` arg, and the NPs set to 0. If a minimizer is 
      provided, set the NPs to their profiled values.
      The `pois` arg can also be a class:`Parameters` object, from which
      the POI values will be taken (and the NP values ignored).

      Args:
         pois : A dict of POI name : value pairs, or a class:`Parameters` object.
         minimizer : an optional minimizer object
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
         a randomly-generated dataset
    """
    if not isinstance(pars, Parameters) : pars = Parameters(pars, model=self)
    ntot = self.tot_bin_exp(pars)
    yields = np.zeros(len(ntot))
    if self.nbins_poisson  > 0 : yields[:self.nbins_poisson] = np.random.poisson(ntot[:self.nbins_poisson])
    if self.nbins_gaussian > 0 : yields[self.nbins_poisson:] = np.random.multivariate_normal(ntot[self.nbins_poisson:], self.poi_hessian)
    return Data(self, yields, [ par.generate_aux(pars[par.name]) for par in self.nps.values() ])

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
         an Asimov dataset
    """
    if not isinstance(pars, Parameters) : pars = Parameters(pars, model=self)
    return Data(self).set_data(self.tot_bin_exp(pars), pars.nps)

  def generate_expected(self, pois : dict, minimizer : 'NPMinimizer' = None) :
    """Generate an Asimov dataset for expected parameter values

      Same functionality as :meth:`Model.generate_asimov`, but with
      NP values that are obtained from the provided POI values in the
      same way as described for :meth:`Model.expected_pars`.

      Args:
         pois : a dict of POIs { name : value } pairs, or a class:`Parameters` object.
         minimizer : an optional minimizer used to compute NP profile values
      Returns:
         an Asimov dataset
    """
    return self.generate_asimov(self.expected_pars(pois, minimizer))

  @staticmethod
  def create(filename : str, verbosity : int = 0, flavor : str = None, use_linear_nps : bool = False) -> 'Model' :
    """Shortcut method to instantiate a model from a markup file

      Same behavior as creating a default model and loading from the file,
      rolled into a single command

      Args:
         filename : name of a markup file containing the model definition
         verbosity: level of verbosity (0=minimal)
         flavor   : input markup flavor (currently supported: 'json' [default], 'yaml')
         use_linear_nps: if `True`, use linear NP impacts (see :meth:`Model.__init__`)
      Returns:
         the created model
    """
    return Model(use_linear_nps=use_linear_nps, verbosity=verbosity).load(filename, flavor=flavor)

  @staticmethod
  def create_from_dict(sdict : dict, verbosity : int = 0) -> 'Model' :
    """Shortcut method to instantiate a model from a markup file

      Same behavior as creating a default model and loading from the file,
      rolled into a single command

      Args:
         sdict : a dictionary containing the model definition
         verbosity: level of verbosity (0=minimal)
      Returns:
         the created model
    """
    return Model(verbosity=verbosity).load_dict(sdict)

  def load_dict(self, sdict : dict) -> 'Model' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    if not 'model'    in sdict : raise KeyError("No 'model' section in specified markup file")
    if not 'POIs'     in sdict['model'] : raise KeyError("No 'POIs' section in specified markup file")
    if not 'channels' in sdict['model'] : raise KeyError("No 'channels' section in specified markup file")
    if self.verbosity > 1 : print('Loading parameters')
    self.name = self.load_field('name', sdict['model'], '', str)
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
    if 'expressions' in sdict['model'] :
      for dict_expr in sdict['model']['expressions'] :
        expression = Expression.instantiate(dict_expr)
        if expression.name in self.expressions :
          raise ValueError('ERROR: multiple expressions defined with the same name (%s)' % expression.name)
        self.expressions[expression.name] = expression
    if self.verbosity > 1 : print('Loading channels')
    for dict_channel in sdict['model']['channels'] :
      if not 'type' in dict_channel or dict_channel['type'] == SingleBinChannel.type_str :
        channel = SingleBinChannel()
      elif dict_channel['type'] == BinnedRangeChannel.type_str :
        channel = BinnedRangeChannel()
      elif dict_channel['type'] == MultiBinChannel.type_str :
        channel = MultiBinChannel()
      elif dict_channel['type'] == GaussianChannel.type_str :
        channel = GaussianChannel()
      else :
        raise ValueError("ERROR: unsupported channel type '%s'" % dict_channel['type'])
      channel.load_dict(dict_channel)
      if channel.name in self.channels :
        raise ValueError('ERROR: multiple channels defined with the same name (%s)' % channel.name)
      for sample in channel.samples.values() :
        # nominal yields can be None, which corresponds to [nominal_norm]*nbins which automatically works
        if sample.nominal_yields is not None and len(sample.nominal_yields) != channel.nbins() :
          raise ValueError("ERROR: sample '%s' of channel '%s' has nominal_yields of the wrong size (%d, expected %d)."
                           % (sample.name, channel.name, len(sample.nominal_yields), channel.nbins()))
      self.channels[channel.name] = channel
    self.set_internal_vars()
    return self

  def fill_dict(self, sdict : dict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: a dictionary containing markup data
    """
    sdict['model'] = {}
    sdict['model']['name'] = self.name
    sdict['model']['POIs'] = []
    sdict['model']['NPs'] = []
    sdict['model']['expressions'] = []
    sdict['model']['aux_obs'] = []
    sdict['model']['channels'] = []
    for poi  in self.pois.values()        : sdict['model']['POIs']        .append(poi.dump_dict())
    for par  in self.nps.values()         : sdict['model']['NPs']         .append(par.dump_dict())
    for expr in self.expressions.values() : sdict['model']['expressions'] .append(expr.dump_dict())
    for aux  in self.aux_obs.values()     : sdict['model']['aux_obs']     .append(aux.dump_dict())
    for channel in self.channels.values() : sdict['model']['channels'].append(channel.dump_dict())

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        the object description
    """
    return 'Model ' + self.string_repr(verbosity = 1)

  def string_repr(self, verbosity : int = 1, pre_indent : str = '', indent : str = '   ') -> str :
    """Return a string representation of the object

      Same as __str__ but with options

      Args:
        verbosity : verbosity of the output
        pre_indent: number of indentation spaces to add to all lines
        indent    : number of indentation spaces to add to fields of this object

      Returns:
        the object description
    """
    rep = '%s%s' % (pre_indent, self.name)
    if verbosity > 0 : rep += '\n'
    rep += '\n%sParameters of interest' % pre_indent
    if verbosity == 0 :
      rep += ' : ' + ', '.join([ par.string_repr(verbosity) for par in self.pois.values() ])
    else :
      for par in self.pois.values() :
        rep += '\n - ' + par.string_repr(verbosity, indent='  ')
    if verbosity > 0 : rep += '\n'
    rep += '\n%sNuisance parameters' % pre_indent
    if verbosity == 0 :
      rep += '    : ' + ', '.join([ par.string_repr(verbosity) for par in self.nps.values() ])
    else :
      for par in self.nps.values() :
        rep += '\n - ' + par.string_repr(verbosity, indent='   ')
    if verbosity > 0 : rep += '\n'
    rep += '\n%sAuxiliary observables' % pre_indent
    if verbosity == 0 :
      rep += '  : ' + ', '.join([ par.string_repr(verbosity) for par in self.aux_obs.values() ])
    else :
      for par in self.aux_obs.values() :
        rep += '\n - ' + par.string_repr(verbosity, indent='   ')
    if verbosity > 0 : rep += '\n'
    rep += '\n%sChannels' % pre_indent
    if verbosity == 0 :
      rep += ' : ' + ', '.join([ channel.string_repr(verbosity) for channel in self.channels.values() ])
    else :
      for channel in self.channels.values() :
        rep += '\n - ' + channel.string_repr(verbosity, indent='   ')
    return rep

# -------------------------------------------------------------------------
class Data (Serializable) :
  """Class representing a dataset

  Stores the complete dataset information:

  * A list of observed bin yields (also including observable values for Gaussian channels)

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

  def clone(self, model : Model = None, counts : np.array = None, aux_obs : np.array = None) :
    """Return a clone of the object

      Performs a deep-copy of the `counts` and `aux_obs` attributes,
      or replaces them with the ones provided as arguments (in this
      case the arguments are not copied).

      Args:
         model  : new reference model, if provided
         counts : new observed bin counts, if provided
         aux_obs: new aux. obs. values, if provided.
      Returns:
        the new clone
    """
    return Data(model=model if model is not None else self.model,
                counts=counts if counts is not None else np.array(self.counts),
                aux_obs=aux_obs if aux_obs is not None else np.array(self.aux_obs))

  def set_counts(self, counts : np.ndarray) -> 'Data' :
    """Set the observed bin counts

      Args:
         counts : new observed bin counts
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

  def set_aux_obs(self, aux_obs : np.ndarray = []) -> 'Data' :
    """Set the aux. obs.

      Args:
         aux_obs : new aux. obs. values
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

  def set_data(self, counts : np.ndarray, aux_obs : np.ndarray) -> 'Data' :
    """Set both the observed bin counts and aux. obs.

      Args:
         counts  : new observed bin counts
         aux_obs : new aux. obs. values
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
      model_channel.load_dataset_dict(channel, self.counts[offset:offset + model_channel.nbins()])
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
      if not par.aux_obs in data_aux_obs : raise ValueError('Auxiliary observable %s defined in model, but not provided in the data' % par.aux_obs)
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
      channel.save_dataset_dict(channel_data, self.model.channel_n_exp(nexp=self.counts, channel=channel_name))
      sdict['data']['channels'].append(channel_data)
    sdict['data']['aux_obs'] = []
    for p, par in enumerate(self.model.nps.values()) :
      if par.aux_obs is None : continue
      aux_data = { 'name' : par.aux_obs, 'value' : par.unscaled_value(self.aux_obs[p]) }
      sdict['data']['aux_obs'].append(aux_data)

  def save_with_model(self, filename : str, flavor : str = None) :
    """Utility function to save model+data information together
    
      Saves the information of the dataset together with that
      of the associated model, in a single markup file

      Args
        filename: file name to save to
        flavor: markup flavor (currently either 'json' or 'yaml')
    """
    sdict = self.model.dump_dict()
    sdict.update(self.dump_dict())
    self.save(filename, flavor, payload=sdict)

  def __str__(self) -> str :
    """Provide a description string

      Returns:
        the object description
    """
    return 'Data ' + self.string_repr(verbosity = 1)

  def string_repr(self, verbosity : int = 1, pre_indent : str = '', indent : str = '   ') -> str :
    """Return a string representation of the object

      Same as __str__ but with options

      Args:
        verbosity : verbosity of the output
        pre_indent: number of indentation spaces to add to all lines
        indent    : number of indentation spaces to add to fields of this object

      Returns:
        the description string
    """
    rep = '%s (model : %s), counts = %g, aux_obs = %s' % (pre_indent,  self.model.name, np.sum(self.counts), str(self.aux_obs))
    if verbosity >= 1 :
      rep = '%s (model : %s)' % (pre_indent,  self.model.name)
      rep += '\n'
      rep += '\n%sCounts' % pre_indent
      for channel_name, channel in self.model.channels.items() :
        counts = self.model.channel_n_exp(nexp=self.counts, channel=channel_name)
        rep += '\n%s - channel %s : %s = %s' % (pre_indent, channel_name, 'total counts' if verbosity == 1 else 'counts', str(np.sum(counts)) if verbosity == 1 else str(counts))
      rep += '\n'
      rep += '\n%sAuxiliary observables' % pre_indent
      for aux_name, aux_val in zip(self.model.aux_obs, self.aux_obs) :
        rep += '\n%s - %s = %g' % (pre_indent, aux_name, aux_val)
    return rep
  
