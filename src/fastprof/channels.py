"""Module containing the `Channel` class, defining a *channel*
(measurement region) of fastprof models, and its derived classes:

  * :class:`Channel` : the base class for channels objects.
  * :class:`SingleBinChannel` : a class representing a channel 
    with a single bin.
  * :class:`BinnedRangeChannel` : a class representing a channel
    with bins spanning a range of a continuous observable.
  * :class:`MultiBinChannel` : a class representing a channel
    with multiple non-adjacent bins.
  * :class:`GaussianChannel` : a class representing a Gaussian
    measurement channel.


  All classes support loading from / saving to markup files. The basic mechanism for this I/O is implemented in the :class:`Serializable` base class from which they all derive. The JSON and YAML markup formats are supported.
"""

from .base import Serializable
from .sample import Sample
import numpy as np
from abc import abstractmethod
import copy


# -------------------------------------------------------------------------
class Channel(Serializable) :
  """Class representing a model channel

  Provides the functionality for HistFactory channel structures,
  representing one or more of measurements.

  Derived class implement different cases, i.e.
  this one: :class:`SingleBinChannel`, :class:`BinnedRangeChannel`,
  :class:`MultiBinChannel` and :class:`GaussianChannel`. 

  This class is the common base, defining :
  * The channel name

  * A list of :class:`Sample` objects representing the processes
    (signal and background) contributing to the event yield in each bin.

  Attributes:
     name (str) : the channel name
     samples (dict) : the channel samples, as a dict mapping the sample names
                      to the sample objects (see :class:`Sample`).
  """

  def __init__(self, name : str = '', samples : dict = None) :
    """Initializes the Channel class

      Args:
         name : the channel name
         samples: the samples, as a { name : object } dict.
    """
    self.name = name
    self.samples = samples if samples is not None else {}


  @abstractmethod
  def nbins(self) -> int :
    """Returns the number of bins in the channel

      Returns:
        the number of bins
    """
    pass

  def sample(self, name : str) :
    """Access a sample by name

    Args:
      name : the sample name

    Returns:
      the named sample, or `None` if not defined
    """
    return self.samples[name] if name in self.samples else None

  def pdf_type(self) -> str :
    """Return the type of PDF implemented by the channel.
       
       Supported types are currently 'poisson' for
       binned measurements (default), and 'gaussian'
       for Gaussian measurements.

    Returns:
      the PDF type
    """
    return 'poisson'

  def __str__(self) -> str :
    """Returns a description string

      Returns:
        the object description
    """
    return 'Channel ' + self.string_repr(verbosity = 1)

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
    rep = '%s%s' % (pre_indent,  self.name)
    if verbosity >= 1 :
      rep += ' : %g bins' % self.nbins()
    if verbosity > 0 :
      for sample in self.samples.values() : rep += '\n%s  o Sample ' % pre_indent + sample.string_repr(verbosity, pre_indent=pre_indent+'    ', indent=indent)
    return rep

  def load_dict(self, sdict : dict) -> 'Channel' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: a dictionary containing markup data

      Returns:
        self
    """
    self.name = sdict['name']
    for dict_sample in sdict['samples'] :
      sample = Sample()
      sample.load_dict(dict_sample)
      self.samples[sample.name] = sample
    return self

  def fill_dict(self, sdict : dict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: a dictionary containing markup data
    """
    sdict['name'] = self.name
    sdict['samples'] = []
    for sample in self.samples.values() : sdict['samples'].append(sample.dump_dict())

  def load_dataset_dict(self, sdict : dict, counts : np.array) :
    """Load observed dataset information from markup

    Utility function called from :class:`fastprof.Data` to parse
    an observed dataset specification for this channel and fill an
    array of event counts.

      Args:
        sdict  : a dictionary containing markup data
        counts : the array of data counts to fill
    """
    if not 'name' in sdict :
      raise KeyError("Data channel definition must contain a 'name' field")
    if sdict['name'] != self.name :
      raise ValueError("Cannot load channel data defined for channel '%s' into channel '%s'" % (sdict['name'], self.name))
    
  def save_dataset_dict(self, sdict : dict, counts : np.array) :
    """Save observed dataset information from markup
    
    Utility function called from :class:`fastprof.Data` to write
    out an observed dataset specification for this channel into the
    appropriate format.

    Args:
      sdict  : A dictionary to fill with markup data
      counts : an array of data counts to read from
    """
    sdict['name'] = self.name
    if len(counts) != self.nbins() :
      raise ValueError("Cannot save channel data counts of length %d for channel '%s' with %d bins" % (len(counts), self.name, self.nbins()))


# -------------------------------------------------------------------------
class SingleBinChannel(Channel) :
  """Class representing a model channel consisting of a single bin

  Class derived from :class:`Channel` to handle the case of a single 
  named counting bin. The channel name also doubles as the bin name.

  Attributes :
     name (str) : the channel name
     samples (dict) : the channel samples as a { name: object } dict (see :class:`Sample`).
  """

  type_str = 'bin'

  def __init__(self, name : str = '', samples : dict = None) :
    """Initialize a SingleBinChannel instance

      Args:
        name : channel name (and bin name)
        samples  : channel samples, as a { name : object } dict
   """
    super().__init__(name, samples)

  def clone(self) -> 'SingleBinChannel' :
    """Clone the channel object
    
    Returns a new object of the same type, recursively cloning
    the channel samples.

      Returns:
        the cloned object
    """
    return SingleBinChannel(self.name, { sample.name : sample.clone() for sample in self.samples.values() })

  def nbins(self) -> int :
    """Return the number of bins in the channel

      Returns:
        the number of bins
    """
    return 1

  def load_dict(self, sdict : dict) :
    """Load object information from a dictionary of markup data

      Args:
        sdict: a dictionary containing markup data

      Returns:
        self
    """
    super().load_dict(sdict)
    if self.load_field('type', sdict, SingleBinChannel.type_str, str) != SingleBinChannel.type_str :
      raise ValueError("Trying to load a BinnedRangeChannel from channel data of type '%s'" % sdict['type'])
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: a dictionary containing markup data
    """
    super().fill_dict(sdict)
    sdict['type'] = SingleBinChannel.type_str
    sdict['samples'] = sdict.pop('samples') # reorder to have samples at end


  def load_dataset_dict(self, sdict : dict, counts : np.array) :
    """Load observed dataset information from markup
    
    Utility function called from :class:`fastprof.Data` to parse
    an observed data specification for this channel and fill an
    array of event counts.

      Args:
        sdict: A dictionary containing markup data
        counts : the array of data counts to fill
    """
    super().load_dataset_dict(sdict, counts)
    if 'type' in sdict and sdict['type'] != SingleBinChannel.type_str :
      raise ValueError("Cannot load channel data defined for type '%s' into channel of type '%s'" % (sdict['type'], SingleBinChannel.type_str))
    if not 'counts' in sdict :
      raise KeyError("No 'counts' section defined for data channel '%s' in specified markup file." % self.name)
    counts[0] = sdict['counts']

  def save_dataset_dict(self, sdict : dict, counts : np.array) :
    """Save observed dataset information from markup
    
    Utility function called from :class:`fastprof.Data` to write
    out an observed data specification for this channel into the
    appropriate format.

    Args:
      sdict  : A dictionary to fill with markup data
      counts : an array of data counts to read from
    """
    super().save_dataset_dict(sdict, counts)
    sdict['type'] = SingleBinChannel.type_str
    sdict['counts'] = int(counts[0])


# -------------------------------------------------------------------------
class BinnedRangeChannel(Channel) :
  """Class representing a model channel consisting of a binned observable range

  Class derived from :class:`Channel` to describe a 
  channel consisting of bins spanning a range of a
  continuous observable.
 
  Each bin is stored as a dict with the format
  { 'lo_edge' : <float value>, 'hi_edge': <float value> }
  providing its lower and upper bound.

  Attributes:
     name (str) : the channel name
     samples (dict) : the channel samples, as a dict mapping the sample names
                      to the sample objects (see :class:`Sample`).
     bin (list): list of bins in the format described above
     obs_name (str): name of the continuous observable
     obs_unit (str): unit of the continuous observable
  """

  type_str = 'binned_range'

  def __init__(self, name : str = '', bins : list = None, obs_name : str = '', obs_unit : str = '', samples : dict = None) :
    """Initialize a BinnedRangeChannel instance

      Args:
        name : the channel name
        bins : list of bin definitions, each in the form of a dict
               with format { 'lo_edge' : <float value>, 'hi_edge': <float value> }
        obs_name : name of the continuous observable
        obs_unit : unit of the continuous observable
        samples  : channel samples, as a { name : object } dict
    """
    super().__init__(name, samples)
    self.bins = bins if bins is not None else []
    self.obs_name = obs_name
    self.obs_unit = obs_unit

  def clone(self) -> 'BinnedRangeChannel' :
    """Clone the channel object
    
    Returns a new object of the same type, recursively cloning
    the channel samples and bin specifications.

      Returns:
        the cloned object
    """
    return BinnedRangeChannel(self.name, copy.deepcopy(self.bins), self.obs_name, self.obs_unit,
                              { sample.name : sample.clone() for sample in self.samples.values() })


  def nbins(self) -> int :
    """Returns the number of bins in the channel

      Returns:
        the number of bins
    """
    return len(self.bins)

  def load_dict(self, sdict : dict) :
    """Load object information from a dictionary of markup data

      Args:
        sdict: a dictionary containing markup data

      Returns:
        self
    """
    super().load_dict(sdict)
    if sdict['type'] != BinnedRangeChannel.type_str :
      raise ValueError("Trying to load a BinnedRangeChannel from channel data of type '%s'" % sdict['type'])
    self.bins = sdict['bins']
    self.obs_name = self.load_field('obs_name', sdict, '', str)
    self.obs_unit = self.load_field('obs_unit', sdict, '', str)
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: a dictionary containing markup data
    """
    super().fill_dict(sdict)
    sdict['type'] = BinnedRangeChannel.type_str
    sdict['obs_name'] = self.obs_name
    sdict['obs_unit'] = self.obs_unit
    sdict['bins'] = self.bins
    sdict['samples'] = sdict.pop('samples') # reorder to have samples at end

  def load_dataset_dict(self, sdict : dict, counts : np.array) :
    """Load observed data information from markup
    
    Utility function called from :class:`fastprof.Data` to parse
    an observed data specification for this channel and fill an
    array of event counts.

      Args:
        sdict: A dictionary containing markup data
        counts : the array of data counts to fill
    """
    super().load_dataset_dict(sdict, counts)
    if not 'name' in sdict :
      raise KeyError("Data channel definition must contain a 'name' field")
    if not 'bins' in sdict :
      raise KeyError("No 'bins' section defined for data channel '%s' in specified markup file." % self.name)
    if len(sdict['bins']) != self.nbins() :
      raise ValueError("Binned range channel '%s' in specified markup file has %d bins, but the model channel has %d." % (sdict['name'], len(sdict['bins']), self.nbins()))
    for b, bin_data in enumerate(sdict['bins']) :
      if bin_data['lo_edge'] != self.bins[b]['lo_edge'] or bin_data['hi_edge'] != self.bins[b]['hi_edge'] :
        raise ValueError("Bin %d in data channel '%s' spans [%g,%g], but the model bin spans [%g,%g]." %
                         (b, self.name, bin_data['lo_edge'], bin_data['hi_edge'], self.bins[b]['lo_edge'], self.bins[b]['hi_edge']))
      counts[b] = bin_data['counts']

  def save_dataset_dict(self, sdict : dict, counts : np.array) :
    """Save observed data information from markup
    
    Utility function called from :class:`fastprof.Data` to write
    out an observed data specification for this channel into the
    appropriate format.

    Args:
      sdict  : A dictionary to fill with markup data
      counts : an array of data counts to read from
    """
    super().save_dataset_dict(sdict, counts)
    sdict['type'] = BinnedRangeChannel.type_str
    sdict['obs_name'] = self.obs_name
    sdict['obs_unit'] = self.obs_unit
    sdict['bins'] = []
    for b, bin_spec in enumerate(self.bins) :
      bin_data = {}
      bin_data['lo_edge'] = bin_spec['lo_edge']
      bin_data['hi_edge'] = bin_spec['hi_edge']
      bin_data['counts'] = int(counts[b])
      sdict['bins'].append(bin_data)

  def string_repr(self, verbosity = 1, pre_indent = '', indent = '   ') :
    """Return a string representation of the object

      Same as __str__ but with options.

      Args:
        verbosity : verbosity of the output
        pre_indent: number of indentation spaces to add to all lines
        indent    : number of indentation spaces to add to fields of this object

      Returns:
        the description string
    """
    rep = '%s%s' % (pre_indent,  self.name)
    unit = ' %s' % self.obs_unit if self.obs_unit is not None and self.obs_unit != '' else ''
    if verbosity == 1 :
      rep += ' : %g bins from %s = %g%s to %g%s' % (self.nbins(), self.obs_name, self.bins[0]['lo_edge'], unit, self.bins[-1]['hi_edge'], unit)
    elif verbosity >= 2 :
        rep += ' : %g bins along observable %s' % (self.nbins(), self.obs_name)
        for i, b in enumerate(self.bins) :
          rep += '\n%sbin %2d : [ %g%s, %g%s ]' % (pre_indent + indent, i, b['lo_edge'], unit, b['hi_edge'], unit)
    if verbosity > 0 :
      for sample in self.samples.values() : rep += '\n%s  o Sample ' % pre_indent + sample.string_repr(verbosity, pre_indent=pre_indent+'    ', indent=indent)
    return rep



# -------------------------------------------------------------------------
class MultiBinChannel(Channel) :
  """Class representing a model channel consisting of multiple discrete bins

  Class derived from :class:`Channel` to describe a channel consisting
  of multiple discrete, named bins. Each bin is specified by a bin name.
  
  Attributes:
     name (str) : the channel name
     samples (dict) : the channel samples, as a dict mapping the sample names
                      to the sample objects (see :class:`Sample`).
     bins (list) : list of channel bin names 
  """
 
  type_str = 'multi_bin'

  def __init__(self, name : str = '', bins : list = None, samples : dict = None) :
    """Initialize the MultiBinChannel class

      Args:
         name : channel name
         bins : list of bin names
         samples  : channel samples, as a { name : object } dict
    """
    super().__init__(name, samples)
    self.bins = bins if bins is not None else {}

  def clone(self) -> 'MultiBinChannel' :
    """Clone the channel object
    
    Returns a new object of the same type, recursively cloning
    the channel samples and bin list.

      Returns:
        the cloned object
    """
    return MultiBinChannel(self.name, copy.deepcopy(self.bins), { sample.name : sample.clone() for sample in self.samples.values() })

  def nbins(self) -> int :
    """Returns the number of bins in the channel

      Returns:
        the number of bins
    """
    return len(self.bins)

  def load_dict(self, sdict : dict) :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    super().load_dict(sdict)
    if sdict['type'] != MultiBinChannel.type_str :
      raise ValueError("Trying to load a MultiBinChannel from channel data of type '%s'" % sdict['type'])
    self.bins = sdict['bins']
    return self

  def fill_dict(self, sdict : dict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    super().fill_dict(sdict)
    sdict['type'] = MultiBinChannel.type_str
    sdict['bins'] = self.bins
    sdict['samples'] = sdict.pop('samples') # reorder to have samples at end

  def load_dataset_dict(self, sdict : dict, counts : np.array) :
    """Load observed dataset information from markup
    
    Utility function called from :class:`fastprof.Data` to parse
    an observed dataset specification for this channel and fill an
    array of event counts.

      Args:
        sdict: a dictionary containing markup data
        counts : the array of data counts to fill
    """
    super().load_dataset_dict(sdict, counts)
    if not 'name' in sdict :
      raise KeyError("Data channel definition must contain a 'name' field")
    if not 'bins' in sdict :
      raise KeyError("No 'bins' section defined for data channel '%s' in specified markup file." % self.name)
    if len(sdict['bins']) != self.nbins() :
      raise ValueError("Binned range channel '%s' in specified markup file has %d bins, but the model channel has %d." % (channel['name'], len(channel['bins']), self.nbins()))
    for b, bin_data in enumerate(sdict['bins']) :
      if bin_data['name'] != self.bins[b] :
        raise ValueError("Bin %d in data channel '%s' has name '%s', but the model specifies name '%s'." % (b, self.name, bin_data['name'], self.bins[b]))
      counts[b] = bin_data['counts']

  def save_dataset_dict(self, sdict : dict, counts : np.array) :
    """Save observed dataset information from markup
    
    Utility function called from :class:`fastprof.Data` to write
    out an observed dataset specification for this channel into the
    appropriate format.

    Args:
      sdict  : a dictionary to fill with markup data
      counts : an array of data counts to read from
    """
    super().save_dataset_dict(sdict, counts)
    sdict['type'] = MultiBinChannel.type_str
    sdict['bins'] = []
    for b, bin_name in enumerate(self.bins) :
      bin_data = {}
      bin_data['name'] = bin_name
      bin_data['counts'] = int(counts[b])
      sdict['bins'].append(bin_data)

  def string_repr(self, verbosity = 1, pre_indent = '', indent = '   ') :
    """Return a string representation of the object

      Same as __str__ but with options.

      Args:
        verbosity : verbosity of the output
        pre_indent: number of indentation spaces to add to all lines
        indent    : number of indentation spaces to add to fields of this object

      Returns:
        the description string
    """
    rep = '%s%s' % (pre_indent,  self.name)
    if verbosity == 1 :
      rep += ' : %g bins (%s)' % (self.nbins(), ','.join(self.bins))
    elif verbosity >= 2 :
      rep += ' : %g bins : '
      for i, b in enumerate(self.bins) :
        rep += '\n%sbin%d : %s' % (pre_indent + indent, i, b)
    if verbosity > 0 :
      for sample in self.samples.values() : rep += '\n%s  o Sample ' % pre_indent + sample.string_repr(verbosity, pre_indent=pre_indent+'    ', indent=indent)
    return rep

# -------------------------------------------------------------------------
class GaussianChannel(Channel) :
  """Class representing a model channel with Gaussian bins

  Class derived from :class:`Channel` to handle a channel consisting
  of a set of named bin with observables following Gaussian distributions
 
  Each bin is specified by a bin name, and the Hessian matrix of the 
  measurement (the inverse of its covariance) must be provided.
  
  Attributes:
     name (str) : the channel name
     samples (dict) : the channel samples, as a dict mapping the sample names
                      to the sample objects (see :class:`Sample`).
     bins (list) : list of channel bin names 
     hessian (np.ndarray) : Hessian matrix (inverse of the covariance matrix)
  """
  type_str = 'gaussian'

  def __init__(self, name : str = '', bins : list = None, samples : dict = None,
               hessian : np.ndarray = None, covariance : np.ndarray = None) :
    """Initializes the GaussianChannel class

      Either the Hessian (inverse of the covariance matrix) or 
      the covariance matrix itself can be provided as input.

      Args:
         name      : channel name
         bins      : list of bin names
         samples   : channel samples, as a { name : object } dict
         hessian   : Hessian matrix of the Gaussian (inverse of the covariance)
         covariance: Covariance matrix of the Gaussian
    """
    super().__init__(name, samples)
    self.bins = bins if bins is not None else []
    if hessian is not None :
      self.hessian = hessian
    elif covariance is not None :
      try :
        self.hessian = np.linalg.inv(covariance)
      except Exception as inst :
        print("ERROR: could not invert the covariance matrix of Gaussian channel '%s', exiting." % name)
        raise inst
    else :
      self.hessian = None

  def clone(self) -> 'GaussianChannel' :
    """Clone the channel object
    
    Returns a new object of the same type, recursively cloning
    the channel samples, bins and Hessian.

      Returns:
        the cloned object
    """
    return GaussianChannel(self.name, copy.deepcopy(self.bins), { sample.name : sample.clone() for sample in self.samples },
                           np.array(self.hessian) if self.hessian is not None else None)

  def nbins(self) -> int :
    """Returns the number of bins in the channel

      Returns:
        the number of bins
    """
    return len(self.bins)

  def pdf_type(self) -> str :
    """Return the type of PDF implemented by the channel.
       
       Supported types are currently 'poisson' for
       binned measurements (default), and 'gaussian'
       for Gaussian measurements. This function
       overrides the default 'poisson' value in the
       :class:`Channel` base class.

    Returns:
      the PDF type
    """
    return 'gaussian'

  def load_dict(self, sdict : dict) :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        self
    """
    super().load_dict(sdict)
    if sdict['type'] != GaussianChannel.type_str :
      raise ValueError("Trying to load a GaussianChannel from channel data of type '%s'" % sdict['type'])
    self.bins = sdict['bins']
    self.hessian = np.linalg.inv(sdict['covariance'])
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: a dictionary containing markup data
    """
    super().fill_dict(sdict)
    sdict['type'] = MultiBinChannel.type_str
    sdict['bins'] = self.bins
    sdict['covariance'] = np.linalg.inv(self.hessian)
    sdict['samples'] = sdict.pop('samples') # reorder to have samples at end

  def load_dataset_dict(self, sdict : dict, obs : np.array) :
    """Load observed data information from markup
    
    Utility function called from :class:`fastprof.Data` to parse
    an observed data specification for this channel and fill an
    array of observables

      Args:
        sdict: a dictionary containing markup data
        obs  : the array of observables to fill
    """
    super().load_dataset_dict(sdict, obs)
    if not 'name' in sdict :
      raise KeyError("Data channel definition must contain a 'name' field")
    if not 'bins' in sdict :
      raise KeyError("No 'bins' section defined for data channel '%s' in the dataset description." % self.name)
    if len(sdict['bins']) != self.nbins() :
      raise ValueError("Binned range channel '%s' in the dataset description has %d bins, but the model channel has %d." % (sdict['name'], len(sdict['bins']), self.nbins()))
    for b, bin_data in enumerate(sdict['bins']) :
      if bin_data['name'] != self.bins[b] :
        raise ValueError("Bin %d in data channel '%s' has name '%s', but the model specifies name '%s'." % (b, self.name, bin_data['name'], self.bins[b]))
      obs[b] = bin_data['obs']

  def save_dataset_dict(self, sdict : dict, obs : np.array) :
    """Save observed data information from markup
    
    Utility function called from :class:`fastprof.Data` to write
    out an observed data specification for this channel into the
    appropriate format

    Args:
      sdict : a dictionary to fill with markup data
      obs   : an array of observables to read from
    """
    super().save_dataset_dict(sdict, obs)
    sdict['type'] = MultiBinChannel.type_str
    sdict['bins'] = []
    for b, bin_name in enumerate(self.bins) :
      bin_data = {}
      bin_data['name'] = bin_name
      bin_data['counts'] = int(obs[b])
      sdict['bins'].append(bin_data)

