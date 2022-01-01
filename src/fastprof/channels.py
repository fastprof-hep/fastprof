"""Module containing the building blocks for fastprof models:

  * :class:`Channel`, defining a measurement region*

"""

from .base import Serializable
from .sample import Sample
import numpy as np
from abc import abstractmethod



# -------------------------------------------------------------------------
class Channel(Serializable) :
  """Class representing a model channel

  Provides the functionality for HistFactory channel structures,
  representing a set of measurement bins. Two types of channels
  are currently implemented, differing in how the
  bin list is handled:

  * `bin` : a channel with a single measurement bin

  * `binned_range` : a channel with multiple bins spanning
    a range of a continuous observable.

  Each type corresponds to a different class derived from
  this one: :class:`SingleBinChannel` and :class:`BinnedRangeChannel`
  respectively.

  This class is the common base, defining :
  * The channel name

  * A list of :class:`Sample` objects representing the processes
    contributing to the event yield in each bin.

  Attributes:
     name (str) : the name of the channel
     samples (dict) : the channel samples, as a dict mapping the sample names
        to the sample objects (see :class:`Sample`).
  """

  def __init__(self, name : str = '', samples : dict = None) :
    """Initializes the Channel class

      Args:
         name : channel name
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

  def __str__(self) -> str :
    """Provides a description string

      Returns:
        The object description
    """
    s = "Channel '%s'" % self.name
    for sample in self.samples.values() : s += '\n    o ' + str(sample)
    return s

  def load_dict(self, sdict : dict) -> 'Channel' :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

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
         sdict: A dictionary containing markup data
    """
    sdict['name'] = self.name
    sdict['samples'] = []
    for sample in self.samples.values() : sdict['samples'].append(sample.dump_dict())

  def load_data_dict(self, sdict : dict, counts : np.array) :
    """Load observed data information from markup
          raise KeyError("Data channel definition must contain a 'type' field")

    Utility function called from :class:`fastprof.Data` to parse
    an observed data specification for this channel and fill an
    array of event counts

      Args:
        sdict: A dictionary containing markup data
        counts : the array of data counts to fill
    """
    if not 'name' in sdict :
      raise KeyError("Data channel definition must contain a 'name' field")
    if sdict['name'] != self.name :
      raise ValueError("Cannot load channel data defined for channel '%s' into channel '%s'" % (sdict['name'], self.name))
    
  def save_data_dict(self, sdict : dict, counts : np.array) :
    """Save observed data information from markup
    
    Utility function called from :class:`fastprof.Data` to write
    out an observed data specification for this channel into the
    appropriate format

    Args:
      sdict  : A dictionary to fill with markup data
      counts : an array of data counts to read from
    """
    sdict['name'] = self.name
    if len(counts) != self.nbins() :
      raise ValueError("Cannot save channel data counts of length %d for channel '%s' with %d bins" % (len(counts), self.name, self.nbins()))


# -------------------------------------------------------------------------
class BinnedRangeChannel(Channel) :
  """Class representing a model channel consisting of a binned observable range

  Class derived from :class:`Channel` to handle a bin channel consisting
  of a list of bins for a continuous observable.
 
  In this case, each bin is stored as a dict with the format
  { 'lo_edge' : <float value>, 'hi_edge': <float value> }
  providing the lower and upper range of the bin

    * `count` type: dict of the form { 'name' : <bin name> }
  """
  type_str = 'binned_range'

  def __init__(self, name : str = '', bins : list = None, obs_name : str = '', obs_unit : str = '', samples : dict = None) :
    """Initializes the BinnedRangeChannel class

      Args:
         name : channel name
         bins : list of bin definitions, each in the form of a dict
                with format { 'lo_edge' : <float value>, 'hi_edge': <float value> }
    """
    super().__init__(name, samples)
    self.bins = bins if bins is not None else []
    self.obs_name = obs_name
    self.obs_unit = obs_unit

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
        Channel: self
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
         sdict: A dictionary containing markup data
    """
    super().fill_dict(sdict)
    sdict['type'] = BinnedRangeChannel.type_str
    sdict['obs_name'] = self.obs_name
    sdict['obs_unit'] = self.obs_unit
    sdict['bins'] = self.bins
    sdict['samples'] = sdict.pop('samples') # reorder to have samples at end

  def load_data_dict(self, sdict : dict, counts : np.array) :
    """Load observed data information from markup
    
    Utility function called from :class:`fastprof.Data` to parse
    an observed data specification for this channel and fill an
    array of event counts

      Args:
        sdict: A dictionary containing markup data
        counts : the array of data counts to fill
    """
    super().load_data_dict(sdict, counts)
    if not 'name' in sdict :
      raise KeyError("Data channel definition must contain a 'name' field")
    if not 'bins' in sdict :
      raise KeyError("No 'bins' section defined for data channel '%s' in specified markup file." % self.name)
    if len(sdict['bins']) != self.nbins() :
      raise ValueError("Binned range channel '%s' in specified markup file has %d bins, but the model channel has %d." % (channel['name'], len(channel['bins']), self.nbins()))
    for b, bin_data in enumerate(sdict['bins']) :
      if bin_data['lo_edge'] != self.bins[b]['lo_edge'] or bin_data['hi_edge'] != self.bins[b]['hi_edge'] :
        raise ValueError("Bin %d in data channel '%s' spans [%g,%g], but the model bin spans [%g,%g]." %
                         (b, self.name, bin_data['lo_edge'], bin_data['hi_edge'], self.bins[b]['lo_edge'], self.bins[b]['hi_edge']))
      counts[b] = bin_data['counts']

  def save_data_dict(self, sdict : dict, counts : np.array) :
    """Save observed data information from markup
    
    Utility function called from :class:`fastprof.Data` to write
    out an observed data specification for this channel into the
    appropriate format

    Args:
      sdict  : A dictionary to fill with markup data
      counts : an array of data counts to read from
    """
    super().save_data_dict(sdict, counts)
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


# -------------------------------------------------------------------------
class SingleBinChannel(Channel) :
  """Class representing a model channel consisting of a single bin

  Class derived from :class:`Channel` to handle the case of a single 
  counting bin
  
  In this case, each bin is stored as a dict with the format
  { 'name' : <name> }.
  """

  type_str = 'bin'

  def __init__(self, name : str = '', samples : dict = None) :
    """Initializes the BinnedRangeChannel class

      Args:
         name : channel name (and bin name)
    """
    super().__init__(name, samples)

  def nbins(self) -> int :
    """Returns the number of bins in the channel

      Returns:
        the number of bins
    """
    return 1

  def load_dict(self, sdict : dict) :
    """Load object information from a dictionary of markup data

      Args:
        sdict: A dictionary containing markup data

      Returns:
        Channel: self
    """
    super().load_dict(sdict)
    if self.load_field('type', sdict, SingleBinChannel.type_str, str) != SingleBinChannel.type_str :
      raise ValueError("Trying to load a BinnedRangeChannel from channel data of type '%s'" % sdict['type'])
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    super().fill_dict(sdict)
    sdict['type'] = SingleBinChannel.type_str
    sdict['samples'] = sdict.pop('samples') # reorder to have samples at end


  def load_data_dict(self, sdict : dict, counts : np.array) :
    """Load observed data information from markup
    
    Utility function called from :class:`fastprof.Data` to parse
    an observed data specification for this channel and fill an
    array of event counts

      Args:
        sdict: A dictionary containing markup data
        counts : the array of data counts to fill
    """
    super().load_data_dict(sdict, counts)
    if 'type' in sdict and sdict['type'] != SingleBinChannel.type_str :
      raise ValueError("Cannot load channel data defined for type '%s' into channel of type '%s'" % (sdict['type'], SingleBinChannel.type_str))
    if not 'counts' in sdict :
      raise KeyError("No 'counts' section defined for data channel '%s' in specified markup file." % self.name)
    counts[0] = sdict['counts']

  def save_data_dict(self, sdict : dict, counts : np.array) :
    """Save observed data information from markup
    
    Utility function called from :class:`fastprof.Data` to write
    out an observed data specification for this channel into the
    appropriate format

    Args:
      sdict  : A dictionary to fill with markup data
      counts : an array of data counts to read from
    """
    super().save_data_dict(sdict, counts)
    sdict['type'] = SingleBinChannel.type_str
    sdict['counts'] = int(counts[0])


# -------------------------------------------------------------------------
class MultiBinChannel(Channel) :
  """Class representing a model channel consisting of a set of discrete bins

  Class derived from :class:`Channel` to handle a bin channel consisting
  of a set of discrete, names bins.
 
  Each bin is specified by a bin name.
  """
  type_str = 'multi_bin'

  def __init__(self, name : str = '', bins : list = None, samples : dict = None) :
    """Initializes the MultiBinChannel class

      Args:
         name : channel name
         bins : list of bin names
    """
    super().__init__(name, samples)
    self.bins = bins if bins is not None else {}

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
        Channel: self
    """
    super().load_dict(sdict)
    if sdict['type'] != MultiBinChannel.type_str :
      raise ValueError("Trying to load a MultiBinChannel from channel data of type '%s'" % sdict['type'])
    self.bins = sdict['bins']
    return self

  def fill_dict(self, sdict) :
    """Save information to a dictionary of markup data

      Args:
         sdict: A dictionary containing markup data
    """
    super().fill_dict(sdict)
    sdict['type'] = MultiBinChannel.type_str
    sdict['bins'] = self.bins
    sdict['samples'] = sdict.pop('samples') # reorder to have samples at end

  def load_data_dict(self, sdict : dict, counts : np.array) :
    """Load observed data information from markup
    
    Utility function called from :class:`fastprof.Data` to parse
    an observed data specification for this channel and fill an
    array of event counts

      Args:
        sdict: A dictionary containing markup data
        counts : the array of data counts to fill
    """
    super().load_data_dict(sdict, counts)
    if not 'name' in sdict :
      raise KeyError("Data channel definition must contain a 'name' field")
    if not 'bins' in sdict :
      raise KeyError("No 'bins' section defined for data channel '%s' in specified markup file." % self.name)
    if len(sdict['bins']) != self.nbins() :
      raise ValueError("Binned range channel '%s' in specified markup file has %d bins, but the model channel has %d." % (channel['name'], len(channel['bins']), self.nbins()))
    for b, bin_data in sdict['bins'] :
      if bin_data['name'] != self.bins[b] :
        raise ValueError("Bin %d in data channel '%s' has name '%s', but the model specifies name '%s'." % (b, self.name, bin_data['name'], self.bins[b]))
      counts[b] = bin_data['counts']

  def save_data_dict(self, sdict : dict, counts : np.array) :
    """Save observed data information from markup
    
    Utility function called from :class:`fastprof.Data` to write
    out an observed data specification for this channel into the
    appropriate format

    Args:
      sdict  : A dictionary to fill with markup data
      counts : an array of data counts to read from
    """
    super().save_data_dict(sdict, counts)
    sdict['type'] = MultiBinChannel.type_str
    sdict['bins'] = []
    for b, bin_name in enumerate(self.bins) :
      bin_data = {}
      bin_data['name'] = bin_name
      bin_data['counts'] = int(counts[b])
      sdict['bins'].append(bin_data)

