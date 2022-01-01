"""
Classes implementing the computation of test statistics.

  The classes are

  * :class:`TestStatisticCalculator` : perform
    test statistic and p-value computations using
    :class:`PLRData` storage. Derived classes are
    defined for the main test statistics (for now,
    :class:`test_statistics.QMu` and
    :class:`test_statistics.QMuTilda`.

"""

from abc import abstractmethod
from timeit import default_timer as timer

from .core import Model, Data, Parameters, ModelPOI
from .fit_data import POIHypo, FitResult, PLRData, Raster
from .minimizers import NPMinimizer, POIMinimizer, OptiMinimizer
from .test_statistics import TMu, QMu, QMuTilda


class TestStatisticCalculator :
  """Base class for test statistic computations

  Base class for classes implementing the computation
  of various test statistics. This includes

  * computing the `tmu` test statistic from a linear
     model (included in the minimizer) and a dataset

  * computing other test statistics from PLR data obtained
    either from the previous step, or externally,
    e.g. using :mod:`fit_ws.py`

  * Filling p-value information in PLR data
    objects based on the test statistic
    information.

  Each step operates on a :class:`PLRData` object,
  which is either created or filled with more data.

  The test statistics themselves are defined in the
  :mod:`test_statistics.py` module. Each type
  corresponds to a separate class derived from
  :class:`TestStatisticCalculator`.

  Attributes:
    minimizer (POIMinimizer) : a minimizer object
      (see :mod:`minimizers.py`).
  """

  def __init__(self, minimizer : POIMinimizer) :
    """Initialize the `TestStatisticCalculator` object

    Args:
      minimizer : a minimizer algorithm.
    """
    self.minimizer = minimizer

  @classmethod
  def pois(cls, plr_data : PLRData) -> str  :
    """Return the name of the POIs in PLR data

    Note that for now only the single-POI case is
    supported.

    Args:
      plr_data : an object storing PLR information
    Returns:
      the POI name
    """
    return list(plr_data.pois)

  @classmethod
  def poi(cls, plr_data : PLRData, index : int = 0) -> str  :
    """Return the name of the POI in PLR data

    Note that for now only the single-POI case is
    supported.

    Args:
      plr_data : an object storing PLR information
    Returns:
      the POI name
    """
    pois = cls.pois(plr_data)
    if index >= len(pois) : raise KeyError('Cannot access POI with index %d, only %d are defined' % (index, len(pois)))
    return pois[index]

  @abstractmethod
  def fill_pv(self, plr_data : PLRData) :
    """Fills p-value information from PLR data

    Abstract method to be reimplemented in derived classes
    corresponding to a specific test statistic.

    Args:
      plr_data : an object storing PLR information
    """
    pass

  def compute_fast_plr(self, hypo : POIHypo, data : Data, full_hypo : Parameters = None, name : str = 'fast') -> PLRData :
    """Compute `tmu` from a dataset

    The computation is performed in the linear approximation
    using `self.minimizer` (which contains the model
    information) on the provided dataset, at the specified
    hypothesis value.
    A class:`PLRData` object is created with the computed
    PLR information, and this is then used to compute
    the `tmu` value.

    Args:
      hypo : the hypothesis for which to compute the PLR
      data : the input dataset
      name : name of the output :class:`PLRData` object
    Returns:
      an object containing the PLR information
    """
    plr_data = PLRData(name, hypo, full_hypo=full_hypo, model=data.model)
    tmu = self.minimizer.tmu(hypo.pars, data, full_hypo)
    plr_data.test_statistics['tmu'] = tmu
    plr_data.free_fit = FitResult('free_fit', self.minimizer.free_pars, self.minimizer.free_nll, model=data.model)
    plr_data.hypo_fit = FitResult('hypo_fit', self.minimizer.hypo_pars, self.minimizer.hypo_nll, model=data.model)
    plr_data.update() # includes `tmu` computation
    return plr_data

  def compute_fast_q(self, hypo : POIHypo, data : Data, full_hypo : Parameters = None, name : str = 'fast') -> PLRData :
    """Compute `tmu` and the associated p-value from a dataset

    Computes `tmu` using the :meth:`compute_fast_plr`
    method above, and adds the Asimov dataset values and
    the p-value computation from :meth:`fill_pv`.

    Args:
      hypo : the hypothesis for which to compute the PLR
      data : the input dataset
      name : name of the output :class:`PLRData` object
    Returns:
      an object containing the PLR information
    """
    fast_plr_data = self.compute_fast_plr(hypo, data, full_hypo, name)
    asimov = data.model.generate_expected(0, NPMinimizer(data))
    asimov_plr_data = self.compute_fast_plr(hypo, asimov, full_hypo, 'fast_asimov')
    fast_plr_data.set_asimov(asimov_plr_data)
    self.fill_pv(fast_plr_data)
    return fast_plr_data

  def fill_all_pv(self, raster : Raster) :
    """Compute p-values in a raster containing PLR values

    Takes as input a raster object containing PLR values at each
    hypothesis point and compute the associated p-values

    Args:
      raster : a raster object
    Returns:
      self
    """
    for plr_data in raster.plr_data.values() : self.fill_pv(plr_data)

  def compute_fast_results(self, hypos : list, data : Data, full_hypos : dict = {}, name : str = 'fast', verbosity : int = 0) -> Raster :
    """Compute fast PLR and p-values for a set of hypotheses

    Builds a raster object with the same hypothesis points as the one
    provided in input, containing PLR data and p-values computing
    using a linear model.

    Args:
      hypos  : a list of POIHypo objects
      data   : the input dataset
      name   : name of the output :class:`PLRData` objects
      verbosity : the levelf of verbosity of the output (0 or 1 currently)
    Returns:
      a raster object containing the fast PLR and p-value results
    """
    fast_plr_data = {}
    start_time = timer()
    for i, hypo in enumerate(hypos) :
      if verbosity > 1 :
        print('Processing hypothesis point %d of %d : %s %s' % (i+1, len(hypos), str(hypo), ('[so far %g s/point]' % ((timer() - start_time)/i)) if i > 0 else ''))
      fast_plr_data[hypo] = self.compute_fast_q(hypo, data, full_hypos[hypo] if hypo in full_hypos else None, '%s_%g' % (name, i))
    fast = Raster(name, fast_plr_data, model=data.model)
    self.fill_all_pv(fast)
    return fast

  def recompute_raster(self, raster : Raster, data : Data, name : str = 'fast') -> Raster :
    """Compute fast PLR and p-values for a set of hypotheses

    Builds a raster object with the same hypothesis points as the one
    provided in input, containing PLR data and p-values computing
    using a linear model.

    Args:
      raster : a raster object
      data   : the input dataset
      name   : name of the output :class:`PLRData` objects
    Returns:
      a raster object containing the fast PLR and p-value results
    """
    return self.compute_fast_results(raster.plr_data.keys(), data, { hypo : plr_data.full_hypo for hypo, plr_data in raster.plr_data.items() if plr_data.full_hypo is not None }, name)


class TMuCalculator(TestStatisticCalculator) :
  """Calculator class for :math:`t_{\mu}`

  Implements test statistic and p-value
  computations in :class:`PLRData` objects for
  the :math:`t_{\mu}` test statistic defined in
  the :mod:`test_statistics.py` module.
  """

  def __init__(self, minimizer : POIMinimizer) :
    """Initialize the `TMuCalculator` object

    Args:
      minimizer : a minimizer algorithm.
    """
    super().__init__(minimizer)

  @classmethod
  def make_q(cls, plr_data : PLRData) -> TMu :
    """Builds a :class:`QMu` test statistic object from PLR data

    Args:
      cls : the TMuCalculator class object (classmethod input)
      plr_data : an object containing PLR information
    Returns:
      the test statistic
    """
    return TMu([ plr_data.hypo[poi] for poi in cls.pois(plr_data) ], tmu=plr_data.test_statistics['tmu'])

  def fill_pv(self, plr_data : PLRData) -> 'TMuCalculator' :
    """Fills p-value information from PLR data

    Builds a :class:`TMu` test statistic from the :class:`PLRData`
    object provided as input, computes the associated test statistic
    value and p-value, and stores these in the :class:`PLRData`
    object.

    Args:
      plr_data : an object storing PLR information
    Returns:
      self
    """
    try :
      q = self.make_q(plr_data)
      plr_data.pvs['pv' ] = q.asymptotic_pv()
    except Exception as inst:
      print("t_mu computation failed for PLR '%s', hypothesis %s, with exception below:" % (plr_data.name, plr_data.hypo))
      raise(inst)
    return self

  def compute_fast_results(self, hypos : list, data : Data, full_hypos : dict = {}, name : str = 'fast', free_fit : str = 'all', verbosity : int = 0) -> Raster :
    """Compute fast PLR and p-values for a set of hypotheses

    Reimplements the general method since it can be done more simply for t_mu
    
    Args:
      raster    : a raster object
      data      : the input dataset
      name      : name of the output :class:`PLRData` objects
      verbosity : the levelf of verbosity of the output (0 or 1 currently)
      free_fit  : can be 'all' (default, same as general method), 'single' (do a single
                  fit near the best fixed fit) of 'best_fixed_fit' (just take the best fixed fit) 
    Returns:
      a raster object containing the fast PLR and p-value results
    """
    if free_fit not in [ 'all', 'single', 'best_fixed_fit' ] :
      raise ValueError("'free_fit' argument to compute_fast_results should be one of 'all', 'single', 'best_fixed_fit', got '%s'" % free_fit)
    fast_plr_data = {}
    fixed_pars = {}
    start_time = timer()
    for i, hypo in enumerate(hypos) :
      if verbosity >= 1 :
        print('Processing hypothesis point %d of %d : %s %s' % (i+1, len(hypos), str(hypo), ('[so far %g s/point]' % ((timer() - start_time)/i)) if i > 0 else ''))
      full_hypo = full_hypos[hypo] if hypo in full_hypos else None
      plr_data = PLRData(name, hypo, full_hypo=full_hypo, model=data.model)
      hypo_minimizer = self.minimizer.clone()
      self.minimizer.set_pois(data.model, init_pars=full_hypo, hypo=hypo.pars, fix_hypo=False)
      hypo_minimizer.set_pois(data.model, init_pars=full_hypo, hypo=hypo.pars, fix_hypo=True)
      # Hypo fit
      if hypo_minimizer.minimize(data) is None : return None
      fixed_pars[hypo] = hypo_minimizer.min_pars
      plr_data.hypo_fit = FitResult('hypo_fit', hypo_minimizer.min_pars, hypo_minimizer.min_nll, model=data.model)
      # Free fit
      if free_fit == 'all' :
        if self.minimizer.minimize(data) is None : return None
        plr_data.free_fit = FitResult('free_fit', self.minimizer.min_pars, self.minimizer.min_nll, model=data.model)
      fast_plr_data[hypo] = plr_data
    stop_time = timer()
    if verbosity >= 1 : print('Processed %d hypothesis points in %g s (%g s/point)' % (len(hypos), (stop_time - start_time), (stop_time - start_time)/len(hypos)))
    if free_fit == 'single' or free_fit == 'best_fixed_fit' :
      best_fixed = min(fast_plr_data.items(), key=lambda plr_data : plr_data[1].hypo_fit.nll)
      if free_fit == 'best_fixed_fit' :
        common_free_fit = FitResult('free_fit', fixed_pars[best_fixed[0]], best_fixed[1].hypo_fit.nll, model=data.model)
      else :
        start_time = timer()
        if verbosity >= 1 : print('Performing global free-POI fit at hypothesis %s' % best_fixed[0])
        self.minimizer.set_pois(data.model, hypo=best_fixed[0])
        if self.minimizer.minimize(data) is None : return None
        if verbosity >= 1 : print('Performed free-POI fit in %g s' % (timer() - start_time))
        common_free_fit = FitResult('free_fit', self.minimizer.min_pars, self.minimizer.min_nll, model=data.model)
      for hypo in hypos :
        fast_plr_data[hypo].free_fit = common_free_fit
    for hypo in hypos :
      fast_plr_data[hypo].update() # includes `tmu` computation
    fast = Raster(name, fast_plr_data, model=data.model)
    self.fill_all_pv(fast)
    return fast


class QMuCalculator(TestStatisticCalculator) :
  """Calculator class for :math:`q_{\mu}`

  Implements test statistic and p-value
  computations in :class:`PLRData` objects for
  the :math:`q_{\mu}` test statistic defined in
  the :mod:`test_statistics.py` module.
  """

  def __init__(self, minimizer : POIMinimizer) :
    """Initialize the `QMuCalculator` object

    Args:
      minimizer : a minimizer algorithm.
    """
    super().__init__(minimizer)

  @classmethod
  def make_q(cls, plr_data : PLRData) -> QMu :
    """Builds a :class:`QMu` test statistic object from PLR data

    Args:
      cls : the QMuCalculator class object (classmethod input)
      plr_data : an object containing PLR information
    Returns:
      the test statistic
    """
    return QMu(test_poi = plr_data.hypo[cls.poi(plr_data)], tmu = plr_data.test_statistics['tmu'],
               best_poi = plr_data.free_fit.fitpars[cls.poi(plr_data)].value, tmu_Amu = plr_data.test_statistics['tmu_A0'])

  def fill_pv(self, plr_data : PLRData) -> 'QMuCalculator' :
    """Fills p-value information from PLR data

    Builds a :class:`QMu` test statistic from the :class:`PLRData`
    object provided as input, computes the associated test statistic
    value and p-value, and stores these in the :class:`PLRData`
    object.

    Args:
      plr_data : an object storing PLR information
    Returns:
      self
    """
    try :
      # since we use tmu_Amu to compute CLb, we need tmu_Amu = tmu_A0 (computed from an Asimov with mu'=0)
      q = self.make_q(plr_data)
      plr_data.test_statistics['q_mu'] = q.value()
      plr_data.pvs['pv' ] = q.asymptotic_pv()
      plr_data.pvs['cls'] = q.asymptotic_cls()
      plr_data.pvs['clb'] = q.asymptotic_clb()
    except Exception as inst:
      print("q_mu computation failed for PLR '%s', hypothesis %s, with exception below:" % (plr_data.name, plr_data.hypo))
      raise(inst)
    return self


class QMuTildaCalculator(TestStatisticCalculator) :
  """Calculator class for :math:`\tilde{q}_{\mu}`

  Implements test statistic and p-value
  computations in :class:`PLRData` objects for
  the :math:`\tilde)q}_{\mu}` test statistic defined
  in the :mod:`test_statistics.py` module.
  """
  def __init__(self, minimizer) :
    """Initialize the `QMuTildaCalculator` object

    Args:
      minimizer : a minimizer algorithm.
    """
    super().__init__(minimizer)
    self.qs = []

  @classmethod
  def make_q(cls, plr_data) :
    """Builds a :class:`QMuTilda` test statistic object from PLR data

    Args:
      cls : the QMuTildaCalculator class object (classmethod input)
      plr_data : an object containing PLR information
    Returns:
      the test statistic
    """
    return QMuTilda(test_poi = plr_data.hypo[cls.poi(plr_data)], tmu = plr_data.test_statistics['tmu'],
                    best_poi = plr_data.free_fit.fitpars[cls.poi(plr_data)].value, tmu_Amu = plr_data.test_statistics['tmu_A0'],
                    tmu_A0 = plr_data.test_statistics['tmu_A0'])

  def fill_pv(self, plr_data) :
    """Fills p-value information from PLR data

    Builds a :class:`QMuTilda` test statistic from the :class:`PLRData`
    object provided as input, computes the associated test statistic
    value and p-value, and stores these in the :class:`PLRData`
    object.

    Args:
      plr_data : an object storing PLR information
    Returns:
      self
    """
    try :
      # since we use tmu_Amu to compute CLb, we need tmu_Amu = tmu_A0 (computed from an Asimov with mu'=0)
      q = self.make_q(plr_data)
      plr_data.test_statistics['q~mu'] = q.value()
      plr_data.pvs['pv' ] = q.asymptotic_pv()
      plr_data.pvs['cls'] = q.asymptotic_cls()
      plr_data.pvs['clb'] = q.asymptotic_clb()
    except Exception as inst:
      print("q~mu computation failed for computation '%s', hypothesis %s, with exception below:" % (plr_data.name, plr_data.hypo))
      raise(inst)
    return self
