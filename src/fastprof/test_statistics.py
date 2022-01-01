"""Definition of the test statistic forms used in statistical tests


The statistical tests used in this package are based on
test statistics derived from the profile likelihood ratio,
defined in `arXiv:1007.1727 <https://arxiv.org/abs/1007.1727>`_.

The classes below each define one of these test statistics,
with the following implemented so far:

* The :class:`TMu` class defines the :math:`t_{\mu}` test
  statistic of the reference above

* The :class:`QMu` class defines the :math:`q_{\mu}` test
  statistic of the reference above

* The :class:`QMuTilda` class defines the :math:`\tilde{q}_{\mu}` test
  statistic of the reference above

In all cases, the classes are initialized with a set of POI values
corresponding to the tested hypothesis. They implement computations of
the test statistic value, and of the corresponding p-value. Optionally,
they should also be able to compute in the other direction, the
test-statistic value for a given p-value, as well as the asymptotic
distribution of the test statistic under the specified hypothesis.
"""

import math
import scipy.stats
from abc import abstractmethod

# -------------------------------------------------------------------------
class TestStatistic :
  """Base class for test statistic classes

  Provides the basic interface for test statistic objets
  The class is initialized with a set of POI values corresponding
  to the tested hypothesis. It implements computations of
  the test statistic value and of the corresponding p-value.
  Optionally, it can also implement the reverse direction, computing
  the test-statistic value for a given p-value.
  It can also provide the PDF of the test statistic under the
  specified hypothesis.

  Attributes:
    test_poi_values (float) : the POI value defining the tested
                              hypothesis
  """
  def __init__(self, test_poi_values : list) :
    """Initialize the `TestStatistic` object

    Args:
      test_poi_values : the POI value that defines the tested hypothesis
    """
    self.test_poi_values = test_poi_values

  def __float__(self) -> float :
    """Conversion method to `float` value

    Returns:
      the test statistic value
    """
    return value()

  def npois(self) :
    return len(self.test_poi_values)

  @abstractmethod
  def value(self) :
    """Value of the test statistic

    Returns:
      the test statistic value
    """
    pass

  @abstractmethod
  def asymptotic_pv(self, ts : float = None) -> float :
    """Asymptotic p-value corresponding
       to the test-statistic value

    Args:
      ts : an alternate test statistic value. If not `None`,
           the p-value is computed for this one, otherwise
           for the value returned by :meth:`value`.

    Returns:
      the asymptotic p-value
    """
    pass

  @abstractmethod
  def asymptotic_pdf(self, ts : float = None) -> float :
    """Value of the PDF of the test statistic, under
       asymptotic assumptions

    Args:
      ts : an alternate test statistic value. If not `None`,
           the PDF value is computed for this one, otherwise
           for the value returned by :meth:`value`.

    Returns:
      the local value of the asymptotic PDF
    """
    pass

  @abstractmethod
  def asymptotic_ts(self, pv : float) -> float :
    """Test statistic value corresponding to a given
       p-value, under asymptotic assumptions

    Args:
      pv : an asymptptic p-value.

    Returns:
      the corresponding test statistic value
    """
    pass


class TMu(TestStatistic) :
  """The :math:`t_{\mu}` (profile-likelihood ratio) test statistic

  Attributes:
    tmu (float) : the value of the profile-likelihood ratio :math:`-2\Delta\log L`
  """

  def __init__(self, test_poi_values : list, tmu : float) :
    """Initialize the `TMu` object

    Args:
      test_poi : the POI value that defines the tested hypothesis
      tmu : the value of the profile likelihood ratio :math:`-2\Delta\log L`
    """
    super().__init__(test_poi_values)
    self.tmu = tmu

  def value(self) -> float :
    """Value of the test statistic

    Returns the stored value of `tmu`.

    Returns:
      the test statistic value
    """
    return self.tmu

  def asymptotic_pv(self, ts : float = None) -> float :
    """Asymptotic p-value corresponding
       to the test-statistic value

    Args:
      ts : an alternate test statistic value. If not `None`,
           the p-value is computed for this one, otherwise
           for the value returned by :meth:`value`.

    Returns:
      the asymptotic p-value
    """
    if ts == None : ts = self.value()
    return scipy.stats.chi2.sf(ts, self.npois())

  def asymptotic_pdf(self, ts) :
    """Value of the PDF of the test statistic, under
       asymptotic assumptions

    Args:
      ts : an alternate test statistic value. If not `None`,
           the PDF value is computed for this one, otherwise
           for the value returned by :meth:`value`.

    Returns:
      the local value of the asymptotic PDF
    """
    if ts == None : ts = self.value()
    return scipy.stats.chi2(math.sqrt(ts), 1)

  def asymptotic_ts(self, pv) :
    """Test statistic value corresponding to a given
       p-value, under asymptotic assumptions

    Args:
      pv : an asymptptic p-value.

    Returns:
      the corresponding test statistic value
    """
    return scipy.stats.norm.isf(pv)


class QMu(TestStatistic) :
  """The :math:`q_{\mu}` test statistic

  The form implemented here corresponds to the "uncapped" version,
  which has values of -tmu for POI values above the hypothesis, instead
  of 0 as defined in `arXiv:1007.1727 <https://arxiv.org/abs/1007.1727>`_.

  Attributes:
    tmu (float) : the value of the profile-likelihood ratio :math:`-2\Delta\log L`
    best_poi (float) : the best-fit value of the POI
    comp_poi (float) : the value of the POI (`mu`) used in the computation of `tmu`.
    tmu_Amu (float)    : the value of `tmu` computed on an Asimov dataset generated
                       under the tested hypothesis.
    sigma (float)    : the Asimov uncertainty on the POI, an alternate way to
                       provide tmu_Amu
  """

  def __init__(self, test_poi : float, tmu : float, best_poi : float, comp_poi  : float = None,
               tmu_Amu  : float = None, sigma  : float = None) :
    """Initialize the `QMu` object

    Args:
      test_poi : the POI value that defines the tested hypothesis
      tmu : the value of the profile likelihood ratio :math:`-2\Delta\log L`
      best_poi : the best-fit value of the POI
      comp_poi : the value of the POI used in the computation of `tmu`. If
                 `None` (default), assumed to be the same as `test_poi`.
      tmu_Amu  : the value of `tmu` computed on an Asimov dataset generated
                 under the tested hypothesis.
      sigma    : the Asimov uncertainty on the POI, an alternate way to
                 provide tmu_Amu
    """
    super().__init__([ test_poi ])
    self.tmu = tmu
    self.best_poi = best_poi
    self.comp_poi = comp_poi if comp_poi != None else self.test_poi()
    self.tmu_Amu = tmu_Amu # Should correspond to mu' = test_poi (when test_poi != comp_poi). In practice mu'=0 to compute CL_b
    self.sigma = sigma

  def test_poi(self) :
    """Tested values of the POI

    Helper function to get the first (and here only) POI value, since `QMu` implements
    a single POI, whereas the general `TestStatistic` class allows multiple.

    Returns:
      the test statistic value
    """
    return self.test_poi_values[0]

  def value(self) :
    """Value of the test statistic

    Implements the computation of :math:`q_{\mu}` : return `tmu` if
    the best-fit POI is below the hypothesis value, 0 otherwise.

    Returns:
      the test statistic value
    """
    return self.tmu if self.best_poi < self.comp_poi else -self.tmu

  @staticmethod
  def signed_sqrt(x : float) -> float :
    """Signed square-root function

    Args:
      x : a numerical value

    Returns:
      sign(x)*sqrt(abs(x))
    """
    return math.sqrt(x) if x >= 0 else -math.sqrt(-x)

  def non_centrality_parameter(self) :
    """The non-centrality parameter of the asymptotic
       chi2 distribution of the test statistic, under
       the POI=0 hypothesis

    In general the non-centrality parameter is defined as
    :math:`\Lambda = t_{\mu, A(\mu')} = (\mu - \mu')^2/\sigma^2`,
    see `arXiv:1007.1727 <https://arxiv.org/abs/1007.1727>`_.

    The result can either be obtained from the value of
    :math:`t_{\mu, A(\mu')}` provided by the `tmu_Amu`
    parameter at initialization, or from a value of
    :math:`\sigma` provided by `sigma`.

    Returns:
      the value of :math:`t_{\mu, A(0)}`
    """
    if self.comp_poi == self.test_poi() : return 0
    if self.tmu_Amu != None :
      if self.tmu_Amu < 0 :
        print('WARNING: tmu_Amu = % g < 0, returning 0' % self.tmu_Amu)
        return 0
      return self.tmu_Amu # Must be the right value for test_poi! (=> tmu_Amu and comp_poi should be set together consistently)
    elif self.sigma != None :
      return ((self.comp_poi - self.test_poi())/self.sigma)**2
    else :
      raise ValueError('Should supply either tmu_Amu or sigma for the asymptotic CL_s computation')

  def asymptotic_pv(self, ts : float = None) -> float :
    """Asymptotic p-value corresponding
       to the test-statistic value

    Args:
      ts : an alternate test statistic value. If not `None`,
           the p-value is computed for this one, otherwise
           for the value returned by :meth:`value`.

    Returns:
      the asymptotic p-value
    """
    if ts == None : ts = self.value()
    return scipy.stats.norm.sf(self.signed_sqrt(ts) - math.sqrt(self.non_centrality_parameter()))

  def asymptotic_pdf(self, ts : float = None) -> float :
    """Value of the PDF of the test statistic, under
       asymptotic assumptions

    Args:
      ts : an alternate test statistic value. If not `None`,
           the PDF value is computed for this one, otherwise
           for the value returned by :meth:`value`.

    Returns:
      the local value of the asymptotic PDF
    """
    if ts == None : ts = self.value()
    pdf_q = scipy.stats.norm.pdf(self.signed_sqrt(ts) - math.sqrt(self.non_centrality_parameter()))
    return pdf_q/2/math.sqrt(abs(ts)) if ts != 0 else 0

  def asymptotic_ts(self, pv : float) -> float :
    """Test statistic value corresponding to a given
       p-value, under asymptotic assumptions

    Args:
      pv : an asymptptic p-value.

    Returns:
      the corresponding test statistic value
    """
    q1 = scipy.stats.norm.isf(pv) + math.sqrt(self.non_centrality_parameter())
    return q1*abs(q1)

  def asymptotic_clb(self) -> float :
    """return the :math:'CL_b` value for the
       current test statistic value

    Returns:
       the :math:'CL_b` value
    """
    return QMu(0, self.tmu, self.best_poi, self.comp_poi, self.tmu_Amu, self.sigma).asymptotic_pv()

  def asymptotic_cls(self) :
    """return the :math:'CL_s` value for the
       current test statistic value

    Returns:
       the :math:'CL_s` value
    """
    clsb = self.asymptotic_pv()
    cl_b = self.asymptotic_clb()
    #print('Asymptotic CLs = %g/%g = %g' % (clsb, cl_b, clsb/cl_b))
    return clsb/cl_b if cl_b > 0 else 1


class QMuTilda(QMu) :
  """The :math:`\tilde{q}_{\mu}` test statistic

  The class derives from :class:`QMu`, since this form of test statistic
  is closely related to :math:`q_{\mu}`.

  The form implemented here corresponds to the "uncapped" version,
  which has values of -tmu for POI values above the hypothesis, instead
  of 0 as defined in `arXiv:1007.1727 <https://arxiv.org/abs/1007.1727>`_.

  Attributes:
    tmu_0 (float) : the value of `tmu` computed with POI=0 instead of
                    if the best-fit value, used to define the test statistic
                    value if the best-fit value is negative
    tmu_A0 (float) : the value of `tmu` computed on an Asimov dataset generated
                     under the POI=0 hypothesis.
  """
  def __init__(self, test_poi : float, tmu : float, best_poi : float, tmu_0 : float = None, comp_poi : float = None,
               tmu_Amu : float = None, tmu_A0 : float = None) :
    """Initialize the `QMuTilda` object

    Requires both the likelihood ratio :math:`-2\log(L(test_poi)/L(best_poi))`
    evaluated at the best-fit POI, and :math:`-2\log(L(test_poi)/L(poi=0))`
    evaluated at POI=0, since both are used in the definition of
    :math:`\tilde{q}_{\mu}`. However if the POI is constrained to be positive
    then `tmu_0` is not required, since it is then equal to `tmu` in the cases
    where it would be used.

    Args:
      test_poi : the POI value that defines the tested hypothesis
      tmu : the value of the profile likelihood ratio :math:`-2\log(L(test_poi)/L(best_poi))`
      best_poi : the best-fit value of the POI
      tmu_0 : the value of the profile likelihood ratio :math:`-2\log(L(test_poi)/L(poi=0))`
              If not provided, assumed to be equal to `tmu`.
      comp_poi : the value of the POI used in the computation of `tmu`. If
                 `None` (default), assumed to be the same as `test_poi`.
      tmu_Amu    : the value of `tmu` computed on an Asimov dataset generated
                 under the tested hypothesis.
      tmu_A0    : the value of `tmu` computed on an Asimov dataset generated
                 under the POI=0 hypothesis.
    """
    super().__init__(test_poi, tmu, best_poi, comp_poi, tmu_Amu)
    self.tmu_0 = tmu_0 if tmu_0 is not None else tmu
    self.tmu_A0 = tmu_A0 # corresponds to the mu=0 hypo

  def value(self) :
    """Value of the test statistic

    Implements the computation of :math:`\tilde{q}_{\mu}` : return `tmu` if
    the best-fit POI is below the hypothesis value but above 0, -tmu if
    the best-fit POI is above the hypothesis value, and tmu_0 if it is
    negative.

    Returns:
      the test statistic value
    """
    return self.tmu_0 if self.best_poi < 0 else self.tmu if self.best_poi < self.comp_poi else -self.tmu

  def threshold(self) : # mu^2/sigma^2
    """return the non-centrality parameter for a test
       hypothesis POI=0

    This parameter corresponds to
    :math:`\Lambda = t_{\mu, A(\mu'=0)} = \mu^2/\sigma^2,
    and is used in the computation of the asymptotic
    p-value. It may differ from the result of
    :meth:`non_centrality_parameter`, which can in principle
    refer to other hypotheses (depending on the parameters
    provided at initialization), although it practice they
    often both refer to the POI=0 case.

    The function returns the value that was provided during
    initialization.

    Returns:
       the non-centrality parameter
    """
    if self.tmu_A0 < 0 :
      print('WARNING: q~mu threshold = % g < 0, returning 0' % self.tmu_A0)
      return 0
    return self.tmu_A0

  def asymptotic_pv(self, ts : float = None) -> float :
    """Asymptotic p-value corresponding
       to the test-statistic value

    Args:
      ts : an alternate test statistic value. If not `None`,
           the p-value is computed for this one, otherwise
           for the value returned by :meth:`value`.

    Returns:
      the asymptotic p-value
    """
    if ts == None : ts = self.value()
    if ts < self.threshold() :
      return QMu.asymptotic_pv(self, ts)
    else :
      return scipy.stats.norm.sf((ts + self.threshold())/(2*math.sqrt(self.threshold())) - math.sqrt(self.non_centrality_parameter()))

  def asymptotic_pdf(self, ts : float = None) -> float :
    """Value of the PDF of the test statistic, under
       asymptotic assumptions

    Args:
      ts : an alternate test statistic value. If not `None`,
           the PDF value is computed for this one, otherwise
           for the value returned by :meth:`value`.

    Returns:
      the local value of the asymptotic PDF
    """
    if ts == None : ts = self.value()
    if ts < self.threshold() :
      return QMu.asymptotic_pdf(self, ts)
    else :
      pdf_q = scipy.stats.norm.pdf((ts + self.threshold())/(2*math.sqrt(self.threshold())) - math.sqrt(self.non_centrality_parameter()))
      return pdf_q/(2*math.sqrt(self.threshold())) if self.threshold() > 0 else 0

  def asymptotic_ts(self, pv : float) -> float :
    """Test statistic value corresponding to a given
       p-value, under asymptotic assumptions

    Args:
      pv : an asymptptic p-value.

    Returns:
      the corresponding test statistic value
    """
    q1 = scipy.stats.norm.isf(pv) + math.sqrt(self.non_centrality_parameter())
    if q1 < self.threshold() :
      return q1*abs(q1)
    else :
      return q1*2*math.sqrt(self.threshold()) - self.threshold()

  def asymptotic_clb(self) -> float :
    """return the :math:'CL_b` value for the
       current test statistic value

    Returns:
       the :math:'CL_b` value
    """
    return QMuTilda(0, tmu=self.tmu, best_poi=self.best_poi, comp_poi=self.comp_poi, tmu_Amu=self.tmu_A0, tmu_A0=self.tmu_A0).asymptotic_pv()
