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
    test_poi (float) : the POI value defining the tested
                       hypothesis
  """
  def __init__(self, test_poi) : 
    self.test_poi = test_poi
  def __float__(self) :
    return value()
  @abstractmethod
  def value(self) :
    pass  
  @abstractmethod
  def asymptotic_pv(self, ts = None) :
    pass
  @abstractmethod
  def asymptotic_pdf(self, ts = None) :
    pass
  @abstractmethod
  def asymptotic_ts(self, pv) :
    pass
    

class TMu(TestStatistic) :
  """The :math:`t_{\mu}` (profile-likelihood ratio) test statistic
    
  Attributes:
    value (float) : the value of the profile-likelihood ratio :math:`-2\Delta\log L`
  """
  def __init__(self, test_poi, twice_dll) :
    super().__init__(test_poi)
    self.value = twice_dll
  def value(self) : 
    return self.value
  def asymptotic_pv(self, ts = None) :
    if ts == None : ts = self.value()
    return scipy.stats.norm.sf(math.sqrt(ts))
  def asymptotic_pdf(self, ts) :
    if ts == None : ts = self.value()
    return scipy.stats.chi2(math.sqrt(ts), 1)
  def asymptotic_ts(self, pv) :
    return scipy.stats.norm.isf(pv)


class QMu(TestStatistic) :
  """The :math:`q_{\mu}` test statistic
    
  Attributes:
    tmu (float) : the value of the profile-likelihood ratio :math:`-2\Delta\log L`
    best_poi (float) : the best-fit value of the POI
    comp_poi (float) : the value of the POI (`mu`) used in the computation of `tmu`.
    tmu_A (float)    : the value of `tmu` computed on an Asimov dataset generated
                       under the tested hypothesis.
    sigma (float)    : the Asimov uncertainty on the POI, an alternate way to
                       provide tmu_A
  """
  def __init__(self, test_poi, tmu, best_poi, comp_poi = None, tmu_A = None, sigma = None) :
    super().__init__(test_poi)
    self.tmu = tmu
    self.best_poi = best_poi
    self.comp_poi = comp_poi if comp_poi != None else self.test_poi
    self.tmu_A = tmu_A # Should correspond to mu' = self.test_poi (when test_poi != comp_poi). In practice mu'=0 to compute CL_b
    self.sigma = sigma

  def value(self) :
    return self.tmu if self.best_poi < self.comp_poi else -self.tmu

  @staticmethod
  def signed_sqrt(x) :
    return math.sqrt(x) if x >= 0 else -math.sqrt(-x)

  def non_centrality_parameter(self) : # tmu_A = (mu - mu')^2/sigma^2, here for mu'=0 for the CL_b computation
    if self.comp_poi == self.test_poi : return 0
    if self.tmu_A != None :
      if self.tmu_A < 0 :
        print('WARNING: tmu_A = % g < 0, returning 0' % self.tmu_A)
        return 0
      return self.tmu_A # Must be the right value for test_poi! (=> tmu_A and comp_poi should be set together consistently)
    elif self.sigma != None :
      return ((self.comp_poi - self.test_poi)/self.sigma)**2
    else :
      raise ValueError('Should supply either tmu_A or sigma for the asymptotic CL_s computation')

  def asymptotic_pv(self, ts = None) :
    if ts == None : ts = self.value()
    return scipy.stats.norm.sf(self.signed_sqrt(ts) - math.sqrt(self.non_centrality_parameter()))

  def asymptotic_pdf(self, ts = None) :
    if ts == None : ts = self.value()
    pdf_q = scipy.stats.norm.pdf(self.signed_sqrt(ts) - math.sqrt(self.non_centrality_parameter()))
    return pdf_q/2/math.sqrt(abs(ts)) if ts != 0 else 0

  def asymptotic_ts(self, pv) :
    q1 = scipy.stats.norm.isf(pv) + math.sqrt(self.non_centrality_parameter())
    return q1*abs(q1)

  def asymptotic_clb(self) :
    return QMu(0, self.tmu, self.best_poi, self.comp_poi, self.tmu_A, self.sigma).asymptotic_pv()

  def asymptotic_cls(self) :
    clsb = self.asymptotic_pv()
    cl_b = self.asymptotic_clb()
    #print('Asymptotic CLs = %g/%g = %g' % (clsb, cl_b, clsb/cl_b))
    return clsb/cl_b if cl_b > 0 else 1

class QMuTilda(QMu) :
  """The :math:`\tilde{q}_{\mu}` test statistic
    
  The class derives from :class:`QMu`, since this form of test statistic
  is closely related to :math:`q_{\mu}`.
    
  Attributes:
    tmu_0 (float)    : the value of `tmu` computed on an Asimov dataset generated
                       under the POI=0 hypothesis.
  """
  def __init__(self, test_poi, tmu, best_poi, comp_poi = None, tmu_A = None, tmu_0 = None) :
    super().__init__(test_poi, tmu, best_poi, comp_poi, tmu_A)
    self.tmu_0 = tmu_0 # corresponds to the mu=0 hypo

  def threshold(self) : # mu^2/sigma^2
    if self.tmu_0 < 0 :
      print('WARNING: q~mu threshold = % g < 0, returning 0' % self.tmu_0)
      return 0
    return self.tmu_0

  def asymptotic_pv(self, ts = None) :
    if ts == None : ts = self.value()
    if ts < self.threshold() :
      return QMu.asymptotic_pv(self, ts)
    else :
      return scipy.stats.norm.sf((ts + self.threshold())/(2*math.sqrt(self.threshold())) - math.sqrt(self.non_centrality_parameter()))

  def asymptotic_pdf(self, ts = None) :
    if ts == None : ts = self.value()
    if ts < self.threshold() :
      return QMu.asymptotic_pdf(self, ts)
    else :
      pdf_q = scipy.stats.norm.pdf((ts + self.threshold())/(2*math.sqrt(self.threshold())) - math.sqrt(self.non_centrality_parameter()))
      return pdf_q/(2*math.sqrt(self.threshold())) if self.threshold() > 0 else 0

  def asymptotic_ts(self, pv) :
    q1 = scipy.stats.norm.isf(pv) + math.sqrt(self.non_centrality_parameter())
    if q1 < self.threshold() :
      return q1*abs(q1)
    else :
      return q1*2*math.sqrt(self.threshold()) - self.threshold()

  def asymptotic_clb(self) :
    return QMuTilda(0, self.tmu, self.best_poi, self.comp_poi, self.tmu_0, self.tmu_0).asymptotic_pv()
