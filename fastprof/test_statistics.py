import math
import scipy.stats
from abc import abstractmethod

# -------------------------------------------------------------------------
class TestStatistic :
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
