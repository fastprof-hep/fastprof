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
  def asymptotic_cl(self) :
    pass
    

class TMu(TestStatistic) :
  def __init__(self, test_poi, twice_dll) :
    super().__init__(test_poi)
    self.value = twice_dll
  def value(self) : 
    return self.value
  def asymptotic_cl(self) :
    return scipy.stats.norm.sf(math.sqrt(self.value))


class QMu(TestStatistic) :
  def __init__(self, test_poi, tmu, best_poi, comp_poi = None, tmu_A = None, sigma = None) :
    super().__init__(test_poi)
    self.tmu = tmu
    self.best_poi = best_poi
    self.comp_poi = comp_poi if comp_poi != None else self.test_poi
    self.tmu_A = tmu_A
    self.sigma = sigma

  def value(self) :
    return self.tmu if self.best_poi < self.comp_poi else -self.tmu

  def non_centrality_parameter(self) :
    if self.comp_poi == self.test_poi : return 0
    if self.tmu_A != None :
      if self.tmu_A < 0 :
        print('Warning: qmutilda tmu_A = % g < 0, returning 0' % self.tmu_A)
        return 0
      return self.tmu_A
    elif self.sigma != None :
      return ((self.comp_poi - self.test_poi)/self.sigma)**2
    else :
      raise ValueError('Should supply either tmu_A or sigma for the asymptotic CL_s computation')

  def asymptotic_cl(self) :
    if self.value() >= 0 :
      return scipy.stats.norm.sf(+math.sqrt(+self.value()) - math.sqrt(self.non_centrality_parameter()))
    else :
      return scipy.stats.norm.sf(-math.sqrt(-self.value()) - math.sqrt(self.non_centrality_parameter()))

  def asymptotic_tmu(self, cl) :
    q1 = scipy.stats.norm.isf(cl) + math.sqrt(self.non_centrality_parameter())
    return q1*abs(q1)

  def asymptotic_clb(self) :
    return QMu(0, self.tmu, self.best_poi, self.comp_poi, self.tmu_A, self.sigma).asymptotic_cl()

  def asymptotic_cls(self) :
    clsb = self.asymptotic_cl()
    cl_b = self.asymptotic_clb()
    #print('Asymptotic CLs = %g/%g = %g' % (clsb, cl_b, clsb/cl_b))
    return clsb/cl_b if cl_b > 0 else 1

class QMuTilda(TestStatistic) :
  def __init__(self, test_poi, tmu, best_poi, comp_poi = None, tmu_A = None, tmu_0 = None) :
    super().__init__(test_poi)
    self.tmu = tmu
    self.best_poi = best_poi
    self.comp_poi = comp_poi if comp_poi != None else self.test_poi
    self.tmu_A = tmu_A # corresponds to the test_poi hypo
    self.tmu_0 = tmu_0 # corresponds to the mu=0 hypo

  def value(self) :
    return self.tmu if self.best_poi < self.comp_poi else -self.tmu

  def non_centrality_parameter(self) : # (mu - mu')^2/sigma^2
    if self.comp_poi == self.test_poi : return 0
    if self.tmu_A < 0 :
      print('Warning: qmutilda tmu_A = % g < 0, returning 0' % self.tmu_A)
      return 0
    return self.tmu_A

  def threshold(self) : # mu^2/sigma^2
    if self.tmu_0 < 0 :
      print('Warning: qmutilda threshold = % g < 0, returning 0' % self.tmu_0)
      return 0
    return self.tmu_0

  def asymptotic_cl(self) :
    if self.value() < self.threshold() :
      if self.value() >= 0 :
        return scipy.stats.norm.sf(+math.sqrt(+self.value()) - math.sqrt(self.non_centrality_parameter()))
      else :
        return scipy.stats.norm.sf(-math.sqrt(-self.value()) - math.sqrt(self.non_centrality_parameter()))
    else :
      return scipy.stats.norm.sf((self.value() + self.threshold())/(2*math.sqrt(self.threshold())) - math.sqrt(self.non_centrality_parameter()))

  def asymptotic_tmu(self, cl) :
    q1 = scipy.stats.norm.isf(cl) + math.sqrt(self.non_centrality_parameter())
    if q1 < self.threshold() :
      return q1*abs(q1)
    else :
      return q1*2*math.sqrt(self.threshold()) - self.threshold()

  def asymptotic_clb(self) :
    return QMuTilda(0, self.tmu, self.best_poi, self.comp_poi, self.tmu_0, self.tmu_0).asymptotic_cl()

  def asymptotic_cls(self) :
    clsb = self.asymptotic_cl()
    cl_b = self.asymptotic_clb()
    #print('Asymptotic CLs = %g/%g = %g' % (clsb, cl_b, clsb/cl_b))
    return clsb/cl_b if cl_b > 0 else 1
